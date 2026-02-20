"""
GPU-accelerated Adaptive Patching for Weather/Climate ViT models.

Based on the GPU QuadTree Merger algorithm from bqdt-s8d-8k.py:
  - Bottom-up MERGE approach (start from finest grid, merge low-variance regions)
  - All statistics precomputed via tensor reshaping (no Python loops for stats)
  - GPU-vectorized batch merge using topk + sorted selection
  - Merge cost = parent_error - sum(child_errors), fully vectorized

Key idea: regions with high spatial variance (e.g., fronts, coastlines) keep
fine patches (more tokens), while smooth regions (e.g., open ocean) get merged
into larger patches (fewer tokens). The fixed-length output ensures a constant
sequence length for the ViT transformer.

The module provides:
  - GPUQuadTreeMerger: variance-based bottom-up merge, fully on GPU tensors
  - GPUPatchify: nn.Module wrapping the merger for use in ViT forward pass
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GPUQuadTreeMerger:
    """
    GPU-resident bottom-up quadtree merger for adaptive patching.

    Mirrors the approach from bqdt-s8d-8k.py but uses PyTorch tensors
    instead of CuPy, making it compatible with autograd and FSDP.

    Algorithm:
      1. Start with the finest grid (min_size patches) — all leaves alive
      2. Precompute SSE (sum of squared errors from mean) at every level
         using tensor reshaping — no Python loops
      3. Compute merge_cost[level] = parent_SSE - sum(4 child SSEs)
         Negative cost = merging REDUCES total error (or increases it less)
      4. Iteratively merge the lowest-cost groups of 4 until we reach
         the target number of leaves (fixed_length)

    Args:
        min_size: minimum patch side length in pixels (finest level)
    """

    def __init__(self, min_size=1):
        self.min_size = min_size

    @torch.no_grad()
    def _compute_level_stats(self, img, block_size):
        """
        Compute SSE for all blocks of a given size.

        Uses tensor reshaping — fully vectorized, no Python loops.
        Mirrors _compute_level_stats from bqdt-s8d-8k.py.

        Args:
            img: (C, H, W) image tensor on GPU
            block_size: side length of each block

        Returns:
            sse: (gh, gw) tensor — sum of squared errors per block
        """
        C, H, W = img.shape
        gh, gw = H // block_size, W // block_size

        # Reshape: (C, gh, bs, gw, bs) -> (C, gh, gw, bs, bs)
        reshaped = img.reshape(C, gh, block_size, gw, block_size)
        reshaped = reshaped.permute(0, 1, 3, 2, 4)  # (C, gh, gw, bs, bs)

        # Variance per block per channel, then sum across channels
        var = reshaped.var(dim=(3, 4))  # (C, gh, gw)
        # SSE = var * num_pixels (summed over channels)
        sse = var.sum(dim=0) * (block_size * block_size)  # (gh, gw)

        return sse

    @torch.no_grad()
    def precompute_all_levels(self, img):
        """
        Precompute SSE at all quadtree levels and merge costs between levels.

        This is the key performance optimization from the GPU reference:
        all statistics are computed via tensor reshaping in one pass per level,
        and merge costs are computed as vectorized differences.

        Args:
            img: (C, H, W) input tensor

        Returns:
            errors: list of (gh, gw) SSE tensors, one per level
            merge_costs: list (indexed by level), merge_costs[l] = parent - sum(children)
            level_shapes: list of (gh, gw) tuples
            max_level: highest level index
            padded_img: the (possibly padded) image tensor
        """
        C, H, W = img.shape

        # Compute max level: how many times we can double the block size
        max_blocks = min(H // self.min_size, W // self.min_size)
        max_level = int(math.floor(math.log2(max_blocks))) if max_blocks >= 1 else 0

        # Pad image so dimensions are divisible by the largest block size
        final_block = self.min_size * (2 ** max_level)
        pad_h = (final_block - (H % final_block)) % final_block
        pad_w = (final_block - (W % final_block)) % final_block
        if pad_h > 0 or pad_w > 0:
            img = F.pad(img, (0, pad_w, 0, pad_h), mode='replicate')

        errors = []
        level_shapes = []
        merge_costs = [None] * (max_level + 1)

        # Level 0 = finest (min_size blocks), level max_level = coarsest
        for l in range(max_level + 1):
            bs = self.min_size * (2 ** l)
            sse_l = self._compute_level_stats(img, bs)
            errors.append(sse_l)
            level_shapes.append(sse_l.shape)

        # Compute merge costs: cost of merging 4 children into 1 parent
        # merge_cost = parent_error - sum(4 children errors)
        # Low/negative cost = good merge (little information lost)
        # This mirrors the _precompute_all_levels loop in bqdt-s8d-8k.py
        for level in range(1, max_level + 1):
            child_err = errors[level - 1]  # (2*Hp, 2*Wp) at child level
            Hp_l, Wp_l = child_err.shape[0] // 2, child_err.shape[1] // 2

            # Reshape 4 children into groups and sum — vectorized
            c_reshaped = child_err.reshape(Hp_l, 2, Wp_l, 2)
            sum_children = c_reshaped.sum(dim=(1, 3))  # (Hp_l, Wp_l)

            parent_err = errors[level]  # (Hp_l, Wp_l)
            merge_costs[level] = parent_err - sum_children

        return errors, merge_costs, level_shapes, max_level, img

    @torch.no_grad()
    def build_tree(self, img, fixed_length):
        """
        Build adaptive quadtree by bottom-up merging.

        Mirrors run_merge from bqdt-s8d-8k.py:
          - Starts with all finest-level blocks alive
          - Collects merge candidates across all levels (GPU kernel launches)
          - Sorts by cost on GPU using topk
          - Applies batch merges by updating alive masks
          - Uses dynamic batch sizing (batch_ratio of current leaves)

        Args:
            img: (C, H, W) input tensor (single sample, on GPU)
            fixed_length: target number of leaf patches

        Returns:
            alive: list of boolean tensors per level indicating alive leaves
            level_shapes: grid shapes per level
            max_level: number of levels
            padded_img: the padded image tensor
        """
        errors, merge_costs, level_shapes, max_level, padded_img = \
            self.precompute_all_levels(img)

        device = img.device

        # Initialize alive arrays: all finest-level blocks are alive
        alive = [torch.zeros(shape, dtype=torch.bool, device=device)
                 for shape in level_shapes]
        alive[0][:] = True

        current_leaves = int(alive[0].sum())

        if current_leaves <= fixed_length:
            # Already at or below target — no merging needed
            return alive, level_shapes, max_level, padded_img

        # --- GPU-vectorized iterative merge (from bqdt-s8d-8k.py) ---
        # Dynamic batch strategy: merge ~15% of current leaves per iteration
        # As current_leaves decreases, batch_k automatically decreases too
        batch_ratio = 0.15
        min_batch = max(64, (current_leaves - fixed_length) // 20)
        max_batch = 100000

        while current_leaves > fixed_length:
            # Dynamic batch size (mirrors bqdt-s8d-8k.py logic)
            target_batch = int(current_leaves * batch_ratio)
            batch_k = max(min_batch, min(target_batch, max_batch))

            # If close to target, try to finish in one pass
            if (current_leaves - fixed_length) < batch_k:
                batch_k = max(min_batch, current_leaves - fixed_length + 100)

            candidates_cost = []
            candidates_meta = []

            # 1. Collect merge candidates across all levels (GPU operations)
            for level in range(1, max_level + 1):
                Hp, Wp = level_shapes[level]

                # Check if all 4 children are alive — vectorized
                child_alive_view = alive[level - 1].reshape(Hp, 2, Wp, 2)
                can_merge = child_alive_view.all(dim=(1, 3))  # (Hp, Wp)

                if not can_merge.any():
                    continue

                costs = merge_costs[level]

                # Get valid indices (GPU tensors)
                valid_indices = torch.where(can_merge)
                valid_costs = costs[valid_indices]
                num_valid = valid_costs.shape[0]

                candidates_cost.append(valid_costs)

                # Build metadata: [level, row, col] on GPU
                level_arr = torch.full(
                    (num_valid,), level, dtype=torch.int32, device=device
                )
                meta = torch.stack(
                    [level_arr, valid_indices[0].int(), valid_indices[1].int()],
                    dim=1,
                )
                candidates_meta.append(meta)

            if not candidates_cost:
                break  # No more merges possible

            # 2. GPU sort & select top-k lowest cost merges
            all_costs = torch.cat(candidates_cost)
            all_meta = torch.cat(candidates_meta)

            k = int(min(batch_k, all_costs.shape[0]))

            # topk with largest=False = select k smallest costs
            # This mirrors the argpartition + argsort from bqdt-s8d-8k.py
            _, top_k_indices = torch.topk(all_costs, k, largest=False, sorted=False)

            # Sort selected k for stable ordering
            subset_costs = all_costs[top_k_indices]
            sorted_local_idx = subset_costs.argsort()
            final_indices = top_k_indices[sorted_local_idx]
            selected_meta = all_meta[final_indices]

            # 3. Apply merges — update alive masks (GPU indexing)
            merges_count = 0

            for lvl in range(1, max_level + 1):
                mask = (selected_meta[:, 0] == lvl)
                rows = selected_meta[mask, 1].long()
                cols = selected_meta[mask, 2].long()

                if rows.numel() == 0:
                    continue

                # Mark parent as alive
                alive[lvl][rows, cols] = True

                # Kill 4 children — mirrors bqdt-s8d-8k.py exactly
                r2 = rows * 2
                c2 = cols * 2
                alive[lvl - 1][r2, c2] = False
                alive[lvl - 1][r2 + 1, c2] = False
                alive[lvl - 1][r2, c2 + 1] = False
                alive[lvl - 1][r2 + 1, c2 + 1] = False

                merges_count += rows.shape[0]

            # Each merge: 4 children -> 1 parent = net -3 leaves
            current_leaves -= 3 * merges_count

        return alive, level_shapes, max_level, padded_img

    @torch.no_grad()
    def alive_to_leaves(self, alive, level_shapes, max_level):
        """
        Convert alive boolean masks to a list of (r0, c0, r1, c1) leaf boundaries.

        Args:
            alive: list of boolean tensors per level
            level_shapes: grid shapes per level
            max_level: highest level

        Returns:
            leaves: list of (r0, c0, r1, c1) tuples in pixel coordinates
        """
        leaves = []
        for level in range(max_level + 1):
            bs = self.min_size * (2 ** level)
            coords = torch.where(alive[level])
            if coords[0].numel() == 0:
                continue
            rows = coords[0].cpu()
            cols = coords[1].cpu()
            for r, c in zip(rows.tolist(), cols.tolist()):
                r0, c0 = r * bs, c * bs
                leaves.append((r0, c0, r0 + bs, c0 + bs))
        return leaves

    @torch.no_grad()
    def serialize(self, x, leaves, target_size):
        """
        Extract patches from x according to leaf boundaries and resize to uniform size.

        Args:
            x: (C, H, W) tensor — single image
            leaves: list of (r0, c0, r1, c1) leaf boundaries
            target_size: (ph, pw) target patch size for uniform tokens

        Returns:
            patches: (num_leaves, C, ph, pw)
        """
        C = x.shape[0]
        ph, pw = target_size
        patches = []

        for r0, c0, r1, c1 in leaves:
            patch = x[:, r0:r1, c0:c1]  # (C, h, w) — variable size
            if patch.shape[1] != ph or patch.shape[2] != pw:
                patch = F.interpolate(
                    patch.unsqueeze(0),
                    size=(ph, pw),
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(0)
            patches.append(patch)

        return torch.stack(patches, dim=0)  # (N, C, ph, pw)

    @torch.no_grad()
    def deserialize(self, patches, leaves, output_size):
        """
        Reconstruct a full image from adaptive patches by placing them back.

        Args:
            patches: (num_leaves, C, ph, pw) decoded patches
            leaves: list of (r0, c0, r1, c1) leaf boundaries
            output_size: (H, W) target output size

        Returns:
            img: (C, H, W) reconstructed image
        """
        C = patches.shape[1]
        H, W = output_size
        img = torch.zeros(C, H, W, device=patches.device, dtype=patches.dtype)
        count = torch.zeros(1, H, W, device=patches.device, dtype=patches.dtype)

        for i, (r0, c0, r1, c1) in enumerate(leaves):
            h, w = r1 - r0, c1 - c0
            patch = patches[i]
            if patch.shape[1] != h or patch.shape[2] != w:
                patch = F.interpolate(
                    patch.unsqueeze(0),
                    size=(h, w),
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(0)
            img[:, r0:r1, c0:c1] += patch
            count[:, r0:r1, c0:c1] += 1.0

        count = count.clamp(min=1.0)
        return img / count


class GPUPatchify(nn.Module):
    """
    GPU-accelerated adaptive patchification module for weather/climate ViT.

    Uses GPUQuadTreeMerger (bottom-up, GPU-vectorized) — NOT the naive
    heap-based split baseline. Key techniques from bqdt-s8d-8k.py:
      1. Precompute all level SSEs via tensor reshaping (vectorized, no loops)
      2. Compute merge costs as parent_error - sum(child_errors)
      3. GPU topk to select lowest-cost merges in batch
      4. Dynamic batch sizing (15% of current leaves per iteration)

    Args:
        fixed_length: number of output tokens (patches)
        patch_size: target patch size for each token
        num_channels: number of input channels
        min_size: finest block size for the quadtree (default 1)
    """

    def __init__(self, fixed_length=256, patch_size=2, num_channels=1, min_size=1):
        super().__init__()
        self.fixed_length = fixed_length
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.quadtree = GPUQuadTreeMerger(min_size=min_size)

    def compute_importance(self, x):
        """
        Compute per-pixel importance using channel-wise variance.

        Args:
            x: (B, C, H, W)

        Returns:
            importance: (B, H, W)
        """
        return x.var(dim=1)

    def forward(self, x):
        """
        Adaptively patchify a batch of images using GPU-vectorized bottom-up merge.

        Args:
            x: (B, C, H, W) input tensor (on GPU)

        Returns:
            patches: (B, fixed_length, C, patch_size, patch_size)
            leaves_batch: list of leaf boundary lists per batch element
        """
        B, C, H, W = x.shape

        patches_list = []
        leaves_batch = []

        for b in range(B):
            # Build quadtree via bottom-up merge (GPU-vectorized)
            alive, level_shapes, max_level, padded_img = \
                self.quadtree.build_tree(x[b], self.fixed_length)

            # Convert alive masks to leaf boundary list
            leaves = self.quadtree.alive_to_leaves(alive, level_shapes, max_level)
            leaves_batch.append(leaves)

            # Serialize: extract + resize patches to uniform size
            # Use padded_img to match coordinates from alive_to_leaves
            sample_patches = self.quadtree.serialize(
                padded_img, leaves, (self.patch_size, self.patch_size)
            )  # (num_leaves, C, ps, ps)

            # Pad or truncate to exactly fixed_length
            n = sample_patches.shape[0]
            if n < self.fixed_length:
                pad = torch.zeros(
                    self.fixed_length - n, C, self.patch_size, self.patch_size,
                    device=x.device, dtype=x.dtype,
                )
                sample_patches = torch.cat([sample_patches, pad], dim=0)
            elif n > self.fixed_length:
                sample_patches = sample_patches[:self.fixed_length]

            patches_list.append(sample_patches)

        patches = torch.stack(patches_list, dim=0)  # (B, N, C, ps, ps)
        return patches, leaves_batch

    def deserialize_batch(self, patches, leaves_batch, output_size):
        """
        Reconstruct full images from adaptive patches.

        Args:
            patches: (B, N, C, ph, pw)
            leaves_batch: list of leaf boundary lists
            output_size: (H, W)

        Returns:
            imgs: (B, C, H, W)
        """
        B = patches.shape[0]
        imgs = []
        for b in range(B):
            img = self.quadtree.deserialize(
                patches[b], leaves_batch[b], output_size
            )
            imgs.append(img)
        return torch.stack(imgs, dim=0)

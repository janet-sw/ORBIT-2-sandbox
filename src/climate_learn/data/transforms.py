import torch
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate
from typing import Tuple, List
import os

def _downsample_cpu(x: torch.Tensor, size: Tuple[int, int], mode: str = "area") -> torch.Tensor:
    """Resize on CPU. Supports [B,C,H,W], [B,T,C,H,W], [T,C,H,W], [C,H,W]."""
    antialias = mode in ("bilinear", "bicubic")
    if x.ndim == 5:  # [B,T,C,H,W] -> fold T into batch
        B, T, C, H, W = x.shape
        x = x.reshape(B * T, C, H, W)
        x = F.interpolate(
            x, size=size,
            mode=("area" if mode == "area" else mode),
            align_corners=False if mode != "area" else None,
            antialias=(antialias if mode != "area" else False),
        )
        return x.reshape(B, T, C, *size)
    if x.ndim == 4:  # [B,C,H,W]
        return F.interpolate(
            x, size=size,
            mode=("area" if mode == "area" else mode),
            align_corners=False if mode != "area" else None,
            antialias=(antialias if mode != "area" else False),
        )
    if x.ndim == 3:  # [C,H,W]
        return _downsample_cpu(x.unsqueeze(0), size, mode).squeeze(0)
    if x.ndim == 2:  # [H,W]
        return _downsample_cpu(x[None, None], size, mode)[0, 0]
    return x

def collate_resize(samples, lr_size=(32,64), mode="area", hr_size=None):
    # (Keep your existing collate_resize code here if needed)
    return default_collate(samples)

def collate_batch_only(samples):
    """
    Robust custom collate that handles uneven tile sizes by padding.
    Safe for 2D and 3D tensors.
    """
    # DEBUG: Print once to verify this code is actually running
    if not hasattr(collate_batch_only, '_printed'):
        print("\n[DEBUG] Using custom PADDED collate_batch_only!\n", flush=True)
        collate_batch_only._printed = True

    first = samples[0]
    
    # Check if samples are tuples with 4 elements (from IterDataset)
    if isinstance(first, (tuple, list)) and len(first) == 4:
        inp_datas, out_datas, variables, out_variables = [], [], [], []
        for inp_data, out_data, vars, out_vars in samples:
            inp_datas.append(inp_data)
            out_datas.append(out_data)
            variables.append(vars)
            out_variables.append(out_vars)
        
        variables = variables[0]
        out_variables = out_variables[0]
        
        # --- PROCESS INPUTS ---
        inp_list = []
        for var in variables:
            tensors = [inp[var] for inp in inp_datas]
            
            # 1. Find max dimensions
            max_h = max(t.shape[-2] for t in tensors)
            max_w = max(t.shape[-1] for t in tensors)
            
            # 2. Pad tensors safely
            padded = []
            for t in tensors:
                pad_h = max_h - t.shape[-2]
                pad_w = max_w - t.shape[-1]
                
                if pad_h > 0 or pad_w > 0:
                    # Fix for Replicate Crash: Ensure tensor is at least 3D for replicate padding
                    original_ndim = t.ndim
                    if original_ndim == 2: # [H, W] -> [1, H, W]
                        t = t.unsqueeze(0)
                    
                    # Pad (Left, Right, Top, Bottom)
                    t = F.pad(t, (0, pad_w, 0, pad_h), mode='replicate')
                    
                    if original_ndim == 2: # Restore to [H, W]
                        t = t.squeeze(0)
                
                padded.append(t)
                
            stacked = torch.stack(padded, dim=0)  # [B, T, H, W]
            inp_list.append(stacked)
        
        # --- PROCESS OUTPUTS ---
        out_list = []
        for var in out_variables:
            tensors = [out[var] for out in out_datas]
            
            # 1. Find max dimensions
            max_h = max(t.shape[-2] for t in tensors)
            max_w = max(t.shape[-1] for t in tensors)
            
            # 2. Pad tensors safely
            padded = []
            for t in tensors:
                pad_h = max_h - t.shape[-2]
                pad_w = max_w - t.shape[-1]
                
                if pad_h > 0 or pad_w > 0:
                    # Fix for Replicate Crash: Output targets are often 2D [H, W]
                    original_ndim = t.ndim
                    if original_ndim == 2:
                        t = t.unsqueeze(0)
                        
                    t = F.pad(t, (0, pad_w, 0, pad_h), mode='replicate')
                    
                    if original_ndim == 2:
                        t = t.squeeze(0)
                        
                padded.append(t)
                
            stacked = torch.stack(padded, dim=0)  # [B, H, W]
            stacked = stacked.unsqueeze(1)        # [B, 1, H, W]
            out_list.append(stacked)
        
        # Concatenate
        inp_batch = torch.cat(inp_list, dim=1)
        out_batch = torch.cat(out_list, dim=1)
        
        return inp_batch, out_batch, variables, out_variables
    else:
        return default_collate(samples)
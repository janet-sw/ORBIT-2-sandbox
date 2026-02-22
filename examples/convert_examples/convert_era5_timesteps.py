#!/usr/bin/env python3
"""
Convert ERA5 monthly .npz files → per-timestep .npy files.

Each output file contains ALL 73 variables for ONE timestep as a single
float16 array of shape (73, 180, 360). At train time, the dataloader
picks any (t, t+lead) pair dynamically — no need to fix lead time at
preprocessing time.

Usage:
    python convert_era5_timesteps.py \
        --input_dir  /lustre/.../ERA5-1hr-superres/1.0_deg/ \
        --output_dir /lustre/.../era5_1.0_deg_timesteps/ \
        --num_workers 16

Output structure:
    output_dir/
    ├── train/
    │   ├── 2017_09_000.npy   (timestep 0 of Sept 2017)
    │   ├── 2017_09_001.npy   (timestep 1 of Sept 2017)
    │   └── ...
    ├── val/
    │   └── ...
    ├── var_names.json         (ordered list of 73 variable names)
    └── metadata.json

Then use ERA5TimestepDataset (included below) as a drop-in MapDataset
with standard PyTorch DataLoader(num_workers=4, shuffle=True).
"""

import argparse
import glob
import json
import os
import numpy as np
import time
from multiprocessing import Pool

# These 2 keys are scalars, not spatial fields — skip them
SKIP_KEYS = {"hrs_each_step", "num_steps_per_shard"}


def get_var_names(npz_path):
    """Extract ordered variable names from a sample .npz file."""
    npz = np.load(npz_path)
    var_names = [k for k in npz.files if k not in SKIP_KEYS]
    var_names.sort()  # deterministic order
    return var_names


def process_single_file(args):
    """Convert one monthly .npz → multiple per-timestep .npy files.
    
    Each output is shape (C, H, W) in float16 where C = number of variables.
    """
    filepath, output_dir, var_names = args
    basename = os.path.basename(filepath)
    
    # Parse year_month from filename (e.g., "2017_9.npz" → "2017_09")
    stem = os.path.splitext(basename)[0]  # "2017_9"
    parts = stem.split("_")
    if len(parts) == 2:
        year, month = parts[0], parts[1].zfill(2)
        prefix = f"{year}_{month}"
    else:
        prefix = stem
    
    try:
        npz = np.load(filepath)
    except Exception as e:
        print(f"  ERROR loading {basename}: {e}", flush=True)
        return (0, basename)
    
    # Get number of timesteps from first variable
    first_var = var_names[0]
    T = npz[first_var].shape[0]
    C = len(var_names)
    
    # Stack all variables: (T, C, H, W)
    # Each var is (T, 1, H, W) — squeeze the channel dim
    try:
        arrays = []
        for var in var_names:
            arr = npz[var]  # (T, 1, 180, 360)
            if arr.ndim == 4 and arr.shape[1] == 1:
                arr = arr[:, 0, :, :]  # (T, 180, 360)
            arrays.append(arr)
        
        # Stack to (T, C, H, W) then convert to float16
        stacked = np.stack(arrays, axis=1).astype(np.float16)  # (T, C, H, W)
    except Exception as e:
        print(f"  ERROR stacking {basename}: {e}", flush=True)
        return (0, basename)
    
    # Save each timestep as a separate .npy file
    count = 0
    for t in range(T):
        out_path = os.path.join(output_dir, f"{prefix}_{t:03d}.npy")
        np.save(out_path, stacked[t])  # (C, H, W) float16, ~9.5 MB
        count += 1
    
    del stacked, arrays
    return (count, basename)


def convert_split(input_split_dir, output_split_dir, var_names, num_workers=8):
    """Convert all .npz files in one split (train or val)."""
    
    os.makedirs(output_split_dir, exist_ok=True)
    
    all_files = sorted(glob.glob(os.path.join(input_split_dir, "*.npz")))
    if not all_files:
        print(f"  No .npz files found in {input_split_dir}", flush=True)
        return 0
    
    print(f"  Found {len(all_files)} monthly files", flush=True)
    
    # Build args
    work_args = [(f, output_split_dir, var_names) for f in all_files]
    
    t0 = time.time()
    total_steps = 0
    
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_single_file, work_args)
    
    for n_steps, basename in results:
        total_steps += n_steps
        if n_steps > 0:
            pass  # quiet success
        else:
            print(f"  WARNING: 0 steps from {basename}", flush=True)
    
    elapsed = time.time() - t0
    print(f"  Extracted {total_steps:,} timesteps from {len(all_files)} files "
          f"in {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)
    
    return total_steps


def main():
    parser = argparse.ArgumentParser(
        description="Convert ERA5 monthly .npz → per-timestep .npy files")
    parser.add_argument("--input_dir", required=True,
                        help="Root ERA5 directory with train/ and val/ subfolders")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for converted timestep files")
    parser.add_argument("--num_workers", type=int, default=16,
                        help="Parallel workers (default: 16)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("ERA5 Monthly NPZ -> Per-Timestep NPY Converter")
    print("=" * 60)
    print(f"Input:   {args.input_dir}")
    print(f"Output:  {args.output_dir}")
    print(f"Workers: {args.num_workers}")
    print()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Discover variable names from first available file
    for split in ["test"]:
        split_dir = os.path.join(args.input_dir, split)
        files = sorted(glob.glob(os.path.join(split_dir, "*.npz")))
        if files:
            var_names = get_var_names(files[0])
            break
    else:
        raise RuntimeError("No .npz files found in train/ or val/")
    
    print(f"Variables ({len(var_names)}):")
    for i, v in enumerate(var_names):
        print(f"  [{i:2d}] {v}")
    print()
    
    # Save variable names for the dataloader
    var_path = os.path.join(args.output_dir, "var_names.json")
    with open(var_path, "w") as f:
        json.dump(var_names, f, indent=2)
    
    metadata = {
        "n_vars": len(var_names),
        "spatial_shape": [180, 360],
        "dtype": "float16",
        "source_dir": args.input_dir,
        "splits": {},
    }
    
    total_all = 0
    t_start = time.time()
    
    for split in ["test"]:
        input_split = os.path.join(args.input_dir, split)
        output_split = os.path.join(args.output_dir, split)
        
        if not os.path.isdir(input_split):
            print(f"Skipping {split}/ (not found)")
            continue
        
        print(f"\nProcessing {split}/...")
        n_steps = convert_split(input_split, output_split, var_names,
                                num_workers=args.num_workers)
        metadata["splits"][split] = n_steps
        total_all += n_steps
    
    # Save metadata
    meta_path = os.path.join(args.output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    elapsed = time.time() - t_start
    
    print()
    print("=" * 60)
    print(f"DONE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Total timesteps: {total_all:,}")
    print(f"Variables: {len(var_names)}")
    print(f"Per-file size: ~{len(var_names) * 180 * 360 * 2 / 1e6:.1f} MB")
    print(f"Metadata: {meta_path}")
    print(f"Var names: {var_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
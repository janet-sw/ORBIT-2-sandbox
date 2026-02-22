#!/usr/bin/env python3
"""
Convert monthly .npz files to per-sample .npz files for fast random-access loading.

Parallelized with multiprocessing for speed. On Frontier with 8 workers,
converts ~4000 files (6.7 TB) in ~20-30 minutes.

Usage:
    python convert_npz_to_samples.py \
        --input_dir /lustre/.../ERA5-1hr-superres/1.0_deg/ \
        --output_dir /lustre/.../era5_1.0_deg_samples/

Output structure:
    output_dir/
    ├── train/
    │   ├── sample_000000.npz
    │   ├── sample_000001.npz
    │   └── ...
    ├── val/
    │   ├── sample_000000.npz
    │   └── ...
    └── metadata.json
"""

import argparse
import glob
import json
import os
import numpy as np
import time
from multiprocessing import Pool


# ─── Configuration (matching era5_forecasting.yaml) ───
IN_VARS = ['land_sea_mask', 'orography', '2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind', 'mean_sea_level_pressure', 'surface_pressure', 'total_precipitation', 'sea_surface_temperature', 'geopotential_50', 'geopotential_100', 'geopotential_150', 'geopotential_200', 'geopotential_250', 'geopotential_300', 'geopotential_400', 'geopotential_500', 'geopotential_600', 'geopotential_700', 'geopotential_850', 'geopotential_925', 'geopotential_1000', 'u_component_of_wind_50', 'u_component_of_wind_100', 'u_component_of_wind_150', 'u_component_of_wind_200', 'u_component_of_wind_250', 'u_component_of_wind_300', 'u_component_of_wind_400', 'u_component_of_wind_500', 'u_component_of_wind_600', 'u_component_of_wind_700', 'u_component_of_wind_850', 'u_component_of_wind_925', 'u_component_of_wind_1000', 'v_component_of_wind_50', 'v_component_of_wind_100', 'v_component_of_wind_150', 'v_component_of_wind_200', 'v_component_of_wind_250', 'v_component_of_wind_300', 'v_component_of_wind_400', 'v_component_of_wind_500', 'v_component_of_wind_600', 'v_component_of_wind_700', 'v_component_of_wind_850', 'v_component_of_wind_925', 'v_component_of_wind_1000', 'temperature_50', 'temperature_100', 'temperature_150', 'temperature_200', 'temperature_250', 'temperature_300', 'temperature_400', 'temperature_500', 'temperature_600', 'temperature_700', 'temperature_850', 'temperature_925', 'temperature_1000', 'specific_humidity_50', 'specific_humidity_100', 'specific_humidity_150', 'specific_humidity_200', 'specific_humidity_250', 'specific_humidity_300', 'specific_humidity_400', 'specific_humidity_500', 'specific_humidity_600', 'specific_humidity_700', 'specific_humidity_850', 'specific_humidity_925', 'specific_humidity_1000', 'days_of_year', 'time_of_day', 'hrs_each_step', 'num_steps_per_shard', 'lattitude']
OUT_VARS = []
PRED_RANGE = 120
SUBSAMPLE = 6
HISTORY = 1


def process_single_file(args):
    """Process one .npz file → multiple per-sample .npz files.
    
    Designed to be called by multiprocessing.Pool.map().
    Returns (n_samples_extracted, filepath) for logging.
    """
    filepath, output_dir, start_idx, save_dtype = args
    basename = os.path.basename(filepath)
    
    all_vars = set(IN_VARS + OUT_VARS)
    
    try:
        npz = np.load(filepath)
    except Exception as e:
        print(f"  ERROR loading {basename}: {e}", flush=True)
        return (0, basename)
    
    # Load required variables
    data = {}
    for var in all_vars:
        if var not in npz:
            print(f"  ERROR: '{var}' not found in {basename}, skipping", flush=True)
            return (0, basename)
        arr = npz[var]
        if arr.ndim == 4 and arr.shape[1] == 1:
            arr = arr.squeeze(1)
        data[var] = arr
    
    # Get valid time indices
    first_var = IN_VARS[0]
    T_total = data[first_var].shape[0]
    indices = list(range(HISTORY, T_total - PRED_RANGE, SUBSAMPLE))
    
    sample_count = 0
    for t in indices:
        try:
            # Build input: [C, H, W]
            x_list = []
            for var in IN_VARS:
                channel = data[var][t - HISTORY : t]  # [history, H, W]
                x_list.append(channel)
            x = np.concatenate(x_list, axis=0).astype(save_dtype)
            
            # Build output: [C_out, H, W]
            y_list = []
            for var in OUT_VARS:
                target = data[var][t + PRED_RANGE - 1]  # [H, W]
                y_list.append(target[np.newaxis])
            y = np.concatenate(y_list, axis=0).astype(save_dtype)
            
            # Skip NaN samples
            if np.isnan(x).any() or np.isnan(y).any():
                continue
            
            out_path = os.path.join(output_dir, f"sample_{start_idx + sample_count:06d}.npz")
            np.savez(out_path, x=x, y=y)  # uncompressed for fast loading
            sample_count += 1
            
        except Exception as e:
            print(f"  Error at t={t} in {basename}: {e}", flush=True)
            continue
    
    del data
    return (sample_count, basename)


def convert_split(input_split_dir, output_split_dir, num_workers=8, save_dtype=np.float16):
    """Convert all .npz files in one split (train or val)."""
    
    os.makedirs(output_split_dir, exist_ok=True)
    
    all_files = sorted(glob.glob(os.path.join(input_split_dir, "*.npz")))
    if not all_files:
        print(f"  No .npz files found in {input_split_dir}", flush=True)
        return 0
    
    print(f"  Found {len(all_files)} files", flush=True)
    
    # Pre-calculate start indices for each file so samples don't collide
    # Each file produces ~52 samples, but we allocate 100 slots per file for safety
    SLOTS_PER_FILE = 100
    
    # Build args for parallel processing
    work_args = []
    for i, filepath in enumerate(all_files):
        start_idx = i * SLOTS_PER_FILE
        work_args.append((filepath, output_split_dir, start_idx, save_dtype))
    
    # Process in parallel
    t0 = time.time()
    total_samples = 0
    
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_single_file, work_args)
    
    for n_samples, basename in results:
        total_samples += n_samples
    
    elapsed = time.time() - t0
    print(f"  Extracted {total_samples} samples from {len(all_files)} files "
          f"in {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)
    
    # Rename files to be contiguous (remove gaps from SLOTS_PER_FILE allocation)
    print(f"  Renaming to contiguous indices...", flush=True)
    all_samples = sorted(glob.glob(os.path.join(output_split_dir, "sample_*.npz")))
    for new_idx, old_path in enumerate(all_samples):
        new_path = os.path.join(output_split_dir, f"sample_{new_idx:06d}.npz")
        if old_path != new_path:
            os.rename(old_path, new_path)
    
    print(f"  Final: {len(all_samples)} contiguous samples", flush=True)
    return len(all_samples)


def main():
    parser = argparse.ArgumentParser(description="Convert monthly .npz to per-sample files")
    parser.add_argument("--input_dir", required=True,
                        help="Root ERA5 directory containing train/ and val/ subfolders")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for converted samples")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Number of parallel workers (default: 8)")
    parser.add_argument("--dtype", default="float16",
                        choices=["float32", "float16"],
                        help="Save dtype (float16 saves space, loaded as bf16 at train time)")
    args = parser.parse_args()
    
    save_dtype = np.float16 if args.dtype == "float16" else np.float32
    
    print("=" * 60)
    print("ERA5 Monthly NPZ -> Per-Sample Converter")
    print("=" * 60)
    print(f"Input:      {args.input_dir}")
    print(f"Output:     {args.output_dir}")
    print(f"Workers:    {args.num_workers}")
    print(f"Save dtype: {args.dtype}")
    print(f"In vars:    {IN_VARS}")
    print(f"Out vars:   {OUT_VARS}")
    print(f"pred_range: {PRED_RANGE}, subsample: {SUBSAMPLE}")
    print()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    metadata = {
        "in_vars": IN_VARS,
        "out_vars": OUT_VARS,
        "pred_range": PRED_RANGE,
        "subsample": SUBSAMPLE,
        "history": HISTORY,
        "save_dtype": args.dtype,
        "source_dir": args.input_dir,
        "splits": {},
    }
    
    total_all = 0
    t_start = time.time()
    
    for split in ["train", "val"]:
        input_split = os.path.join(args.input_dir, split)
        output_split = os.path.join(args.output_dir, split)
        
        if not os.path.isdir(input_split):
            print(f"Skipping {split}/ (not found at {input_split})")
            continue
        
        print(f"\nProcessing {split}/...")
        n_samples = convert_split(input_split, output_split, 
                                   num_workers=args.num_workers,
                                   save_dtype=save_dtype)
        metadata["splits"][split] = n_samples
        total_all += n_samples
    
    # Save metadata
    meta_path = os.path.join(args.output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    elapsed = time.time() - t_start
    
    # Check output size
    total_bytes = 0
    for split in ["train", "val"]:
        split_dir = os.path.join(args.output_dir, split)
        if os.path.isdir(split_dir):
            for f in os.listdir(split_dir):
                total_bytes += os.path.getsize(os.path.join(split_dir, f))
    
    print()
    print("=" * 60)
    print(f"DONE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Total samples: {total_all:,}")
    print(f"Output size: {total_bytes / 1e9:.1f} GB")
    print(f"Metadata: {meta_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
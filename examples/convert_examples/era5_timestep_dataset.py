"""
ERA5TimestepDataset — drop-in MapDataset for per-timestep .npy files.

Works with standard PyTorch DataLoader(num_workers=4, shuffle=True).
Dynamically selects (input, target) pairs for any lead time at train time.

Usage:
    from era5_timestep_dataset import ERA5TimestepDataset

    dataset = ERA5TimestepDataset(
        data_dir="/lustre/.../era5_1.0_deg_timesteps/train",
        var_names_path="/lustre/.../era5_1.0_deg_timesteps/var_names.json",
        in_vars=["temperature_850", "2m_temperature", "10m_u_component_of_wind", ...],
        out_vars=["2m_temperature"],
        pred_range=120,   # lead time in timesteps (each step = hrs_each_step)
        history=1,        # number of input frames
    )
    loader = DataLoader(dataset, batch_size=32, num_workers=4, shuffle=True)
    
    for x, y in loader:
        # x: (B, C_in * history, H, W)
        # y: (B, C_out, H, W)
"""

import json
import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict


class ERA5TimestepDataset(Dataset):
    """Map-style dataset over per-timestep .npy files.
    
    File naming convention: {year}_{month}_{timestep:03d}.npy
    Each file: (C, H, W) float16 array with all variables.
    
    The dataset groups files by (year, month), then generates valid
    (input, target) index pairs based on pred_range and history.
    """
    
    def __init__(self, data_dir, var_names_path, in_vars, out_vars,
                 pred_range=120, history=1):
        """
        Args:
            data_dir:       Path to split directory (e.g., .../train/)
            var_names_path: Path to var_names.json (channel ordering)
            in_vars:        List of input variable names
            out_vars:       List of output variable names  
            pred_range:     Lead time in timesteps
            history:        Number of input timesteps (default: 1)
        """
        self.data_dir = data_dir
        self.pred_range = pred_range
        self.history = history
        
        # Load variable name ordering
        with open(var_names_path) as f:
            all_var_names = json.load(f)
        
        # Build channel index maps
        var_to_idx = {v: i for i, v in enumerate(all_var_names)}
        self.in_indices = [var_to_idx[v] for v in in_vars]
        self.out_indices = [var_to_idx[v] for v in out_vars]
        
        # Discover and group files by (year, month)
        # Filename pattern: 2017_09_042.npy
        all_files = sorted(os.listdir(data_dir))
        npy_files = [f for f in all_files if f.endswith(".npy")]
        
        # Group by year_month prefix
        groups = defaultdict(list)
        pattern = re.compile(r"^(\d{4}_\d{2})_(\d{3})\.npy$")
        for fname in npy_files:
            m = pattern.match(fname)
            if m:
                prefix = m.group(1)
                step = int(m.group(2))
                groups[prefix].append((step, fname))
        
        # For each group, sort by timestep and generate valid pairs
        self.samples = []  # list of (input_paths, target_path)
        
        for prefix in sorted(groups.keys()):
            steps = sorted(groups[prefix], key=lambda x: x[0])
            step_to_file = {s: f for s, f in steps}
            max_step = max(s for s, _ in steps)
            
            for step, fname in steps:
                # Need history-1 steps before and pred_range steps after
                input_start = step - (history - 1)
                target_step = step + pred_range
                
                if input_start < 0:
                    continue
                if target_step > max_step:
                    continue
                
                # Check all required timesteps exist
                input_steps = list(range(input_start, step + 1))
                if not all(s in step_to_file for s in input_steps):
                    continue
                if target_step not in step_to_file:
                    continue
                
                input_paths = [os.path.join(data_dir, step_to_file[s]) 
                               for s in input_steps]
                target_path = os.path.join(data_dir, step_to_file[target_step])
                self.samples.append((input_paths, target_path))
        
        print(f"ERA5TimestepDataset: {len(self.samples)} samples "
              f"from {len(groups)} months, pred_range={pred_range}, "
              f"history={history}, in_vars={len(in_vars)}, out_vars={len(out_vars)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        input_paths, target_path = self.samples[idx]
        
        # Load input frames and select channels
        x_frames = []
        for path in input_paths:
            arr = np.load(path)  # (C_all, H, W) float16
            x_frames.append(arr[self.in_indices])  # (C_in, H, W)
        x = np.concatenate(x_frames, axis=0)  # (C_in * history, H, W)
        
        # Load target and select channels
        y = np.load(target_path)[self.out_indices]  # (C_out, H, W)
        
        return (
            torch.from_numpy(x.copy()).to(torch.bfloat16),
            torch.from_numpy(y.copy()).to(torch.bfloat16),
        )
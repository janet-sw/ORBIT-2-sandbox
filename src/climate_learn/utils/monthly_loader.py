import torch
import numpy as np
import glob
import os
import random
import torch.distributed as dist
from torch.utils.data import IterableDataset

class SequentialMonthlyDataset(IterableDataset):
    def __init__(self, 
                 root_dir, 
                 in_vars, 
                 out_vars, 
                 pred_range=120, 
                 subsample=6,
                 transform=None,
                 rank=0,
                 world_size=1,
                 dtype=torch.float32): # Add dtype support
        """
        Reads .npz files sequentially. 
        Shards files based on rank to ensure unique data per GPU.
        """
        self.in_vars = in_vars
        self.out_vars = out_vars
        self.history = 1 # Hardcoded as per your config
        self.pred_range = pred_range
        self.subsample = subsample
        self.transform = transform
        self.dtype = dtype
        self.rank = rank
        
        # 1. Get all files
        all_files = sorted(glob.glob(os.path.join(root_dir, "*.npz")))
        
        if len(all_files) == 0:
            raise ValueError(f"No .npz files found in {root_dir}")
            
        # 2. Shard files (Distribute among GPUs)
        # Rank 0 gets [0, 8, 16...], Rank 1 gets [1, 9, 17...]
        self.files = all_files[rank::world_size]
        
        print(f"[Rank {rank}] Assigned {len(self.files)}/{len(all_files)} monthly files.", flush=True)

    def _load_month_to_ram(self, filepath):
        try:
            # print(f"[Rank {self.rank}] Loading {os.path.basename(filepath)}...", flush=True)
            npz = np.load(filepath) 
            data = {}
            all_vars = list(set(self.in_vars + self.out_vars))
            
            for var in all_vars:
                if var in npz:
                    # LOAD AND CAST IMMEDIATELY to save RAM
                    # Copy makes it a torch tensor, then cast to reduce size
                    tensor = torch.from_numpy(npz[var])
                    if self.dtype != torch.float32:
                        tensor = tensor.to(self.dtype)
                    data[var] = tensor
                else:
                    print(f"Warning: {var} not found in {filepath}")
            return data
        except Exception as e:
            print(f"Failed to load {filepath}: {e}")
            return None

    def __iter__(self):
        # Shuffle order of months for this rank
        random.shuffle(self.files)

        for filepath in self.files:
            data_dict = self._load_month_to_ram(filepath)
            if data_dict is None: continue
            
            # Determine time dimension from first variable
            first_var = next(iter(data_dict.values()))
            T_total = first_var.shape[0]
            
            indices = list(range(self.history, T_total - self.pred_range, self.subsample))
            random.shuffle(indices)
            
            for t in indices:
                try:
                    # Construct Input
                    # For history=1: each var is [1, H, W], we want final shape [C, H, W]
                    # For history>1: each var is [history, H, W], flatten to [history*C, H, W]
                    x_list = []
                    actual_in_vars = []  # Track which variables were actually loaded
                    for var in self.in_vars:
                        if var not in data_dict:
                            continue  # Skip if variable not in data file
                        # Slice [t-history : t]
                        channel = data_dict[var][t - self.history : t]  # [history, H, W]
                        x_list.append(channel)
                        actual_in_vars.append(var)
                    
                    if len(x_list) == 0:
                        continue  # No valid input variables
                        
                    # Stack creates [num_vars, history, H, W], then reshape to [num_vars*history, H, W]
                    x = torch.cat(x_list, dim=0)  # Concatenate along time/history dimension
                    
                    # Construct Target
                    y_list = []
                    actual_out_vars = []  # Track which variables were actually loaded
                    for var in self.out_vars:
                        if var not in data_dict:
                            continue  # Skip if variable not in data file
                        target = data_dict[var][t + self.pred_range - 1]
                        y_list.append(target)
                        actual_out_vars.append(var)
                    
                    if len(y_list) == 0:
                        continue  # No valid output variables
                        
                    y = torch.stack(y_list, dim=0)
                    
                    # Transform (Normalization)
                    if self.transform:
                        # Ensure transform handles the dtype correctly
                        x, y = self.transform(x, y)
                        
                    yield x, y, actual_in_vars, actual_out_vars
                    
                except Exception as e:
                    continue
            
            # Free memory explicitly
            del data_dict
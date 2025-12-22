import torch
import torch.nn.functional as F
from functools import partial
from typing import Tuple, List

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
    return x  # unknown shape → no-op
     

def collate_resize(samples, lr_size=(32,64), mode="area", hr_size=None):
    """
    Custom collate function that downsamples inputs to lr_size on CPU.
    Handles samples from IterDataset which returns:
    (inp_data_dict, out_data_dict, variables, out_variables)
    
    Args:
        lr_size: Target size for inputs (H, W)
        mode: Interpolation mode for downsampling
        hr_size: Optional target size for outputs (H, W). If None, keeps original HR size.
                 If provided, downsamples outputs to this size (e.g., (128, 256))
    """
    import torch
    from torch.utils.data._utils.collate import default_collate
    import os
    
    first = samples[0]
    
    # Check if samples are tuples with 4 elements (from IterDataset)
    if isinstance(first, (tuple, list)) and len(first) == 4:
        inp_datas, out_datas, variables, out_variables = [], [], [], []
        for inp_data, out_data, vars, out_vars in samples:
            inp_datas.append(inp_data)
            out_datas.append(out_data)
            variables.append(vars)
            out_variables.append(out_vars)
        
        # Use the first sample's variables as reference
        variables = variables[0]
        out_variables = out_variables[0]
        
        # Stack all variable tensors across batch and concatenate along channel dim
        inp_list = []
        for var in variables:
            tensors = [inp[var] for inp in inp_datas]
            stacked = torch.stack(tensors, dim=0)  # [B, T, H, W]
            downsampled = _downsample_cpu(stacked, lr_size, mode=mode)
            inp_list.append(downsampled)
        
        out_list = []
        for var in out_variables:
            tensors = [out[var] for out in out_datas]
            stacked = torch.stack(tensors, dim=0)  # [B, H, W]
            # Add channel dimension: [B, H, W] -> [B, 1, H, W]
            stacked = stacked.unsqueeze(1)
            # Downsample outputs to hr_size if specified
            if hr_size is not None:
                stacked = _downsample_cpu(stacked, hr_size, mode=mode)
            out_list.append(stacked)
        
        # Concatenate along channel dimension
        x = torch.cat(inp_list, dim=1)  # [B, T*C, H, W]
        y = torch.cat(out_list, dim=1)  # [B, C, HR_H, HR_W]
        
        # Debug print (only once per process)
        if not hasattr(collate_resize, '_debug_printed'):
            rank = int(os.environ.get('RANK', 0))
            if rank == 0:
                print(f"[COLLATE DEBUG] Input shape after resize: {x.shape}, Target shape: {y.shape}", flush=True)
            collate_resize._debug_printed = True
        
        return x, y, variables, out_variables
    
    # Fallback for dict-based samples
    elif isinstance(first, dict):
        xs = [s["inputs"] for s in samples]
        ys = [s["targets"] for s in samples]
        x = torch.stack(xs, 0)
        y = torch.stack(ys, 0)
        x = _downsample_cpu(x, lr_size, mode=mode)
        return x, y
    
    # Fallback for simple (x, y) tuples
    elif isinstance(first, (tuple, list)) and len(first) == 2:
        xlist, ylist = zip(*samples)
        x = torch.stack(list(xlist), 0)
        y = torch.stack(list(ylist), 0)
        x = _downsample_cpu(x, lr_size, mode=mode)
        return x, y
    
    else:
        return default_collate(samples)
    
    
# def collate_resize(samples, lr_size=(32,64), mode="area"):
#     """
#     Custom collate function that downsamples inputs to lr_size on CPU.
#     Handles samples from IterDataset which returns:
#     (inp_data_dict, out_data_dict, variables, out_variables)
#     """
#     import torch
#     from torch.utils.data._utils.collate import default_collate
    
#     first = samples[0]
    
#     # Check if samples are tuples with 4 elements (from IterDataset)
#     if isinstance(first, (tuple, list)) and len(first) == 4:
#         inp_datas, out_datas, variables, out_variables = [], [], [], []
#         for inp_data, out_data, vars, out_vars in samples:
#             inp_datas.append(inp_data)
#             out_datas.append(out_data)
#             variables.append(vars)
#             out_variables.append(out_vars)
        
#         # Use the first sample's variables as reference
#         variables = variables[0]
#         out_variables = out_variables[0]
        
#         # Stack all variable tensors across batch and concatenate along channel dim
#         inp_list = []
#         for var in variables:
#             tensors = [inp[var] for inp in inp_datas]
#             stacked = torch.stack(tensors, dim=0)  # [B, T, H, W]
#             downsampled = _downsample_cpu(stacked, lr_size, mode=mode)
#             inp_list.append(downsampled)
        
#         out_list = []
#         for var in out_variables:
#             tensors = [out[var] for out in out_datas]
#             stacked = torch.stack(tensors, dim=0)  # [B, H, W] - keep HR
#             out_list.append(stacked)
        
#         # Concatenate along channel dimension
#         x = torch.cat(inp_list, dim=1)  # [B, T*C, H, W]
#         y = torch.cat(out_list, dim=1)  # [B, C, H, W]
        
#         return x, y, variables, out_variables
    
#     # Fallback for dict-based samples
#     elif isinstance(first, dict):
#         xs = [s["inputs"] for s in samples]
#         ys = [s["targets"] for s in samples]
#         x = torch.stack(xs, 0)
#         y = torch.stack(ys, 0)
#         x = _downsample_cpu(x, lr_size, mode=mode)
#         return x, y
    
#     # Fallback for simple (x, y) tuples
#     elif isinstance(first, (tuple, list)) and len(first) == 2:
#         xlist, ylist = zip(*samples)
#         x = torch.stack(list(xlist), 0)
#         y = torch.stack(list(ylist), 0)
#         x = _downsample_cpu(x, lr_size, mode=mode)
#         return x, y
    
#     else:
#         return default_collate(samples)
    
    
# def collate_resize(samples, lr_size=(32,64), mode="area"):
#     import torch
#     from torch.utils.data._utils.collate import default_collate

#     first = samples[0]
#     if isinstance(first, dict):
#         xs = [s["inputs"] for s in samples]
#         ys = [s["targets"] for s in samples]
#         x = torch.stack(xs, 0)    # CPU
#         y = torch.stack(ys, 0)    # CPU (HR)
#     elif isinstance(first, (tuple, list)) and len(first) == 2:
#         xlist, ylist = zip(*samples)
#         x = torch.stack(list(xlist), 0)
#         y = torch.stack(list(ylist), 0)
#     else:
#         # fall back to default and hope dataset already packs tensors
#         return default_collate(samples)

#     x = _downsample_cpu(x, lr_size, mode=mode)  # inputs -> LR
#     return x, y  # keep targets HR
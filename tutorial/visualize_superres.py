"""
Visualize super-resolution results: 
(1) Original HR input, (2) Downsampled LR input, (3) Model SR output, (4) Ground truth
Saves 4 separate PNG files.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import climate_learn as cl
from climate_learn.data.processing.era5_constants import PRESSURE_LEVEL_VARS, DEFAULT_PRESSURE_LEVELS
from climate_learn.data.transforms import _downsample_cpu
import glob
import os

# Configuration
CHECKPOINT_DIR = "/lustre/orion/csc662/proj-shared/janet/forecasting/v4_test_trans_resolution_res_slimvit_direct_forecasting_120/checkpoints"
ERA5_DIR = "/lustre/orion/lrn036/world-shared/ERA5_npz/1.40625_deg/"
OUTPUT_DIR = "/ccs/home/janetw/diffusion/ORBIT-2-sandbox/tutorial/superres_outputs"

# Model parameters (must match training config)
LR_SIZE = (32, 64)
HR_SIZE = (128, 256)
SUPERRES_FACTOR = 4
HISTORY = 1

# Data configuration (must match training)
variables = [
    "temperature",
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "lattitude",
]
out_variables = ["2m_temperature"]

def get_latest_checkpoint(ckpt_dir):
    """Find the latest checkpoint in directory."""
    ckpt_files = glob.glob(f"{ckpt_dir}/*.ckpt")
    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
    return max(ckpt_files, key=lambda x: Path(x).stat().st_mtime)

def prepare_data():
    """Load one sample from the dataset."""
    # Build variable lists
    in_vars = []
    for var in variables:
        if var in PRESSURE_LEVEL_VARS:
            for level in DEFAULT_PRESSURE_LEVELS:
                in_vars.append(var + "_" + str(level))
        else:
            in_vars.append(var)
    
    out_vars = []
    for var in out_variables:
        if var in PRESSURE_LEVEL_VARS:
            for level in DEFAULT_PRESSURE_LEVELS:
                out_vars.append(var + "_" + str(level))
        else:
            out_vars.append(var)
    
    # Create data module
    dm = cl.data.IterDataModule(
        "direct-forecasting",
        ERA5_DIR,
        ERA5_DIR,
        in_vars,
        out_vars,
        src="era5",
        history=HISTORY,
        window=6,
        pred_range=120,
        subsample=6,
        batch_size=1,
        num_workers=1,
    )
    dm.setup()
    
    # Get one sample from train dataloader
    train_loader = dm.train_dataloader()
    sample = next(iter(train_loader))
    
    return sample, in_vars, out_vars

def process_sample(sample):
    """Process sample to get HR input, LR input, and ground truth.
    
    Sample from dataloader is already processed by collate_fn:
    - x: [B, C*T, lr_h, lr_w] - LR inputs
    - y: [B, out_C, hr_h, hr_w] - HR ground truth
    
    We need to reconstruct original HR inputs manually from NPZ files.
    """
    x, y = sample  # Already collated batch
    
    # For visualization, we'll use first sample in batch
    lr_input = x[0]  # [C*T, 32, 64]
    gt = y[0]  # [out_C, 128, 256]
    
    # Load original HR data from NPZ
    # Get first NPZ file from dataset
    import glob
    npz_files = sorted(glob.glob(f"{ERA5_DIR}/train/*.npz"))
    first_npz = npz_files[0]
    
    data = np.load(first_npz)
    
    # Extract HR input for first timestep
    # Build HR input matching the in_vars
    in_vars = []
    for var in variables:
        if var in PRESSURE_LEVEL_VARS:
            for level in DEFAULT_PRESSURE_LEVELS:
                in_vars.append(var + "_" + str(level))
        else:
            in_vars.append(var)
    
    hr_channels = []
    for var in in_vars:
        if var in data:
            hr_channels.append(torch.from_numpy(data[var][0]))  # [H, W] - first timestep
    
    hr_input = torch.stack(hr_channels, dim=0)  # [C, H, W]
    
    return hr_input, lr_input, gt

def load_model_and_predict(checkpoint_path, lr_input):
    """Load model and generate super-resolution prediction."""
    # Calculate input/output channels
    in_vars = []
    for var in variables:
        if var in PRESSURE_LEVEL_VARS:
            for level in DEFAULT_PRESSURE_LEVELS:
                in_vars.append(var + "_" + str(level))
        else:
            in_vars.append(var)
    
    out_vars = []
    for var in out_variables:
        if var in PRESSURE_LEVEL_VARS:
            for level in DEFAULT_PRESSURE_LEVELS:
                out_vars.append(var + "_" + str(level))
        else:
            out_vars.append(var)
    
    # Build model kwargs
    model_kwargs = {
        "img_size": LR_SIZE,
        "in_channels": len(in_vars) * HISTORY,
        "out_channels": len(out_vars),
        "history": HISTORY,
        "superres_factor": SUPERRES_FACTOR,
        "cnn_ratio": 4,
        "patch_size": 4,
        "embed_dim": 512,
        "depth": 16,
        "decoder_depth": 4,
        "learn_pos_emb": True,
        "num_heads": 8,
    }
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create model
    from climate_learn.models.hub.res_slimvit import Res_Slim_ViT
    net = Res_Slim_ViT(**model_kwargs)
    
    # Load state dict (handle FSDP wrapper if present)
    state_dict = checkpoint['state_dict']
    new_state_dict = {}
    for key, value in state_dict.items():
        # Remove 'net.' prefix if present
        new_key = key.replace('net.', '') if key.startswith('net.') else key
        # Remove FSDP wrapper prefixes
        new_key = new_key.replace('_fsdp_wrapped_module.', '')
        new_key = new_key.replace('_forward_module.', '')
        new_state_dict[new_key] = value
    
    net.load_state_dict(new_state_dict, strict=False)
    net.eval()
    
    # Generate prediction
    with torch.no_grad():
        lr_input_batch = lr_input.unsqueeze(0)  # [1, C*T, 32, 64]
        sr_output = net(lr_input_batch)  # [1, out_C, 128, 256]
    
    return sr_output.squeeze(0)  # [out_C, 128, 256]

def save_images(hr_input, lr_input, sr_output, gt, output_dir):
    """Save 4 separate PNG files without any formatting."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Select first channel for visualization (temperature)
    hr_img = hr_input[0].cpu().numpy()  # First channel from [C, H, W]
    lr_img = lr_input[0].cpu().numpy()  # First channel
    sr_img = sr_output[0].cpu().numpy()  # First output channel
    gt_img = gt[0].cpu().numpy()  # First ground truth channel
    
    # Common colormap range
    vmin = min(hr_img.min(), sr_img.min(), gt_img.min())
    vmax = max(hr_img.max(), sr_img.max(), gt_img.max())
    
    # Save each image separately
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(hr_img, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    ax.axis('off')
    plt.savefig(f"{output_dir}/1_hr_input_128x256.png", bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close()
    print(f"✓ Saved: {output_dir}/1_hr_input_128x256.png")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(lr_img, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    ax.axis('off')
    plt.savefig(f"{output_dir}/2_lr_input_32x64.png", bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close()
    print(f"✓ Saved: {output_dir}/2_lr_input_32x64.png")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(sr_img, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    ax.axis('off')
    plt.savefig(f"{output_dir}/3_sr_output_128x256.png", bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close()
    print(f"✓ Saved: {output_dir}/3_sr_output_128x256.png")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(gt_img, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    ax.axis('off')
    plt.savefig(f"{output_dir}/4_ground_truth_128x256.png", bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close()
    print(f"✓ Saved: {output_dir}/4_ground_truth_128x256.png")
    
    # Calculate error stats
    error = np.abs(sr_img - gt_img)
    mae = error.mean()
    rmse = np.sqrt((error ** 2).mean())
    print(f"\nError metrics: MAE={mae:.4f}, RMSE={rmse:.4f}")

def main():
    print("="*70)
    print("Super-Resolution Visualization")
    print("="*70)
    
    # Find checkpoint
    ckpt_path = get_latest_checkpoint(CHECKPOINT_DIR)
    print(f"Using checkpoint: {Path(ckpt_path).name}\n")
    
    # Load data
    print("Loading data sample...")
    sample, in_vars, out_vars = prepare_data()
    print(f"✓ Loaded sample with {len(in_vars)} input variables and {len(out_vars)} output variable(s)")
    
    # Process sample
    print("\nProcessing sample...")
    hr_input, lr_input, gt = process_sample(sample)
    print(f"✓ HR input shape: {hr_input.shape}")
    print(f"✓ LR input shape: {lr_input.shape}")
    print(f"✓ Ground truth shape: {gt.shape}")
    
    # Generate prediction
    print("\nGenerating super-resolution prediction...")
    sr_output = load_model_and_predict(ckpt_path, lr_input)
    print(f"✓ SR output shape: {sr_output.shape}")
    
    # Save images
    print("\nSaving images...")
    save_images(hr_input, lr_input, sr_output, gt, OUTPUT_DIR)
    
    print("\n" + "="*70)
    print(f"All images saved to: {OUTPUT_DIR}")
    print("="*70)

if __name__ == "__main__":
    main()

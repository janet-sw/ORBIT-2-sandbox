# Standard library
from argparse import ArgumentParser
import os
import sys
import functools
from datetime import timedelta

# PyTorch
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
from torch.distributed.fsdp import MixedPrecision

# Third party
import climate_learn as cl
from climate_learn.data.processing.era5_constants import (
    PRESSURE_LEVEL_VARS,
    DEFAULT_PRESSURE_LEVELS,
    CONSTANTS,
)
from climate_learn.metrics import MetricsMetaInfo
from climate_learn.utils.loaders import get_climatology
from climate_learn.models.hub.components.vit_blocks import Block
from climate_learn.data.transforms import collate_batch_only
from climate_learn.utils.monthly_loader import SequentialMonthlyDataset
from utils import seed_everything

import yaml
import numpy as np
from pathlib import Path


def log_gpu_memory(device, message="", world_rank=None):
    """Log GPU memory usage including max allocated with optional message and rank."""
    memory_allocated = torch.cuda.memory_allocated(device) / 1024 / 1024 / 1024
    max_memory_allocated = torch.cuda.max_memory_allocated(device) / 1024 / 1024 / 1024
    memory_reserved = torch.cuda.memory_reserved(device) / 1024 / 1024 / 1024
    
    if world_rank is not None:
        print(
            f"Rank {world_rank} {message} Allocated: {memory_allocated:.2f}GB, "
            f"Max: {max_memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB",
            flush=True,
        )
    else:
        print(
            f"{message} Allocated: {memory_allocated:.2f}GB, "
            f"Max: {max_memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB",
            flush=True,
        )


def clip_replace_constant(y, yhat, out_variables):
    """Clip precipitation to non-negative values and replace constants with ground truth."""
    # Clip precipitation to non-negative values if it exists
    if "total_precipitation_24hr" in out_variables:
        idx = out_variables.index("total_precipitation_24hr")
        yhat[:, idx] = torch.clamp(yhat[:, idx], min=0)

    # Replace constant variables (like land_sea_mask, orography) with ground-truth values
    for i in range(yhat.shape[1]):
        if out_variables[i] in CONSTANTS:
            yhat[:, i] = y[:, i]
    return yhat


def test_step(batch, batch_idx, net, device, test_loss_metrics, test_target_transforms, resize_config=None):
    """Run a single test step and return loss dictionary."""
    x, y, in_variables, out_variables = batch
    x = x.to(device)
    y = y.to(device)
    
    # Flatten 5D [B, C, T, H, W] to 4D [B, C*T, H, W]
    if x.dim() == 5:
        B, C, T, H, W = x.shape
        x = x.reshape(B, C * T, H, W)
    if y.dim() == 5:
        B, C, T, H, W = y.shape
        y = y.reshape(B, C * T, H, W)
    
    # GPU-based downsampling: only downsample x (input) to LR
    # y stays at HR because the model outputs at HR (superres)
    if resize_config is not None:
        lr_h, lr_w = resize_config["lr_size"]
        mode = resize_config["mode"]
        antialias = resize_config.get("antialias", None)
        if antialias is None:
            antialias = mode in ("bilinear", "bicubic")
        if x.shape[2] != lr_h or x.shape[3] != lr_w:
            if mode in ('max_pool', 'maxpool'):
                x = torch.nn.functional.adaptive_max_pool2d(x, output_size=(lr_h, lr_w))
            elif mode in ('avg_pool', 'avgpool'):
                x = torch.nn.functional.adaptive_avg_pool2d(x, output_size=(lr_h, lr_w))
            elif mode in ('linear', 'bilinear', 'bicubic', 'trilinear'):
                x = torch.nn.functional.interpolate(
                    x, size=(lr_h, lr_w), mode=mode, align_corners=False, antialias=antialias
                )
            else:
                x = torch.nn.functional.interpolate(x, size=(lr_h, lr_w), mode=mode)

    yhat = net.forward(x, in_variables, out_variables)
    yhat = clip_replace_constant(y, yhat, out_variables)

    loss_dict = {}
    for i, lf in enumerate(test_loss_metrics):
        loss_name = getattr(lf, "name", f"metric_{i}")
        
        # Apply denormalization if needed
        if test_target_transforms is not None and test_target_transforms[i] is not None:
            yhat_ = test_target_transforms[i](yhat)
            y_ = test_target_transforms[i](y)
        else:
            yhat_ = yhat
            y_ = y
        
        # Compute loss
        if y_.size(dim=2) != yhat_.size(dim=2) or y_.size(dim=3) != yhat_.size(dim=3):
            losses = lf(yhat_, y_[:, :, 0:yhat_.size(dim=2), 0:yhat_.size(dim=3)])
        else:
            losses = lf(yhat_, y_)
        
        if losses.dim() == 0:
            loss_dict[f"test/{loss_name}:aggregate"] = losses
        else:
            if losses.numel() > 1:
                for idx, var_name in enumerate(out_variables):
                    if idx < losses.numel():
                        loss_dict[f"test/{loss_name}:{var_name}"] = losses[idx]
                loss_dict[f"test/{loss_name}:aggregate"] = losses[-1]
            else:
                loss_dict[f"test/{loss_name}:aggregate"] = losses.item() if losses.numel() == 1 else losses
    
    return loss_dict


def parse_config(config_path, world_rank):
    """Parse configuration from YAML file."""
    if world_rank == 0:
        print(f"Loading config from {config_path}", flush=True)

    with open(config_path, "r") as f:
        conf = yaml.safe_load(f)

    return conf


def create_model(config, device, world_rank, in_vars, out_vars):
    """Create model for testing."""
    preset = config["model"]["preset"]
    lr_h = config["model"]["img_size"][0]
    lr_w = config["model"]["img_size"][1]
    superres_factor = config["model"]["superres_factor"]
    
    in_channels = len(in_vars)
    out_channels = len(out_vars)
    history = config["model"]["history"]
    
    if world_rank == 0:
        print(f"Creating {preset} model: {in_channels} -> {out_channels} channels", flush=True)

    if preset == "res_slimvit":
        from climate_learn.models.hub import Res_Slim_ViT
        from climate_learn.utils.fused_attn import FusedAttn
        
        # Count constant variables
        constant_vars = ["land_sea_mask", "orography", "lattitude", "landcover"]
        num_constant_vars = sum(1 for var in constant_vars if var in in_vars)
        
        # Model needs ALL variables (input + output) for var_map
        all_vars = []
        seen = set()
        for var in in_vars + out_vars:
            if var not in seen:
                all_vars.append(var)
                seen.add(var)
        
        model = Res_Slim_ViT(
            default_vars=all_vars,
            img_size=(lr_h, lr_w),
            in_channels=in_channels,
            out_channels=out_channels,
            superres_mag=superres_factor,
            history=history,
            patch_size=config["model"]["patch_size"],
            cnn_ratio=config["model"]["cnn_ratio"],
            learn_pos_emb=config["model"]["learn_pos_emb"],
            embed_dim=config["model"]["embed_dim"],
            depth=config["model"]["depth"],
            decoder_depth=config["model"]["decoder_depth"],
            num_heads=config["model"]["num_heads"],
            mlp_ratio=config["model"].get("mlp_ratio", 4.0),
            drop_path=config["model"].get("drop_path", 0.1),
            drop_rate=config["model"].get("drop_rate", 0.0),
            num_constant_vars=num_constant_vars,
            FusedAttn_option=FusedAttn.DEFAULT,
            input_refine_cnn=config["model"].get("input_refine_cnn", False),
        )
    else:
        raise ValueError(f"Unknown model preset: {preset}")
    
    if world_rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}", flush=True)
        print(f"Trainable parameters: {trainable_params:,}", flush=True)

    return model


def create_test_data_module(config, world_rank):
    """Create data module for testing."""
    era5_dir = config["data"]["era5_dir"]
    forecast_type = config["data"]["forecast_type"]
    pred_range = config["data"]["pred_range"]
    
    # Define variables
    variables = config["data"]["variables"]
    in_vars = []
    for var in variables:
        if var in PRESSURE_LEVEL_VARS:
            in_vars.extend([f"{var}_{level}" for level in DEFAULT_PRESSURE_LEVELS])
        else:
            in_vars.append(var)
    
    out_variables = config["data"]["out_variables"]
    out_vars = []
    for var in out_variables:
        if var in PRESSURE_LEVEL_VARS:
            out_vars.extend([f"{var}_{level}" for level in DEFAULT_PRESSURE_LEVELS])
        else:
            out_vars.append(var)
    
    if world_rank == 0:
        print(f"Input variables ({len(in_vars)}): {in_vars[:5]}...", flush=True)
        print(f"Output variables ({len(out_vars)}): {out_vars[:5]}...", flush=True)

    batch_size = config["trainer"]["batch_size"]
    num_workers = config["trainer"]["num_workers"]
    history = config["model"]["history"]
    
    world_size_value = int(os.environ.get("SLURM_NTASKS", 1))
    test_batch_size = config["trainer"].get("test_batch_size", 64)
    
    if world_rank == 0:
        print(f"Creating data module with batch_size={test_batch_size}, num_workers={num_workers}", flush=True)
        
    buffer_size = config["trainer"]["buffer_size"]

    # For single GPU, don't use data_par_size and set num_workers=0 to avoid distributed issues
    if world_size_value == 1:
        if forecast_type in ("direct", "iterative"):
            data_module = cl.data.IterDataModule(
                f"{forecast_type}-forecasting",
                era5_dir, era5_dir, in_vars, out_vars,
                src="era5",
                history=history,
                window=6,
                pred_range=pred_range,
                subsample=1,
                batch_size=test_batch_size,
                num_workers=0,
            )
        elif forecast_type == "continuous":
            data_module = cl.data.IterDataModule(
                "continuous-forecasting",
                era5_dir, era5_dir, in_vars, out_vars,
                src="era5",
                history=history,
                window=6,
                pred_range=1,
                max_pred_range=120,
                random_lead_time=True,
                hrs_each_step=1,
                subsample=1,
                batch_size=test_batch_size,
                buffer_size=buffer_size,
                num_workers=0,
            )
    else:
        if forecast_type in ("direct", "iterative"):
            data_module = cl.data.IterDataModule(
                f"{forecast_type}-forecasting",
                era5_dir, era5_dir, in_vars, out_vars,
                data_par_size=world_size_value,
                src="era5",
                history=history,
                window=6,
                pred_range=pred_range,
                subsample=1,
                batch_size=test_batch_size,
                num_workers=num_workers,
            )
        elif forecast_type == "continuous":
            data_module = cl.data.IterDataModule(
                "continuous-forecasting",
                era5_dir, era5_dir, in_vars, out_vars,
                data_par_size=world_size_value,
                src="era5",
                history=history,
                window=6,
                pred_range=1,
                max_pred_range=120,
                random_lead_time=True,
                hrs_each_step=1,
                subsample=1,
                batch_size=test_batch_size,
                buffer_size=buffer_size,
                num_workers=num_workers,
            )
    
    lr_h, lr_w = config["model"]["img_size"]
    superres_factor = config["model"]["superres_factor"]
    hr_h, hr_w = lr_h * superres_factor, lr_w * superres_factor
    downsample_mode = config["model"]["downsample_mode"]
    
    # Use custom collate function for batching
    data_module.collate_fn = collate_batch_only
    
    # Store resize config (matching training script's format)
    # antialias: read from config; if absent, auto-derive (True for bilinear/bicubic)
    antialias_setting = config["model"].get("antialias", None)
    if not hasattr(data_module, 'resize_config'):
        data_module.resize_config = {
            "lr_size": (lr_h, lr_w),
            "hr_size": (hr_h, hr_w),
            "mode": downsample_mode,
            "antialias": antialias_setting,
        }

    if world_rank == 0:
        effective_aa = antialias_setting if antialias_setting is not None else (downsample_mode in ("bilinear", "bicubic"))
        print(f"Resize config: mode={downsample_mode}, antialias={antialias_setting} (effective={effective_aa})", flush=True)
        print("Setting up data module...", flush=True)

    data_module.setup()

    # Create test data loader
    test_dataloader = data_module.test_dataloader()

    return data_module, test_dataloader, in_vars, out_vars


def create_test_losses(config, device, world_rank, data_module, in_vars, out_vars):
    """Create loss functions for testing."""
    lat, lon = data_module.get_lat_lon()
    
    if world_rank == 0:
        print("Loading test climatology...", flush=True)
    test_clim = get_climatology(data_module, split="test")
    
    test_metainfo = MetricsMetaInfo(in_vars, out_vars, lat, lon, test_clim)
    
    # Test losses (pass None for model - not needed for metrics)
    test_losses = [
        cl.load_loss(device, None, "lat_rmse", False, test_metainfo),
        cl.load_loss(device, None, "lat_acc", False, test_metainfo),
        cl.load_loss(device, None, "lat_mse", False, test_metainfo),
    ]
    
    # Test transforms
    test_transforms = [
        cl.load_transform("denormalize", data_module),
        cl.load_transform("denormalize", data_module),
        None,
    ]
    
    if world_rank == 0:
        print("Test losses and transforms created.", flush=True)
    
    return test_losses, test_transforms


def load_checkpoint(checkpoint_path, model, world_rank, is_distributed=False):
    """Load checkpoint for testing with FSDP-aware loading."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    if world_rank == 0:
        print(f"Loading checkpoint from {checkpoint_path}...", flush=True)
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Use FSDP-aware state dict loading if distributed
    if is_distributed:
        load_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, load_policy):
            model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
    
    epoch = checkpoint.get("epoch", "unknown")
    
    if world_rank == 0:
        print(f"Loaded checkpoint from epoch {epoch}", flush=True)
    
    del checkpoint
    torch.cuda.empty_cache()
    
    return epoch


def test_model(
    model,
    test_dataloader,
    world_rank,
    device,
    test_loss_metrics,
    test_target_transforms,
    in_vars,
    out_vars,
    data_module=None,
):
    """Test model on test dataset."""
    model.eval()
    
    # Initialize loss accumulation
    loss_dict_sum = {}
    num_batches = 0

    if world_rank == 0:
        print(f"\n{'='*80}", flush=True)
        print(f"Running test evaluation...", flush=True)
        print(f"{'='*80}\n", flush=True)

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            resize_config = getattr(data_module, 'resize_config', None)
            
            loss_dict = test_step(
                batch, batch_idx, model, device, test_loss_metrics, test_target_transforms, resize_config
            )
            
            # Accumulate losses
            for key, value in loss_dict.items():
                if key not in loss_dict_sum:
                    loss_dict_sum[key] = 0.0
                loss_dict_sum[key] += value.item() if torch.is_tensor(value) else value
            
            num_batches += 1
            
            if world_rank == 0 and batch_idx % 50 == 0:
                print(f"Processed {batch_idx} batches", flush=True)

    # Average losses
    if num_batches > 0:
        for key in loss_dict_sum:
            loss_dict_sum[key] /= num_batches

    if world_rank == 0:
        print(f"\n{'='*80}", flush=True)
        print(f"Test Results:", flush=True)
        print(f"{'='*80}", flush=True)
        for key, value in sorted(loss_dict_sum.items()):
            print(f"  {key}: {value:.6f}", flush=True)
        print(f"{'='*80}\n", flush=True)

    return loss_dict_sum


def setup_environment():
    """Set up environment - supports single GPU and multi-GPU."""
    if "SLURM_NTASKS" in os.environ and int(os.environ["SLURM_NTASKS"]) > 1:
        # Distributed setup
        world_size = int(os.environ["SLURM_NTASKS"])
        world_rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(os.environ["SLURM_LOCALID"])
        
        os.environ["MASTER_ADDR"] = str(os.environ["HOSTNAME"])
        os.environ["MASTER_PORT"] = "29500"
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", timeout=timedelta(seconds=7200), rank=world_rank, world_size=world_size)
        
        if world_rank == 0:
            print(f"Initialized distributed process group. world_size={world_size}")
            print(f"world_size={world_size}, world_rank={world_rank}, local_rank={local_rank}")
    else:
        # Single GPU setup - still need process group for dataset compatibility
        world_size = 1
        world_rank = 0
        local_rank = 0
        
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        
        torch.cuda.set_device(0)
        dist.init_process_group("nccl", timeout=timedelta(seconds=7200), rank=0, world_size=1)
        print("Running on single GPU (initialized process group for dataset compatibility)", flush=True)
    
    return world_size, world_rank, local_rank


def main():
    """Main testing function."""
    # Setup environment
    world_size, world_rank, local_rank = setup_environment()
    device = torch.cuda.current_device()

    # Parse arguments
    if len(sys.argv) < 3:
        if world_rank == 0:
            print("Usage: python test_era5_forecasting.py <config_path> <checkpoint_path>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    checkpoint_path = sys.argv[2]

    # Parse config
    config = parse_config(config_path, world_rank)

    # Extract parameters
    data_type = config["trainer"].get("data_type", "float32")

    if world_rank == 0:
        print(f"\n{'='*80}", flush=True)
        print(f"Testing Configuration", flush=True)
        print(f"{'='*80}", flush=True)
        print(f"Config: {config_path}", flush=True)
        print(f"Checkpoint: {checkpoint_path}", flush=True)
        print(f"Data type: {data_type}", flush=True)
        print(f"World size: {world_size}", flush=True)
        print(f"{'='*80}\n", flush=True)

    # Seed everything
    seed_everything(42)

    # Create data module
    data_module, test_dataloader, in_vars, out_vars = create_test_data_module(
        config, world_rank
    )

    # Create model
    model = create_model(config, device, world_rank, in_vars, out_vars)

    # Wrap model: FSDP for multi-GPU, simple .to(device) for single GPU
    if world_size > 1:
        if data_type == "bfloat16":
            bfloat_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
        else:
            bfloat_policy = None

        model = FSDP(
            model,
            device_id=local_rank,
            mixed_precision=bfloat_policy,
            use_orig_params=True,
        )
        if world_rank == 0:
            print("Model wrapped with FSDP", flush=True)
    else:
        model = model.to(device)
        if world_rank == 0:
            print("Model moved to GPU", flush=True)

    # Load checkpoint
    epoch = load_checkpoint(checkpoint_path, model, world_rank, is_distributed=(world_size > 1))

    # Create test losses
    test_losses, test_transforms = create_test_losses(
        config, device, world_rank, data_module, in_vars, out_vars
    )

    if world_rank == 0:
        print("Starting testing...\n", flush=True)

    # Run test
    test_results = test_model(
        model,
        test_dataloader,
        world_rank,
        device,
        test_losses,
        test_transforms,
        in_vars,
        out_vars,
        data_module=data_module,
    )

    # Save results to text file
    if world_rank == 0:
        output_dir = os.path.dirname(checkpoint_path)
        results_file = os.path.join(output_dir, "test_results.txt")
        with open(results_file, "w") as f:
            f.write(f"Checkpoint: {checkpoint_path}\n")
            f.write(f"Epoch: {epoch}\n")
            f.write(f"{'='*80}\n")
            f.write("Test Results:\n")
            f.write(f"{'='*80}\n")
            for key, value in sorted(test_results.items()):
                f.write(f"{key}: {value:.6f}\n")
        print(f"Results saved to: {results_file}", flush=True)
        print("Testing complete!", flush=True)

    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

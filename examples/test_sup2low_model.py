#!/usr/bin/env python3
"""
Test script for super-resolution forecasting model trained without PyTorch Lightning.
Similar to test_trans_res_parallel.py but for sup2low_forecasting_no_lightning.py checkpoints.
"""

import os
import sys
import yaml
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from functools import partial
from datetime import timedelta
import numpy as np

# Third party
import climate_learn as cl
from climate_learn.data.processing.era5_constants import (
    PRESSURE_LEVEL_VARS,
    DEFAULT_PRESSURE_LEVELS,
    CONSTANTS,
)
from climate_learn.metrics import MetricsMetaInfo
from climate_learn.utils.fused_attn import FusedAttn
from climate_learn.data.transforms import collate_resize
from utils import seed_everything


def setup_environment():
    """Set up environment - single GPU for testing."""
    # Check if running in distributed mode
    if "SLURM_NTASKS" in os.environ and int(os.environ["SLURM_NTASKS"]) > 1:
        # Distributed setup
        world_size = int(os.environ["SLURM_NTASKS"])
        world_rank = int(os.environ["SLURM_PROCID"])
        local_rank = int(os.environ["SLURM_LOCALID"])
        
        os.environ["MASTER_ADDR"] = str(os.environ["HOSTNAME"])
        os.environ["MASTER_PORT"] = "29500"
        
        torch.cuda.set_device(0)
        dist.init_process_group("nccl", timeout=timedelta(seconds=7200), rank=world_rank, world_size=world_size)
        
        if world_rank == 0:
            print(f"Initialized process group. world_size={world_size}")
            print(f"world_size={world_size}, world_rank={world_rank}, local_rank={local_rank}")
    else:
        # Single GPU setup - still need to initialize process group for dataset compatibility
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


def clip_replace_constant(y, yhat, out_variables):
    """Replace constants in predictions with ground truth values."""
    if "total_precipitation_24hr" in out_variables:
        prcp_index = out_variables.index("total_precipitation_24hr")
        torch.clamp_(yhat[:, prcp_index, :, :], min=0.0)
    for i in range(yhat.shape[1]):
        if out_variables[i] in CONSTANTS:
            yhat[:, i] = y[:, i]
    return yhat


def test_step(batch, batch_idx, net, device, test_loss_metrics, test_target_transforms):
    """Run one test step."""
    x, y, in_variables, out_variables = batch
    x = x.to(device)
    y = y.to(device)

    yhat = net.forward(x, in_variables, out_variables)
    yhat = clip_replace_constant(y, yhat, out_variables)

    loss_dict = {}
    for i, lf in enumerate(test_loss_metrics):
        if test_target_transforms is not None and test_target_transforms[i] is not None:
            yhat_ = test_target_transforms[i](yhat)
            y_ = test_target_transforms[i](y)
        else:
            yhat_ = yhat
            y_ = y
            
        if y_.size(dim=2) != yhat_.size(dim=2) or y_.size(dim=3) != yhat_.size(dim=3):
            losses = lf(yhat_, y_[:, :, 0 : yhat_.size(dim=2), 0 : yhat_.size(dim=3)])
        else:
            losses = lf(yhat_, y_)

        loss_name = getattr(lf, "name", f"loss_{i}")
        if losses.dim() == 0:
            loss_dict[f"test/{loss_name}:aggregate"] = losses
        else:
            if losses.numel() > 1:
                for idx, var_name in enumerate(out_variables):
                    if idx < losses.numel():
                        name = f"test/{loss_name}:{var_name}"
                        loss_dict[name] = losses[idx]
                loss_dict[f"test/{loss_name}:aggregate"] = losses[-1]
            else:
                loss_dict[f"test/{loss_name}:aggregate"] = losses.item() if losses.numel() == 1 else losses
                
    return loss_dict


def test_epoch(model, test_dataloader, world_rank, device, test_loss_metrics, test_target_transforms, in_vars, out_vars):
    """Run testing on the test dataset."""
    model.eval()
    
    loss_dict_sum = {}
    num_batches = 0

    if world_rank == 0:
        print(f"\n{'='*80}")
        print(f"Running test evaluation...")
        print(f"{'='*80}\n", flush=True)

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            loss_dict = test_step(
                batch, batch_idx, model, device, test_loss_metrics, test_target_transforms
            )
            
            for key, value in loss_dict.items():
                if key not in loss_dict_sum:
                    loss_dict_sum[key] = 0.0
                loss_dict_sum[key] += value.item()
            
            num_batches += 1
            
            if world_rank == 0 and batch_idx % 50 == 0:
                print(f"Processed {batch_idx} batches", flush=True)

    if num_batches > 0:
        for key in loss_dict_sum:
            loss_dict_sum[key] /= num_batches

    if world_rank == 0:
        print(f"\n{'='*80}")
        print(f"Test Results:")
        print(f"{'='*80}")
        for key, value in sorted(loss_dict_sum.items()):
            print(f"  {key}: {value:.6f}")
        print(f"{'='*80}\n", flush=True)

    return loss_dict_sum


def load_checkpoint_for_test(checkpoint_path, model, world_rank, is_distributed=False):
    """Load checkpoint for testing."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    if world_rank == 0:
        print(f"Loading checkpoint from: {checkpoint_path}", flush=True)
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Load model state dict
    if is_distributed:
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        load_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, load_policy):
            model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # Single GPU - direct load
        model.load_state_dict(checkpoint["model_state_dict"])
    
    epoch = checkpoint.get("epoch", -1)
    if world_rank == 0:
        print(f"Loaded checkpoint from epoch {epoch}", flush=True)
    
    del checkpoint
    return epoch


def create_test_data_module(config, world_rank):
    """Create test data module."""
    era5_dir = config["data"]["era5_dir"]
    variables = config["data"]["variables"]
    out_variables = config["data"]["out_variables"]
    
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
    
    # Single GPU or distributed
    world_size_value = int(os.environ.get("SLURM_NTASKS", 1))
    
    # Configure collate function for super-resolution
    lr_h, lr_w = config["model"]["img_size"]
    superres_factor = config["model"]["superres_factor"]
    hr_h, hr_w = lr_h * superres_factor, lr_w * superres_factor
    collate = partial(collate_resize, lr_size=(lr_h, lr_w), hr_size=(hr_h, hr_w), mode=config["model"]["downsample_mode"])
    
    if world_rank == 0:
        print(f"Creating test data module for {config['data']['forecast_type']} forecasting", flush=True)
        print(f"Input variables: {in_vars}", flush=True)
        print(f"Output variables: {out_vars}", flush=True)
    
    # For single GPU, don't use data_par_size and set num_workers=0 to avoid distributed issues
    if world_size_value == 1:
        data_module = cl.data.IterDataModule(
            f"{config['data']['forecast_type']}-forecasting",
            era5_dir, era5_dir, in_vars, out_vars,
            src="era5", 
            history=config["model"]["history"], 
            window=6, 
            pred_range=config["data"]["pred_range"],
            subsample=1,
            batch_size=config["trainer"].get("test_batch_size", 64),
            num_workers=0,  # Set to 0 to avoid torch.distributed issues in single GPU mode
        )
    else:
        data_module = cl.data.IterDataModule(
            f"{config['data']['forecast_type']}-forecasting",
            era5_dir, era5_dir, in_vars, out_vars,
            data_par_size=world_size_value,
            src="era5", 
            history=config["model"]["history"], 
            window=6, 
            pred_range=config["data"]["pred_range"],
            subsample=1,
            batch_size=config["trainer"].get("test_batch_size", 64),
            num_workers=config["trainer"]["num_workers"],
        )
    
    data_module.collate_fn = collate
    data_module.setup()
    
    return data_module, data_module.test_dataloader(), in_vars, out_vars


def create_model(config, device, world_rank, in_vars, out_vars):
    """Create model architecture (same as training)."""
    preset = config["model"]["preset"]
    lr_h, lr_w = config["model"]["img_size"]
    superres_factor = config["model"]["superres_factor"]
    in_channels = len(in_vars)
    out_channels = len(out_vars)
    history = config["model"]["history"]
    
    if world_rank == 0:
        print(f"Creating model: {preset}", flush=True)
        print(f"Model config: img_size=({lr_h}, {lr_w}), in_channels={in_channels}, out_channels={out_channels}, history={history}, superres_factor={superres_factor}", flush=True)

    if preset == "res_slimvit":
        from climate_learn.models.hub import Res_Slim_ViT
        
        constant_vars = ["land_sea_mask", "orography", "lattitude", "landcover"]
        num_constant_vars = sum(1 for var in constant_vars if var in in_vars)
        
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
        )
    else:
        raise NotImplementedError(f"Model preset {preset} not implemented")
    
    if world_rank == 0:
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M", flush=True)
    
    return model


def create_test_losses(config, device, world_rank, data_module, in_vars, out_vars):
    """Create test loss metrics."""
    if world_rank == 0:
        print("Loading test climatology...", flush=True)
    
    test_clim = cl.utils.loaders.get_climatology(data_module, split="test")
    
    lat, lon = data_module.get_lat_lon()
    
    test_loss_metrics = [
        cl.load_loss(device, None, "lat_rmse", False, MetricsMetaInfo(in_vars, out_vars, lat, lon, test_clim)),
        cl.load_loss(device, None, "lat_acc", False, MetricsMetaInfo(in_vars, out_vars, lat, lon, test_clim)),
        cl.load_loss(device, None, "lat_mse", False, MetricsMetaInfo(in_vars, out_vars, lat, lon, test_clim)),
    ]
    
    test_target_transforms = [
        cl.load_transform("denormalize", data_module),
        cl.load_transform("denormalize", data_module),
        None,
    ]
    
    if world_rank == 0:
        print("Test losses created successfully", flush=True)
    
    return test_loss_metrics, test_target_transforms


def main():
    # Setup distributed environment
    world_size, world_rank, local_rank = setup_environment()
    device = torch.cuda.current_device()
    
    # Parse arguments
    if len(sys.argv) < 3:
        if world_rank == 0:
            print("Usage: python test_sup2low_model.py <config_path> <checkpoint_path>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    checkpoint_path = sys.argv[2]
    
    # Load config
    if world_rank == 0:
        print(f"\n{'='*80}")
        print(f"Testing Configuration")
        print(f"{'='*80}")
        print(f"Config: {config_path}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"{'='*80}\n", flush=True)
    
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    seed_everything(42)
    
    # Create data module
    data_module, test_loader, in_vars, out_vars = create_test_data_module(config, world_rank)
    
    # Create model
    model = create_model(config, device, world_rank, in_vars, out_vars)
    
    # Wrap with FSDP only if distributed, otherwise just move to GPU
    if world_size > 1:
        data_type = config["trainer"].get("data_type", "float32")
        bfloat_policy = MixedPrecision(
            param_dtype=torch.bfloat16, 
            reduce_dtype=torch.bfloat16, 
            buffer_dtype=torch.bfloat16
        ) if data_type == "bfloat16" else None
        
        model = FSDP(model, device_id=0, mixed_precision=bfloat_policy, use_orig_params=True)
    else:
        model = model.to(device)
    
    # Load checkpoint
    load_checkpoint_for_test(checkpoint_path, model, world_rank, is_distributed=(world_size > 1))
    
    # Create test losses (pass in_vars and out_vars from data module creation)
    test_loss_metrics, test_target_transforms = create_test_losses(config, device, world_rank, data_module, in_vars, out_vars)
    
    # Run testing
    test_results = test_epoch(
        model, test_loader, world_rank, device, 
        test_loss_metrics, test_target_transforms, in_vars, out_vars
    )
    
    # Save results
    if world_rank == 0:
        output_dir = os.path.dirname(checkpoint_path)
        results_file = os.path.join(output_dir, "test_results_v2.txt")
        with open(results_file, "w") as f:
            f.write(f"Checkpoint: {checkpoint_path}\n")
            f.write(f"{'='*80}\n")
            f.write("Test Results:\n")
            f.write(f"{'='*80}\n")
            for key, value in sorted(test_results.items()):
                f.write(f"{key}: {value:.6f}\n")
        print(f"Results saved to: {results_file}", flush=True)
    
    # Clean up - always destroy since we always initialize now
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

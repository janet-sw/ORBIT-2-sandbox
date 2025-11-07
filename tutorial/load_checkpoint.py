"""
Script to load a trained res_slim_vit model checkpoint for weather forecasting and run testing.

Usage:
    python load_checkpoint.py --checkpoint /path/to/checkpoint.ckpt --era5_dir /path/to/era5/data
    
Example:
    python load_checkpoint.py \
        --checkpoint /lustre/orion/csc662/proj-shared/janet/forecasting/res_slimvit_direct_forecasting_120/checkpoints/epoch_004-v1.ckpt \
        --era5_dir /path/to/era5/data \
        --forecast_type direct \
        --pred_range 120 \
        --devices 1
"""

# Standard library
from argparse import ArgumentParser
import os
import functools

# Third party
import torch
import pytorch_lightning as pl
import climate_learn as cl
from climate_learn.data.processing.era5_constants import (
    PRESSURE_LEVEL_VARS,
    DEFAULT_PRESSURE_LEVELS,
)
from pytorch_lightning.strategies import FSDPStrategy
from timm.models.vision_transformer import Block
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


def setup_data_module(era5_dir, forecast_type="direct", pred_range=120, batch_size=128):
    """
    Set up the data module with the same configuration as training.
    
    Args:
        era5_dir: Path to ERA5 data directory
        forecast_type: Type of forecasting (direct, iterative, or continuous)
        pred_range: Prediction range in hours
        batch_size: Batch size for data loading
    
    Returns:
        Configured data module
    """
    # Define variables (same as training)
    variables = [
        "geopotential",
        "temperature",
        "u_component_of_wind",
        "v_component_of_wind",
        "specific_humidity",
        "2m_temperature",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "land_sea_mask",
        "orography",
        "lattitude",
        "total_precipitation",
    ]
    
    # Process input variables
    in_vars = []
    for var in variables:
        if var in PRESSURE_LEVEL_VARS:
            for level in DEFAULT_PRESSURE_LEVELS:
                in_vars.append(var + "_" + str(level))
        else:
            in_vars.append(var)
    
    # Define output variables for direct forecasting
    if forecast_type in ("direct", "continuous"):
        out_variables = [
            "2m_temperature", 
            "geopotential_500", 
            "temperature_850", 
            "total_precipitation", 
            "10m_u_component_of_wind", 
            "10m_v_component_of_wind"
        ]
    elif forecast_type == "iterative":
        out_variables = variables
    
    # Process output variables
    out_vars = []
    for var in out_variables:
        if var in PRESSURE_LEVEL_VARS:
            for level in DEFAULT_PRESSURE_LEVELS:
                out_vars.append(var + "_" + str(level))
        else:
            out_vars.append(var)
    
    # Create data module
    if forecast_type in ("direct", "iterative"):
        dm = cl.data.IterDataModule(
            f"{forecast_type}-forecasting",
            era5_dir,
            era5_dir,
            in_vars,
            out_vars,
            src="era5",
            history=3,
            window=6,
            pred_range=pred_range,
            subsample=6,
            batch_size=batch_size,
            num_workers=8,
        )
    elif forecast_type == "continuous":
        dm = cl.data.IterDataModule(
            "continuous-forecasting",
            era5_dir,
            era5_dir,
            in_vars,
            out_vars,
            src="era5",
            history=3,
            window=6,
            pred_range=1,
            max_pred_range=120,
            random_lead_time=True,
            hrs_each_step=1,
            subsample=6,
            batch_size=batch_size,
            buffer_size=2000,
            num_workers=8,
        )
    
    dm.setup()
    return dm, in_vars, out_vars


def create_model_architecture(in_channels, out_channels, device='cpu'):
    """
    Create the res_slim_vit model architecture.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        device: Device to place the model on
    
    Returns:
        Configured climate_learn model
    """
    # Model configuration (same as training)
    model_kwargs = {
        "img_size": (32, 64),
        "in_channels": in_channels,
        "out_channels": out_channels,
        "history": 3,
        "superres_factor": 1,  # no upscaling
        "cnn_ratio": 4,
        "patch_size": 2,
        "embed_dim": 128,
        "depth": 8,
        "decoder_depth": 2,
        "learn_pos_emb": True,
        "num_heads": 4,
    }
    
    # Optimizer configuration (for reference, not used in inference)
    optim_kwargs = {"lr": 5e-4, "weight_decay": 1e-5, "betas": (0.9, 0.99)}
    sched_kwargs = {
        "warmup_epochs": 5,
        "max_epochs": 50,
        "warmup_start_lr": 1e-8,
        "eta_min": 1e-8,
    }
    
    return model_kwargs, optim_kwargs, sched_kwargs


def load_checkpoint(checkpoint_path, era5_dir, forecast_type="direct", pred_range=120, device='cuda'):
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to the .ckpt file
        era5_dir: Path to ERA5 data directory
        forecast_type: Type of forecasting (direct, iterative, or continuous)
        pred_range: Prediction range in hours
        device: Device to load the model on ('cuda' or 'cpu')
    
    Returns:
        Loaded model and data module
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Setup data module
    print("Setting up data module...")
    dm, in_vars, out_vars = setup_data_module(era5_dir, forecast_type, pred_range)
    
    # Calculate channels
    in_channels = 42  # As per training script
    if forecast_type == "continuous":
        in_channels += 1  # time dimension
    
    if forecast_type == "iterative":
        out_channels = in_channels
    else:
        out_channels = len([
            "2m_temperature", 
            "geopotential_500", 
            "temperature_850", 
            "total_precipitation", 
            "10m_u_component_of_wind", 
            "10m_v_component_of_wind"
        ])
    
    print(f"Model configuration: in_channels={in_channels}, out_channels={out_channels}")
    
    # Create model architecture
    model_kwargs, optim_kwargs, sched_kwargs = create_model_architecture(in_channels, out_channels, device)
    
    # Create the model with the same architecture
    print("Creating model architecture...")
    model = cl.load_forecasting_module(
        data_module=dm,
        model="res_slimvit",
        model_kwargs=model_kwargs,
        optim="adamw",
        optim_kwargs=optim_kwargs,
        sched="linear-warmup-cosine-annealing",
        sched_kwargs=sched_kwargs,
        device=device,
    )
    
    # Load checkpoint weights
    print("Loading checkpoint weights...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    
    # Set model to evaluation mode
    model.eval()
    print("Model loaded successfully!")
    
    return model, dm


def iterative_testing(model, trainer, args, dm, in_vars, out_vars):
    """Run iterative testing for different lead times."""
    print("\n" + "="*60)
    print("Running iterative testing for multiple lead times...")
    print("="*60)
    
    for lead_time in [6, 24, 72, 120, 240]:
        print(f"\nTesting lead time: {lead_time} hours")
        n_iters = lead_time // args.pred_range
        model.set_mode("iter")
        model.set_n_iters(n_iters)
        test_dm = cl.data.IterDataModule(
            "iterative-forecasting",
            args.era5_dir,
            args.era5_dir,
            in_vars,
            out_vars,
            src="era5",
            history=3,
            window=6,
            pred_range=lead_time,
            subsample=1,
        )
        trainer.test(model, datamodule=test_dm)


def continuous_testing(model, trainer, args, dm, in_vars, out_vars):
    """Run continuous testing for different lead times."""
    print("\n" + "="*60)
    print("Running continuous testing for multiple lead times...")
    print("="*60)
    
    for lead_time in [6, 24, 72, 120, 240]:
        print(f"\nTesting lead time: {lead_time} hours")
        test_dm = cl.data.IterDataModule(
            "continuous-forecasting",
            args.era5_dir,
            args.era5_dir,
            in_vars,
            out_vars,
            src="era5",
            history=3,
            window=6,
            pred_range=lead_time,
            max_pred_range=lead_time,
            random_lead_time=False,
            hrs_each_step=1,
            subsample=1,
            batch_size=128,
            buffer_size=2000,
            num_workers=8,
        )
        trainer.test(model, datamodule=test_dm)


def main():
    parser = ArgumentParser(description="Load a trained res_slim_vit checkpoint and run testing")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the checkpoint file (.ckpt)"
    )
    parser.add_argument(
        "--era5_dir",
        type=str,
        required=True,
        help="Path to ERA5 data directory"
    )
    parser.add_argument(
        "--forecast_type",
        type=str,
        default="direct",
        choices=["direct", "iterative", "continuous"],
        help="Type of forecasting"
    )
    parser.add_argument(
        "--pred_range",
        type=int,
        default=120,
        help="Prediction range in hours (used during training)"
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of GPUs to use for testing"
    )
    parser.add_argument(
        "--skip_test",
        action="store_true",
        help="Skip running test evaluation (only load model)"
    )
    
    args = parser.parse_args()
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load the checkpoint
    model, dm = load_checkpoint(
        args.checkpoint,
        args.era5_dir,
        args.forecast_type,
        args.pred_range,
        device
    )
    
    print("\n" + "="*60)
    print("Model loaded successfully!")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Forecast type: {args.forecast_type}")
    print(f"Prediction range: {args.pred_range} hours")
    print(f"Device: {device}")
    print(f"Number of devices: {args.devices}")
    print("="*60)
    
    # Run test evaluation (unless skipped)
    if not args.skip_test:
        print("\n" + "="*60)
        print("Starting test evaluation...")
        print("="*60)
        
        # Setup trainer for testing
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={Block}
        )
        
        if args.devices > 1:
            strategy = FSDPStrategy(
                auto_wrap_policy=auto_wrap_policy,
                activation_checkpointing=Block
            )
        else:
            strategy = "auto"
        
        trainer = pl.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=args.devices,
            strategy=strategy,
            precision="bf16-mixed",
        )
        
        # Get in_vars and out_vars from data module for multi-lead-time testing
        variables = [
            "geopotential",
            "temperature",
            "u_component_of_wind",
            "v_component_of_wind",
            "specific_humidity",
            "2m_temperature",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "land_sea_mask",
            "orography",
            "lattitude",
            "total_precipitation",
        ]
        
        in_vars = []
        for var in variables:
            if var in PRESSURE_LEVEL_VARS:
                for level in DEFAULT_PRESSURE_LEVELS:
                    in_vars.append(var + "_" + str(level))
            else:
                in_vars.append(var)
        
        if args.forecast_type in ("direct", "continuous"):
            out_variables = [
                "2m_temperature", 
                "geopotential_500", 
                "temperature_850", 
                "total_precipitation", 
                "10m_u_component_of_wind", 
                "10m_v_component_of_wind"
            ]
        elif args.forecast_type == "iterative":
            out_variables = variables
        
        out_vars = []
        for var in out_variables:
            if var in PRESSURE_LEVEL_VARS:
                for level in DEFAULT_PRESSURE_LEVELS:
                    out_vars.append(var + "_" + str(level))
            else:
                out_vars.append(var)
        
        # Run appropriate testing regime
        if args.forecast_type == "direct":
            print("\nRunning direct forecasting test...")
            trainer.test(model, datamodule=dm)
        elif args.forecast_type == "iterative":
            iterative_testing(model, trainer, args, dm, in_vars, out_vars)
        elif args.forecast_type == "continuous":
            continuous_testing(model, trainer, args, dm, in_vars, out_vars)
        
        print("\n" + "="*60)
        print("Testing complete!")
        print("="*60)
    else:
        print("\nSkipping test evaluation. Model is ready for inference!")
    
    return model, dm


if __name__ == "__main__":
    model, dm = main()

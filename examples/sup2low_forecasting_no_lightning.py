# Standard library
from argparse import ArgumentParser
import os
import sys
import time
import yaml
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import wrap, transformer_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
    CheckpointWrapper,
)
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data._utils.collate import default_collate
from torch.nn import SyncBatchNorm
from functools import partial
from datetime import timedelta
import functools

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
from climate_learn.models.hub.components.cnn_blocks import (
    DownBlock,
    MiddleBlock,
    UpBlock,
    ResidualBlock,
)
from climate_learn.utils.fused_attn import FusedAttn
from climate_learn.models.hub.components.pos_embed import interpolate_pos_embed
from climate_learn.data.transforms import collate_resize, collate_batch_only
from utils import seed_everything, init_par_groups


def log_gpu_memory(device, message="", world_rank=None):
    """Log GPU memory usage with optional message and rank."""
    memory_allocated = torch.cuda.memory_allocated(device) / 1024 / 1024 / 1024
    memory_reserved = torch.cuda.memory_reserved(device) / 1024 / 1024 / 1024
    if world_rank is not None:
        print(
            f"rank {world_rank} {message} Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB",
            flush=True,
        )
    else:
        print(f"{message} Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB", flush=True)


def get_checkpoint_filename(save_path, epoch, world_rank, tensor_par_size):
    """Generate checkpoint filename for given epoch and rank."""
    base_filename = f"{save_path}/epoch_{epoch}.ckpt"
    if tensor_par_size > 1:
        return f"{base_filename}_rank_{world_rank}"
    return base_filename


def clip_replace_constant(y, yhat, out_variables):
    # Clip precipitation to non-negative values if it exists
    if "total_precipitation_24hr" in out_variables:
        prcp_index = out_variables.index("total_precipitation_24hr")
        torch.clamp_(yhat[:, prcp_index, :, :], min=0.0)

    # Replace constant variables (like land_sea_mask, orography) with ground-truth values
    for i in range(yhat.shape[1]):
        if out_variables[i] in CONSTANTS:
            yhat[:, i] = y[:, i]
    return yhat


def training_step(
    batch, batch_idx, net, device: int, var_weights, train_loss_metric, resize_config=None
) -> torch.Tensor:
    x, y, in_variables, out_variables = batch
    x = x.to(device)
    y = y.to(device)
    
    # GPU-based interpolation (much faster than CPU collate_resize)
    if resize_config is not None:
        # lr_h, lr_w = resize_config['lr_size']
        # hr_h, hr_w = resize_config['hr_size']
        mode = resize_config['mode']
        scale = resize_config['scale_factor']
        
        # Dynamic resizing (safe for tiles)
        if mode in ['linear', 'bilinear', 'bicubic', 'trilinear']:
            x = torch.nn.functional.interpolate(
                x, scale_factor=scale, mode=mode, align_corners=False, recompute_scale_factor=False
            )
        else:
            x = torch.nn.functional.interpolate(
                x, scale_factor=scale, mode=mode, recompute_scale_factor=False
            )
        
        # Interpolate input to low-resolution on GPU
        # x shape: [B, C, H, W] - downsample to model's input size
        # if mode in ['linear', 'bilinear', 'bicubic', 'trilinear']:
        #     x = torch.nn.functional.interpolate(
        #         x, size=(lr_h, lr_w), mode=mode, align_corners=False
        #     )
        # else:
        #     x = torch.nn.functional.interpolate(
        #         x, size=(lr_h, lr_w), mode=mode
        #     )
        
        # Interpolate target to high-resolution on GPU  
        # y shape: [B, C, H, W] - upsample/keep at target size
        # if mode in ['linear', 'bilinear', 'bicubic', 'trilinear']:
        #     y = torch.nn.functional.interpolate(
        #         y, size=(hr_h, hr_w), mode=mode, align_corners=False
        #     )
        # else:
        #     y = torch.nn.functional.interpolate(
        #         y, size=(hr_h, hr_w), mode=mode
        #     )

    yhat = net.forward(x, in_variables, out_variables)
    yhat = clip_replace_constant(y, yhat, out_variables)

    # lat_mse doesn't accept var_names/var_weights, only pred and target
    if y.size(dim=2) != yhat.size(dim=2) or y.size(dim=3) != yhat.size(dim=3):
        losses = train_loss_metric(
            yhat,
            y[:, :, 0 : yhat.size(dim=2), 0 : yhat.size(dim=3)],
        )
    else:
        losses = train_loss_metric(yhat, y)
    
    loss_name = getattr(train_loss_metric, "name", "loss")
    if losses.dim() == 0:  # aggregate loss only
        loss = losses
    else:  # per channel + aggregate
        loss = losses[-1]

    return loss


def validation_step(
    batch, batch_idx: int, net, device: int, val_loss_metrics, val_target_transforms, resize_config=None
) -> torch.Tensor:

    return evaluate_func(
        batch, "val", net, device, val_loss_metrics, val_target_transforms, resize_config
    )


def evaluate_func(batch, stage: str, net, device: int, loss_metrics, target_transforms, resize_config=None):

    x, y, in_variables, out_variables = batch
    x = x.to(device)
    y = y.to(device)
    
    # GPU-based interpolation (same as training_step)
    if resize_config is not None:
        # lr_h, lr_w = resize_config['lr_size']
        # hr_h, hr_w = resize_config['hr_size']
        mode = resize_config['mode']
        scale = resize_config['scale_factor']
        
        # Dynamic resizing (safe for tiles)
        if mode in ['linear', 'bilinear', 'bicubic', 'trilinear']:
            x = torch.nn.functional.interpolate(
                x, scale_factor=scale, mode=mode, align_corners=False, recompute_scale_factor=False
            )
        else:
            x = torch.nn.functional.interpolate(
                x, scale_factor=scale, mode=mode, recompute_scale_factor=False
            )
        
        # if mode in ['linear', 'bilinear', 'bicubic', 'trilinear']:
        #     x = torch.nn.functional.interpolate(
        #         x, size=(lr_h, lr_w), mode=mode, align_corners=False
        #     )
        # else:
        #     x = torch.nn.functional.interpolate(
        #         x, size=(lr_h, lr_w), mode=mode
        #     )
        
        # if mode in ['linear', 'bilinear', 'bicubic', 'trilinear']:
        #     y = torch.nn.functional.interpolate(
        #         y, size=(hr_h, hr_w), mode=mode, align_corners=False
        #     )
        # else:
        #     y = torch.nn.functional.interpolate(
        #         y, size=(hr_h, hr_w), mode=mode
        #     )

    yhat = net.forward(x, in_variables, out_variables)
    yhat = clip_replace_constant(y, yhat, out_variables)

    if stage == "val":
        loss_fns = loss_metrics
        transforms = target_transforms
    elif stage == "test":
        loss_fns = loss_metrics
        transforms = target_transforms
    else:
        raise RuntimeError("Invalid evaluation stage")
    
    loss_dict = {}
    for i, lf in enumerate(loss_fns):
        if transforms is not None and transforms[i] is not None:
            yhat_ = transforms[i](yhat)
            y_ = transforms[i](y)
        else:
            yhat_ = yhat
            y_ = y
            
        # Validation losses (lat_rmse, lat_acc, lat_mse) don't take var_names
        # They already have variable info from MetricsMetaInfo passed during creation
        if y_.size(dim=2) != yhat_.size(dim=2) or y_.size(dim=3) != yhat_.size(dim=3):
            losses = lf(yhat_, y_[:, :, 0 : yhat_.size(dim=2), 0 : yhat_.size(dim=3)])
        else:
            losses = lf(yhat_, y_)

        loss_name = getattr(lf, "name", f"loss_{i}")
        if losses.dim() == 0:  # aggregate loss only (scalar)
            loss_dict[f"{stage}/{loss_name}:aggregate"] = losses
        else:  # per-variable losses returned as tensor
            # Check if losses tensor has more than one element (per-variable + aggregate)
            if losses.numel() > 1:
                # Iterate through each variable and corresponding loss value
                for idx, var_name in enumerate(out_variables):
                    if idx < losses.numel():
                        name = f"{stage}/{loss_name}:{var_name}"
                        loss_dict[name] = losses[idx]
                # Last element is typically the aggregate
                loss_dict[f"{stage}/{loss_name}:aggregate"] = losses[-1]
            else:
                # Single element tensor, treat as aggregate
                loss_dict[f"{stage}/{loss_name}:aggregate"] = losses.item() if losses.numel() == 1 else losses
    return loss_dict


def setup_environment():
    """Set up distributed training environment from SLURM."""
    world_size = int(os.environ["SLURM_NTASKS"])
    world_rank = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ["SLURM_LOCALID"])

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    print(f"Rank {world_rank}/{world_size} using device {device}", flush=True)

    return device, world_size, world_rank, local_rank


def parse_config(config_path, world_rank):
    """Parse configuration from YAML file."""
    if world_rank == 0:
        print(f"Loading config from: {config_path}", flush=True)

    with open(config_path, "r") as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)

    return conf


def create_model_and_losses(config, device, world_rank, in_vars, out_vars, data_module):
    """
    Create model, optimizer, scheduler, and loss functions.
    
    Returns:
        tuple: (model, optimizer, scheduler, train_loss, val_losses, val_transforms)
    """
    # Extract configuration
    preset = config["model"]["preset"]
    lr_h = config["model"]["img_size"][0]
    lr_w = config["model"]["img_size"][1]
    superres_factor = config["model"]["superres_factor"]
    
    # Calculate actual number of channels (pressure levels are already expanded in in_vars/out_vars)
    in_channels = len(in_vars)
    out_channels = len(out_vars)
    history = config["model"]["history"]
    
    if world_rank == 0:
        print(f"Creating model: {preset}", flush=True)
        print(f"Model config: img_size=({lr_h}, {lr_w}), in_channels={in_channels}, out_channels={out_channels}, history={history}, superres_factor={superres_factor}", flush=True)

    # Create model directly without using load_forecasting_module
    if preset == "res_slimvit":
        from climate_learn.models.hub import Res_Slim_ViT
        from climate_learn.utils.fused_attn import FusedAttn
        
        # Count constant variables (geographical features that don't change over time)
        constant_vars = ["land_sea_mask", "orography", "lattitude", "landcover"]
        num_constant_vars = sum(1 for var in constant_vars if var in in_vars)
        
        # CRITICAL: Model needs to know about ALL variables (input + output) for var_map
        # Create a list of all unique variables, preserving order
        all_vars = []
        seen = set()
        for var in in_vars + out_vars:
            if var not in seen:
                all_vars.append(var)
                seen.add(var)
        
        model = Res_Slim_ViT(
            default_vars=all_vars,  # Pass ALL variables, not just output variables
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
            # num_constant_vars=num_constant_vars,
            FusedAttn_option=FusedAttn.DEFAULT,  # Use PyTorch native attention instead of xformers (ROCm compatibility)
        )
    else:
        raise NotImplementedError(f"Model preset {preset} not implemented")
    
    # Convert to SyncBatchNorm for distributed training
    # model = SyncBatchNorm.convert_sync_batchnorm(model)
    
    # NOTE: Do NOT move model to device before FSDP wrapping
    # FSDP will handle device placement automatically
    
    if world_rank == 0:
        print(
            f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M",
            flush=True,
        )

    # Create optimizer
    base_lr = float(config["model"]["base_lr"])
    weight_decay = float(config["model"]["weight_decay"])
    beta_1 = float(config["model"]["beta_1"])
    beta_2 = float(config["model"]["beta_2"])
    num_gpus = int(config["trainer"]["num_gpus"])
    
    scaling_factor = (num_gpus ** 0.8) # was 0.75
    lr = base_lr * scaling_factor
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(beta_1, beta_2)
    )

    # Create scheduler
    warmup_epochs = config["model"]["warmup_epochs"]
    max_epochs = config["trainer"]["max_epochs"]
    warmup_start_lr = float(config["model"]["warmup_start_lr"])
    eta_min = float(config["model"]["eta_min"])
    
    scheduler = cl.load_lr_scheduler(
        "linear-warmup-cosine-annealing",
        optimizer,
        {
            "warmup_epochs": warmup_epochs,
            "max_epochs": max_epochs,
            "warmup_start_lr": warmup_start_lr,
            "eta_min": eta_min,
        },
    )
    
    # NOTE: Delay climatology loading until AFTER FSDP wrapping
    # Climatology creates tensors that can interfere with FSDP initialization
    
    return model, optimizer, scheduler


def create_losses_after_fsdp(config, device, world_rank, model, data_module, in_vars, out_vars):
    """
    Create loss functions AFTER FSDP wrapping.
    """
    # Get lat/lon for validation metrics
    lat, lon = data_module.get_lat_lon()
    
    # Load training climatology for lat_mse
    if world_rank == 0:
        print("Loading training climatology for lat_mse (this may take a while)...", flush=True)
    train_clim = get_climatology(data_module, split="train")
    
    # Load validation climatology
    if world_rank == 0:
        print("Loading validation climatology...", flush=True)
    val_clim = get_climatology(data_module, split="val")
    
    # Create MetricsMetaInfo for training and validation
    train_metainfo = MetricsMetaInfo(in_vars, out_vars, lat, lon, train_clim)
    val_metainfo = MetricsMetaInfo(in_vars, out_vars, lat, lon, val_clim)
    
    # Load training loss - lat_mse (latitude-weighted MSE with climatology)
    if world_rank == 0:
        print("Using lat_mse for training loss", flush=True)
    train_loss = cl.load_loss(device, model, "lat_mse", True, train_metainfo)
    
    # Validation losses (these use validation climatology, which is small)
    val_losses = [
        cl.load_loss(device, model, "lat_rmse", False, val_metainfo),
        cl.load_loss(device, model, "lat_acc", False, val_metainfo),
        cl.load_loss(device, model, "lat_mse", False, val_metainfo),
    ]
    
    # Validation transforms
    val_transforms = [
        cl.load_transform("denormalize", data_module),
        cl.load_transform("denormalize", data_module),
        None,
    ]
    
    if world_rank == 0:
        print("Loss functions created successfully", flush=True)
    
    return train_loss, val_losses, val_transforms

def create_data_module(config, world_rank, device, do_tiling=False, div=1, overlap=0):
    """
    Create data module and loaders.
    
    Args:
        config (dict): Configuration dictionary
        world_rank (int): Process rank
        device: Training device
        do_tiling (bool): Whether to use TILES algorithm for large images
        div (int): Tile division factor (images split into div×div tiles)
        overlap (int): Tile overlap in pixels
    
    Returns:
        tuple: (data_module, train_dataloader, val_dataloader, in_vars, out_vars)
    """
    era5_dir = config["data"]["era5_dir"]
    forecast_type = config["data"]["forecast_type"]
    pred_range = config["data"]["pred_range"]
    
    # Define variables
    variables = config["data"]["variables"]
    in_vars = []
    for var in variables:
        if var in PRESSURE_LEVEL_VARS:
            for level in DEFAULT_PRESSURE_LEVELS:
                in_vars.append(var + "_" + str(level))
        else:
            in_vars.append(var)
    
    out_variables = config["data"]["out_variables"]
    out_vars = []
    for var in out_variables:
        if var in PRESSURE_LEVEL_VARS:
            for level in DEFAULT_PRESSURE_LEVELS:
                out_vars.append(var + "_" + str(level))
        else:
            out_vars.append(var)
    
    if world_rank == 0:
        print(f"Creating data module for {forecast_type} forecasting", flush=True)
        print(f"Input variables: {in_vars}", flush=True)
        print(f"Output variables: {out_vars}", flush=True)

    # Create data module
    batch_size = config["trainer"]["batch_size"]
    num_workers = config["trainer"]["num_workers"]
    history = config["model"]["history"]
    
    # CRITICAL: Pass distributed parameters so each GPU reads only its chunk of data!
    # Without this, all 8 GPUs read the entire 40-year dataset simultaneously
    world_size_value = int(os.environ.get("SLURM_NTASKS", 1))
    
    if world_rank == 0:
        print(f"Setting up distributed data loading: world_size={world_size_value}", flush=True)
        print(f"Each GPU will process 1/{world_size_value} of the dataset", flush=True)
        
    # Add buffer_size explicitly
    buffer_size = config["trainer"]["buffer_size"]
    
    if forecast_type in ("direct", "iterative"):
        data_module = cl.data.IterDataModule(
            f"{forecast_type}-forecasting",
            era5_dir,
            era5_dir,
            in_vars,
            out_vars,
            data_par_size=world_size_value,  # Enable distributed data sharding
            data_par_group=None,  # Use default process group
            src="era5",
            history=history,
            window=6,
            pred_range=pred_range,
            subsample=6,
            batch_size=batch_size,
            num_workers=num_workers,
            buffer_size=buffer_size,
            div=div,          
            overlap=overlap,
        )
    elif forecast_type == "continuous":
        data_module = cl.data.IterDataModule(
            "continuous-forecasting",
            era5_dir,
            era5_dir,
            in_vars,
            out_vars,
            data_par_size=world_size_value,  # Enable distributed data sharding
            data_par_group=None,  # Use default process group
            src="era5",
            history=history,
            window=6,
            pred_range=1,
            max_pred_range=120,
            random_lead_time=True,
            hrs_each_step=1,
            subsample=6,
            batch_size=batch_size,
            num_workers=num_workers,
            buffer_size=buffer_size,
            div=div,          
            overlap=overlap,
        )
    
    lr_h, lr_w = config["model"]["img_size"]
    superres_factor = config["model"]["superres_factor"]
    hr_h, hr_w = lr_h * superres_factor, lr_w * superres_factor
    downsample_mode = config["model"]["downsample_mode"]
    
    # NEW: Use custom collate function for batching (without interpolation)
    # Interpolation will be done on GPU in training_step for better performance
    data_module.collate_fn = collate_batch_only
    
    # Store these parameters for GPU interpolation later
    if not hasattr(data_module, 'resize_config'):
        data_module.resize_config = {
            'lr_size': (lr_h, lr_w),
            'hr_size': (hr_h, hr_w),
            'mode': downsample_mode,
            'scale_factor': 1.0 / superres_factor
        }

    # Check tiling compatibility with patch size
    if do_tiling:
        patch_size = config["model"]["patch_size"]
        # For super-resolution forecasting: tiles are in low-res space
        # Calculate tile dimensions based on division factor
        yout = hr_h // div  # Output tile height in high-res space
        yinp = yout // superres_factor + overlap  # Input tile height in low-res space

        if yinp % patch_size != 0:
            if world_rank == 0:
                print(f"Tile height: {yinp}, patch_size {patch_size}", flush=True)
                print(
                    f"Overlap must be adjusted to accommodate patch_size. Need to increase by {yinp % patch_size}",
                    flush=True,
                )
            sys.exit("Please adjust overlap according to the instructions above")

    if world_rank == 0:
        log_gpu_memory(device, f"after data_module creation")

    # Setup data module (creates datasets)
    data_module.setup()

    # Create data loaders
    # Tiling is handled at the IterDataModule level via div and overlap parameters
    train_dataloader = data_module.train_dataloader()
    
    val_dataloader = data_module.val_dataloader()

    return data_module, train_dataloader, val_dataloader, in_vars, out_vars


def train_epoch(
    model,
    train_dataloader,
    optimizer,
    scheduler,
    scaler,
    epoch,
    world_rank,
    device,
    var_weights,
    train_loss,
    in_vars,
    out_vars,
    data_module=None,
    data_type="float32",
    log_interval=50,
):
    """Train model for one epoch."""
    model.train()
    epoch_loss = 0.0
    num_batches = 0

    if world_rank == 0:
        print(f"\n{'='*80}")
        print(f"Starting epoch {epoch}")
        print(f"{'='*80}\n", flush=True)

    for batch_idx, batch in enumerate(train_dataloader):
        # Time measurement
        if world_rank == 0:
            torch.cuda.synchronize(device=device)
            tic = time.perf_counter()

        # Forward pass (with GPU interpolation)
        resize_config = getattr(data_module, 'resize_config', None) if data_module is not None else None
        loss = training_step(batch, batch_idx, model, device, var_weights, train_loss, resize_config)
        epoch_loss += loss.item()
        num_batches += 1

        # Backward pass
        optimizer.zero_grad()

        if data_type == "bfloat16":
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            # Ensure minimum scale
            if hasattr(scaler, "_scale") and scaler._scale < 128:
                scaler._scale = torch.tensor(128.0, device=device)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        # Logging
        if world_rank == 0 and batch_idx % log_interval == 0:
            torch.cuda.synchronize(device=device)
            toc = time.perf_counter()
            time_per_batch = (toc - tic)
            
            print(
                f"Epoch {epoch} | Batch {batch_idx} | "
                f"Loss: {loss.item():.6f} | "
                f"Time: {time_per_batch:.3f}s | "
                f"LR: {scheduler.get_last_lr()[0]:.2e}",
                flush=True
            )
            log_gpu_memory(device, f"Batch {batch_idx}", world_rank)

    # Step scheduler
    scheduler.step()

    avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0

    if world_rank == 0:
        print(f"\n{'='*80}")
        print(f"Epoch {epoch} completed | Average Loss: {avg_loss:.6f}")
        print(f"{'='*80}\n", flush=True)

    return avg_loss


def validate_epoch(
    model,
    val_dataloader,
    epoch,
    world_rank,
    device,
    val_loss_metrics,
    val_target_transforms,
    in_vars,
    out_vars,
    data_module=None,
):
    """Validate model for one epoch."""
    model.eval()
    
    # Initialize loss accumulation
    loss_dict_sum = {}
    num_batches = 0

    if world_rank == 0:
        print(f"\nRunning validation for epoch {epoch}...", flush=True)

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_dataloader):
            resize_config = getattr(data_module, 'resize_config', None) if data_module is not None else None
            loss_dict = validation_step(
                batch, batch_idx, model, device, val_loss_metrics, val_target_transforms, resize_config
            )
            
            # Accumulate losses
            for key, value in loss_dict.items():
                if key not in loss_dict_sum:
                    loss_dict_sum[key] = 0.0
                loss_dict_sum[key] += value.item()
            
            num_batches += 1

    # Average losses
    if num_batches > 0:
        for key in loss_dict_sum:
            loss_dict_sum[key] /= num_batches

    if world_rank == 0:
        print(f"\nValidation results for epoch {epoch}:")
        for key, value in loss_dict_sum.items():
            print(f"  {key}: {value:.6f}")
        print(flush=True)

    return loss_dict_sum


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    cp_save_path,
    world_rank,
    local_rank,
    tensor_par_size,
    device,
):
    """Save model checkpoint."""
    if world_rank == 0:
        if not os.path.exists(cp_save_path):
            os.makedirs(cp_save_path)
            print(f"Created checkpoint directory: {cp_save_path}", flush=True)

    if world_rank == 0:
        log_gpu_memory(device, "Before saving checkpoint", world_rank)
        
    
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig

    # Configure policy: Offload to CPU to save VRAM, gather on Rank 0 only
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)

    # ALL ranks must enter this context manager and call state_dict()
    # FSDP handles the communication so that only Rank 0 receives the full dict
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        model_states = model.state_dict()

    # Optimizer and Scheduler also need careful handling with FSDP, 
    # but for standard DDP/FSDP mix, basic state_dict is usually fine provided
    # you aren't sharding optimizer states (which you are likely doing implicitly with FSDP).
    # For now, we will grab them on all ranks, but FSDP optimizer saving is complex.
    # To keep it simple and working like your script intended:
    optimizer_states = optimizer.state_dict() 
    scheduler_states = scheduler.state_dict()
    

    # Only Rank 0 saves to disk
    if world_rank == 0:
        file_name = os.path.join(cp_save_path, f"checkpoint_epoch_{epoch:04d}.pt")
        
        checkpoint_dict = {
            "epoch": epoch,
            "model_state_dict": model_states,
            "optimizer_state_dict": optimizer_states,
            "scheduler_state_dict": scheduler_states,
        }
        
        torch.save(checkpoint_dict, file_name)
        print(f"Saved checkpoint to: {file_name}", flush=True)
        
    # Synchronize all ranks after checkpoint saving
    dist.barrier()

    if world_rank == 0:
        log_gpu_memory(device, "After saving checkpoint", world_rank)

    # Clean up to free memory (important!)
    del model_states
    del optimizer_states
    del scheduler_states


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, world_rank, reset_optimizer=False):
    """Load checkpoint and return start epoch.
    
    Args:
        reset_optimizer: If True, only load model weights and reset optimizer state.
                        Useful when resuming with different FSDP/activation checkpointing config.
    """
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        return 0
    
    if world_rank == 0:
        print(f"Loading checkpoint from: {checkpoint_path}", flush=True)
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Always load model state
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Optionally load optimizer and scheduler state
    if reset_optimizer:
        if world_rank == 0:
            print("WARNING: Resetting optimizer and scheduler state (keeping only model weights)", flush=True)
            print("This is normal when resuming with different FSDP/activation checkpointing settings", flush=True)
    else:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        except (RuntimeError, ValueError) as e:
            if world_rank == 0:
                print(f"WARNING: Failed to load optimizer/scheduler state: {e}", flush=True)
                print("Continuing with fresh optimizer state...", flush=True)
    
    start_epoch = checkpoint["epoch"] + 1
    
    if world_rank == 0:
        print(f"Resumed from epoch {checkpoint['epoch']}", flush=True)
    
    del checkpoint
    
    return start_epoch


def main(device):
    """Main training function."""
    # Setup environment
    world_size = int(os.environ["SLURM_NTASKS"])
    world_rank = dist.get_rank()
    local_rank = int(os.environ["SLURM_LOCALID"])

    print(
        f"world_size={world_size}, world_rank={world_rank}, local_rank={local_rank}",
        flush=True,
    )

    # Parse config
    config_path = sys.argv[1]
    config = parse_config(config_path, world_rank)

    # Extract key parameters
    max_epochs = config["trainer"]["max_epochs"]
    checkpoint_path = config["trainer"].get("checkpoint", None)
    data_type = config["trainer"].get("data_type", "float32")
    cp_save_path = config["trainer"].get("checkpoint_save_path", "checkpoints/forecasting")
    
    # Parallelism config
    fsdp_size = config["parallelism"].get("fsdp", 1)
    simple_ddp_size = config["parallelism"].get("simple_ddp", 1)
    tensor_par_size = config["parallelism"].get("tensor_par", 1)
    seq_par_size = config["parallelism"].get("seq_par", 1)
    
    # Tiling config for large images
    try:
        do_tiling = config["tiling"]["do_tiling"]
        if do_tiling:
            div = config["tiling"]["div"]
            overlap = config["tiling"]["overlap"]
        else:
            div = 1
            overlap = 0
    except (KeyError, TypeError):
        do_tiling = False
        div = 1
        overlap = 0

    if world_rank == 0:
        print("\n" + "=" * 80)
        print("Training Configuration Summary")
        print("=" * 80)
        print(f"Config: {config_path}")
        print(f"Max epochs: {max_epochs}")
        print(f"Data type: {data_type}")
        print(f"Checkpoint: {checkpoint_path if checkpoint_path else 'None'}")
        print(f"Save path: {cp_save_path}")
        print("=" * 80 + "\n", flush=True)

    # Seed everything
    seed_everything(42)

    # Create data module
    data_module, train_dataloader, val_dataloader, in_vars, out_vars = create_data_module(
        config, world_rank, device, do_tiling, div, overlap
    )

    # Create model and optimizer (WITHOUT losses yet)
    model, optimizer, scheduler = create_model_and_losses(
        config, device, world_rank, in_vars, out_vars, data_module
    )
    
    # FSDP CONFIGURATION with activation checkpointing for memory efficiency
    # This matches PyTorch Lightning's strategy to handle large 128x256 images
    
    if world_rank == 0:
        print("Using FSDP with activation checkpointing for memory efficiency...", flush=True)
    
    # Setup mixed precision policy
    if data_type == "bfloat16":
        bfloat_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    else:
        bfloat_policy = None
        
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Block}  # Wrap each Transformer block
    )
        
    model = FSDP(
        model,
        device_id=local_rank,
        mixed_precision=bfloat_policy,
        auto_wrap_policy=auto_wrap_policy,  # Enable per-layer wrapping of Block layers
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # CRITICAL: Actually shard parameters!
        use_orig_params=True,  # Keep True for ROCm stability (False causes RCCL crashes)
    )
    
    # Now apply activation checkpointing AFTER FSDP wrapping (critical for memory!)
    # This reduces memory by ~3x by recomputing activations during backward pass
    # Use REENTRANT mode - more stable with use_orig_params=True on ROCm
    reentrant_wrapper = functools.partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.REENTRANT,
    )
    # non_reentrant_wrapper = functools.partial(
    #     checkpoint_wrapper,
    #     checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    # )
    
    check_fn = lambda submodule: isinstance(submodule, Block)
    
    apply_activation_checkpointing(
        model, 
        checkpoint_wrapper_fn=reentrant_wrapper, #non_reentrant_wrapper
        check_fn=check_fn
    )
    
    if world_rank == 0:
        print("FSDP wrapping with activation checkpointing completed", flush=True)

    if world_rank == 0:
        print("FSDP wrapping completed, now creating losses...", flush=True)

    # Create losses AFTER FSDP wrapping (climatology can interfere with FSDP init)
    train_loss, val_losses, val_transforms = create_losses_after_fsdp(
        config, device, world_rank, model, data_module, in_vars, out_vars
    )

    if world_rank == 0:
        print("Losses created successfully", flush=True)

    # Create gradient scaler for mixed precision
    if data_type == "bfloat16":
        scaler = ShardedGradScaler(init_scale=8192, growth_interval=100)
    else:
        scaler = None

    # Load checkpoint if available
    # Note: Set reset_optimizer=True if resuming with different FSDP settings
    start_epoch = load_checkpoint(checkpoint_path, model, optimizer, scheduler, world_rank, reset_optimizer=True)

    # Variable weights for loss
    var_weights = config["data"].get("var_weights", {})

    # Training loop
    for epoch in range(start_epoch, max_epochs):
        # # Clear memory cache before each epoch to prevent fragmentation
        # if epoch > 0:
        #     torch.cuda.empty_cache()
        #     if world_rank == 0:
        #         print(f"Cleared CUDA cache before epoch {epoch}", flush=True)

        # Train
        avg_loss = train_epoch(
            model,
            train_dataloader,
            optimizer,
            scheduler,
            scaler,
            epoch,
            world_rank,
            device,
            var_weights,
            train_loss,
            in_vars,
            out_vars,
            data_module,
            data_type=data_type,
            log_interval=50,
        )

        # Validate (optional, can be skipped for speed)
        if (epoch + 1) % 5 == 0:  # Validate every 5 epochs
            val_losses_dict = validate_epoch(
                model,
                val_dataloader,
                epoch,
                world_rank,
                device,
                val_losses,
                val_transforms,
                in_vars,
                out_vars,
                data_module,
            )

        # Save checkpoint
        if (epoch + 1) % 1 == 0:  # Save every epoch
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                cp_save_path,
                world_rank,
                local_rank,
                tensor_par_size,
                device,
            )

    if world_rank == 0:
        print("\n" + "=" * 80)
        print("Training completed!")
        print("=" * 80 + "\n", flush=True)


if __name__ == "__main__":
    
    os.environ["MASTER_ADDR"] = str(os.environ["HOSTNAME"])
    os.environ["MASTER_PORT"] = "29500"
    os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
    os.environ["RANK"] = os.environ["SLURM_PROCID"]

    world_size = int(os.environ["SLURM_NTASKS"])
    world_rank = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ["SLURM_LOCALID"])

    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()

    dist.init_process_group(
        "nccl",
        timeout=timedelta(seconds=7200),
        rank=world_rank,
        world_size=world_size,
    )

    print(f"Initialized process group. world_size={world_size}", flush=True)

    main(device)

    dist.destroy_process_group()
# Standard library
from argparse import ArgumentParser

# Third party
import climate_learn as cl
from climate_learn.data.processing.era5_constants import (
    PRESSURE_LEVEL_VARS,
    DEFAULT_PRESSURE_LEVELS,
)
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
    Timer,
    TQDMProgressBar
)
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
import torch
import os, glob, datetime
from pytorch_lightning.strategies import FSDPStrategy
from timm.models.vision_transformer import Block
from climate_learn.models.hub.components.cnn_blocks import (
    DownBlock,
    MiddleBlock,
    UpBlock,
    ResidualBlock
)

from pytorch_lightning.callbacks import DeviceStatsMonitor
import functools
from functools import partial
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import os, glob, datetime
import torch
import torch.nn as nn
import torch.nn.functional as F

from climate_learn.data.transforms import collate_resize
from torch.utils.data import DataLoader
from torch.nn import SyncBatchNorm


parser = ArgumentParser()

parser.add_argument("--summary_depth", type=int, default=1)
parser.add_argument("--max_epochs", type=int, default=50)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--gpu", type=int, default=-1)
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--output_dir", default=None)
parser.add_argument("--img_size", type=int, nargs=2, default=[32, 64], help="LR token grid (H W)")
parser.add_argument("--superres_factor", type=int, default=4, help="SR factor (e.g., 4 → 128x256)")
parser.add_argument("--downsample_mode", type=str, default="area",
                    choices=["area", "bilinear", "bicubic", "nearest"],
                    help="CPU resize mode for inputs")
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--prefetch_factor", type=int, default=4)
parser.add_argument("--pin_memory", action="store_true", default=True)
parser.add_argument("--persistent_workers", action="store_true", default=True)
parser.add_argument("--history", type=int, default=1, help="Number of historical time steps")
parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
parser.add_argument("--patch_size", type=int, default=4, help="Patch size")

subparsers = parser.add_subparsers(
    help="Whether to perform direct, iterative, or continuous forecasting.",
    dest="forecast_type",
)
direct = subparsers.add_parser("direct")
iterative = subparsers.add_parser("iterative")
continuous = subparsers.add_parser("continuous")

direct.add_argument("era5_dir")
direct.add_argument("model", choices=["resnet", "unet", "vit", "res_slimvit"])
direct.add_argument("pred_range", type=int, choices=[6, 24, 72, 120, 240])

iterative.add_argument("era5_dir")
iterative.add_argument("model", choices=["resnet", "unet", "vit"])
iterative.add_argument("pred_range", type=int, choices=[6, 24, 72, 120, 240])

continuous.add_argument("era5_dir")
continuous.add_argument("model", choices=["resnet", "unet", "vit"])

args = parser.parse_args()


if args.checkpoint is None: ## train
    os.environ['MASTER_ADDR'] = str(os.environ['HOSTNAME'])
    os.environ['MASTER_PORT'] = "29500"
    os.environ['WORLD_SIZE'] = os.environ['SLURM_NTASKS']
    os.environ['RANK'] = os.environ['SLURM_PROCID']

    world_size = int(os.environ['SLURM_NTASKS'])
    world_rank = int(os.environ['SLURM_PROCID'])
    local_rank = int(os.environ['SLURM_LOCALID'])
    num_nodes = world_size//8

    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()

    if world_rank==0:
        print("world_size",world_size,"num_nodes",num_nodes,flush=True)
        
else: ## eval
    torch.cuda.is_available():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    

class ShapeDebugCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.printed = False
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if not self.printed and trainer.global_rank == 0:
            x, y = batch[0], batch[1]
            print("="*70, flush=True)
            print("ACTUAL TENSOR SHAPES IN TRAINING", flush=True)
            print("="*70, flush=True)
            print(f"Input (x) shape: {x.shape}", flush=True)
            print(f"Target (y) shape: {y.shape}", flush=True)
            print(f"Expected input: [batch, 26, 32, 64]", flush=True)
            print(f"Expected target: [batch, 5, 128, 256]", flush=True)
            print("="*70, flush=True)
            self.printed = True

class MemoryLogger(pl.Callback):
    """Log detailed GPU memory usage"""
    def __init__(self, log_every_n_steps=50):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
    
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.log_every_n_steps == 0 and trainer.global_rank == 0:
            if torch.cuda.is_available():
                mem_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                mem_reserved = torch.cuda.memory_reserved() / 1024**3    # GB
                mem_max = torch.cuda.max_memory_allocated() / 1024**3    # GB
                
                print(f"[Step {trainer.global_step}] GPU Memory - "
                      f"Allocated: {mem_allocated:.2f}GB, "
                      f"Reserved: {mem_reserved:.2f}GB, "
                      f"Peak: {mem_max:.2f}GB", flush=True)
                
                # Log to tensorboard
                pl_module.log("memory/allocated_gb", mem_allocated, on_step=True, rank_zero_only=True)
                pl_module.log("memory/reserved_gb", mem_reserved, on_step=True, rank_zero_only=True)
                pl_module.log("memory/peak_gb", mem_max, on_step=True, rank_zero_only=True)

class GracefulStop(pl.Callback):
    def __init__(self, flag_path, ckpt_dir):
        super().__init__()
        self.flag_path = flag_path
        self.ckpt_dir = ckpt_dir

    def _should_stop(self):
        return os.path.exists(self.flag_path)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._should_stop():
            # save a final checkpoint quickly
            path = os.path.join(self.ckpt_dir, "last.ckpt")
            trainer.save_checkpoint(path)
            print(f">> Caught stop flag, saved {path}. Stopping now...", flush=True)
            trainer.should_stop = True

def latest_ckpt(ckpt_dir):
    if not os.path.isdir(ckpt_dir):
        return None
    cands = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    return max(cands, key=os.path.getmtime) if cands else None

# Set up data
variables = [
    # "geopotential",
    "temperature",
    # "u_component_of_wind",
    # "v_component_of_wind",
    # "relative_humidity",
    # "specific_humidity",
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    # "toa_incident_solar_radiation",
    # "land_sea_mask",
    # "orography",
    "lattitude",
    # "total_precipitation",
]
in_vars = []
for var in variables:
    if var in PRESSURE_LEVEL_VARS:
        for level in DEFAULT_PRESSURE_LEVELS:
            in_vars.append(var + "_" + str(level))
    else:
        in_vars.append(var)
if args.forecast_type in ("direct", "continuous"):
    out_variables = ["2m_temperature", 
                    #  "geopotential_500", 
                    #  "temperature_850", 
                    #  "total_precipitation", 
                    #  "10m_u_component_of_wind", 
                    #  "10m_v_component_of_wind"
                    ] 
    print("Input variables:", in_vars, flush=True)
    print("Output variables:", out_variables, flush=True)
elif args.forecast_type == "iterative":
    out_variables = variables
out_vars = []
for var in out_variables:
    if var in PRESSURE_LEVEL_VARS:
        for level in DEFAULT_PRESSURE_LEVELS:
            out_vars.append(var + "_" + str(level))
    else:
        out_vars.append(var)
if args.forecast_type in ("direct", "iterative"):
    dm = cl.data.IterDataModule(
        f"{args.forecast_type}-forecasting",
        args.era5_dir,
        args.era5_dir,
        in_vars,
        out_vars,
        src="era5",
        history=args.history,
        window=6,
        pred_range=args.pred_range,
        subsample=6,
        batch_size=args.batch_size, ### was 128
        num_workers=8,  # Increased from 2 to match GPU count
    ) 
elif args.forecast_type == "continuous":
    dm = cl.data.IterDataModule(
        "continuous-forecasting",
        args.era5_dir,
        args.era5_dir,
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
        batch_size=128,
        buffer_size=2000,
        num_workers=8,
    )
# dm.setup()

# Set up deep learning model
# in_channels = 39 ### 49
lr_h, lr_w = args.img_size
# Set target HR size to 128x256 (4x upsampling from 32x64 LR)
hr_h_target, hr_w_target = lr_h * args.superres_factor, lr_w * args.superres_factor
dm.collate_fn = partial(collate_resize, lr_size=(lr_h, lr_w), hr_size=(hr_h_target, hr_w_target), mode=args.downsample_mode)


in_channels = 11 ## was 26
if args.forecast_type == "continuous":
    in_channels += 1  # time dimension
if args.forecast_type == "iterative":  # iterative predicts every var
    out_channels = in_channels
else:
    out_channels = len(out_variables)
if args.model == "resnet":
    model_kwargs = {  # override some of the defaults
        "in_channels": in_channels,
        "out_channels": out_channels,
        "history": 3,
        "n_blocks": 28,
    }
elif args.model == "unet":
    model_kwargs = {  # override some of the defaults
        "in_channels": in_channels,
        "out_channels": out_channels,
        "history": 3,
        "ch_mults": (1, 2, 2),
        "is_attn": (False, False, False),
    }
elif args.model == "vit":
    model_kwargs = {  # override some of the defaults
        "img_size": (128, 256),
        "in_channels": in_channels,
        "out_channels": out_channels,
        "history": 3,
        "patch_size": 8, # was 2
        "embed_dim": 128,
        "depth": 8,
        "decoder_depth": 2,
        "learn_pos_emb": True,
        "num_heads": 4,
    }
elif args.model == "res_slimvit":
        model_kwargs = {  # override some of the defaults
        "img_size": (lr_h, lr_w),
        "in_channels": in_channels,
        "out_channels": out_channels,
        "history": args.history,
        "superres_factor": args.superres_factor, 
        "cnn_ratio": 4,
        "patch_size": args.patch_size,
        "embed_dim": 512,
        "depth": 16,
        "decoder_depth": 4,
        "learn_pos_emb": True,
        "num_heads": 8,
    }


# With hr_size parameter in collate_fn, targets are downsampled to match model output
# So we keep superres_factor=4 for 32x64 -> 128x256 upsampling
if world_rank == 0:
    print(f"[INFO] Using superres_factor={args.superres_factor} for LR {lr_h}x{lr_w} -> HR {hr_h_target}x{hr_w_target}", flush=True)

base_lr = 5e-4
num_gpus = 8
scaling_factor = (num_gpus ** 0.8) # was 0.75
scaled_lr = base_lr * scaling_factor

optim_kwargs = {"lr": scaled_lr, "weight_decay": 1e-5, "betas": (0.9, 0.99)}
sched_kwargs = {
    "warmup_epochs": 5,
    "max_epochs": 50,
    "warmup_start_lr": 1e-8,
    "eta_min": 1e-8,
}

model = cl.load_forecasting_module(
    data_module=dm,
    model=args.model,
    model_kwargs=model_kwargs,
    optim="adamw",
    optim_kwargs=optim_kwargs,
    sched="linear-warmup-cosine-annealing",
    sched_kwargs=sched_kwargs,
    device=device
)

# Wrap the backbone: resize HR inputs -> LR grid before forward
model.net = SyncBatchNorm.convert_sync_batchnorm(model.net)

# Setup trainer
pl.seed_everything(0)
default_root_dir = args.output_dir
logger = TensorBoardLogger(save_dir=f"{default_root_dir}/logs")
# early_stopping = "val/lat_mse:aggregate"
early_stopping = "train/lat_mse:aggregate"
gpu_stats = DeviceStatsMonitor()

ckpt_dir = os.path.join(default_root_dir, "checkpoints")
os.makedirs(ckpt_dir, exist_ok=True)

# Add this debug block around line 355
print("="*50, flush=True)
print(f"[DEBUG] Model Selected: {args.model}", flush=True)
print(f"[DEBUG] Model Kwargs: {model_kwargs}", flush=True)
print("="*50, flush=True)

    
if world_rank == 0:
    print(
        f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M",
        flush=True,
    )

model = cl.load_forecasting_module(
    data_module=dm,
    model=args.model,
    model_kwargs=model_kwargs,
    optim="adamw",
    optim_kwargs=optim_kwargs,
    sched="linear-warmup-cosine-annealing",
    sched_kwargs=sched_kwargs,
    device=device
)

callbacks = [
    ShapeDebugCallback(),  # Add debug callback to print shapes
    MemoryLogger(log_every_n_steps=50),  # Log memory every 50 steps
    TQDMProgressBar(refresh_rate=20),
    RichModelSummary(max_depth=args.summary_depth),
    EarlyStopping(monitor=early_stopping, 
                  patience=args.patience,
                  check_on_train_epoch_end=True,
                  mode="min"
                  ),
    gpu_stats,
    ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="epoch_{epoch:03d}-step_{step}",
        monitor=early_stopping,   # or your metric
        mode="min",
        save_top_k=1,
        save_last=True,
        every_n_train_steps=500,           # save mid-epoch too
        auto_insert_metric_name=False,
    ),
    # Belt-and-suspenders: stop a bit before time limit even if signal is missed
    Timer(duration=datetime.timedelta(minutes=115), interval="epoch", verbose=True),
]

auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            Block  # < ---- Your Transformer layer class
        },
    )

strategy = FSDPStrategy(auto_wrap_policy=auto_wrap_policy,
                        activation_checkpointing=Block)
accelerator = "cuda" if torch.cuda.is_available() else "cpu"

# stop_flag = os.environ.get("PL_STOP_FLAG",
#                            os.path.join(default_root_dir, "STOP"))
# # Clean up stale flag if any (from a previous slice)
# if os.path.exists(stop_flag):
#     os.remove(stop_flag)
# callbacks.insert(0, GracefulStop(stop_flag, ckpt_dir))

# num_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", 1))

trainer = pl.Trainer(
    logger=logger,
    callbacks=callbacks,
    default_root_dir=default_root_dir,
    accelerator="gpu",
    devices=8, # was 8
    max_epochs=args.max_epochs,
    strategy=strategy, ### was "ddp"
    precision="bf16-mixed",
    limit_val_batches=0,
    num_sanity_val_steps=0,
    check_val_every_n_epoch=1,
)

# trainer = pl.Trainer(
#     logger=logger,
#     callbacks=callbacks,
#     default_root_dir=default_root_dir,
#     accelerator="gpu" if args.gpu != -1 else None,
#     devices=[args.gpu] if args.gpu != -1 else None,
#     max_epochs=args.max_epochs,
#     strategy="auto",           # single-GPU? prefer auto over ddp
#     num_sanity_val_steps=0,    # avoid long silent sanity checks on resume
#     log_every_n_steps=10,
#     precision="16",
#     enable_checkpointing=True,
# )

# Define testing regime for iterative forecasting
def iterative_testing(model, trainer, args, from_checkpoint=False):
    for lead_time in [6, 24, 72, 120, 240]:
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
        if from_checkpoint:
            trainer.test(model, datamodule=test_dm)
        else:
            trainer.test(model, datamodule=test_dm, ckpt_path="best")


# Define testing regime for continuous forecasting
def continuous_testing(model, trainer, args, from_checkpoint=False):
    for lead_time in [6, 24, 72, 120, 240]:
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
        if from_checkpoint:
            trainer.test(model, datamodule=test_dm)
        else:
            trainer.test(model, datamodule=test_dm, ckpt_path="best")


# Train and evaluate model from scratch
if args.checkpoint is None:
    # trainer.fit(model, datamodule=dm)
    resume_path = latest_ckpt(ckpt_dir)
    print(f">> Resuming from: {resume_path}" if resume_path else ">> Fresh run", flush=True)
    trainer.fit(model, datamodule=dm, ckpt_path=resume_path)
    if args.forecast_type == "direct":
        trainer.test(model, datamodule=dm, 
                    #  ckpt_path="best"
                     )
    elif args.forecast_type == "iterative":
        iterative_testing(model, trainer, args)
    elif args.forecast_type == "continuous":
        continuous_testing(model, trainer, args)
# Evaluate saved model checkpoint
else:
    model = cl.LitModule.load_from_checkpoint(
        args.checkpoint,
        net=model.net,
        optimizer=model.optimizer,
        lr_scheduler=None,
        train_loss=None,
        val_loss=None,
        test_loss=model.test_loss,
        test_target_tranfsorms=model.test_target_transforms,
    )
    if args.forecast_type == "direct":
        trainer.test(model, datamodule=dm)
    elif args.forecast_type == "iterative":
        iterative_testing(model, trainer, args, from_checkpoint=True)
    elif args.forecast_type == "continuous":
        continuous_testing(model, trainer, args, from_checkpoint=True)

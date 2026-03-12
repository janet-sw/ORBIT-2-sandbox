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
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.nn import SyncBatchNorm


parser = ArgumentParser()

parser.add_argument("--summary_depth", type=int, default=1)
parser.add_argument("--max_epochs", type=int, default=30)
parser.add_argument("--patience", type=int, default=20)
parser.add_argument("--gpu", type=int, default=-1)
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--output_dir", default=None)
# ── img_size = native grid for 1.40625°: 128 x 256 ──
parser.add_argument("--img_size", type=int, nargs=2, default=[128, 256],
                    help="Native image grid (H W) — no superres for ViT baseline")
parser.add_argument("--superres_factor", type=int, default=1,
                    help="SR factor (1 = no superres, matching res_slimvit)")
parser.add_argument("--num_workers", type=int, default=2)
parser.add_argument("--prefetch_factor", type=int, default=4)
parser.add_argument("--pin_memory", action="store_true", default=True)
parser.add_argument("--persistent_workers", action="store_true", default=True)
# ── history=1 to match res_slimvit ──
parser.add_argument("--history", type=int, default=1,
                    help="Number of historical time steps (1 to match res_slimvit)")
# ── batch_size=16 to match res_slimvit ──
parser.add_argument("--batch_size", type=int, default=16,
                    help="Batch size (16 to match res_slimvit)")
parser.add_argument("--patch_size", type=int, default=2,
                    help="Patch size (2 to match res_slimvit)")

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


if args.checkpoint is None:  # train
    os.environ['MASTER_ADDR'] = str(os.environ['HOSTNAME'])
    os.environ['MASTER_PORT'] = "29500"
    os.environ['WORLD_SIZE'] = os.environ['SLURM_NTASKS']
    os.environ['RANK'] = os.environ['SLURM_PROCID']

    world_size = int(os.environ['SLURM_NTASKS'])
    world_rank = int(os.environ['SLURM_PROCID'])
    local_rank = int(os.environ['SLURM_LOCALID'])
    num_nodes = world_size // 8

    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()

    if world_rank == 0:
        print("world_size", world_size, "num_nodes", num_nodes, flush=True)

else:  # eval — fixed syntax error (was missing `if`)
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    world_rank = 0


class ShapeDebugCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.printed = False

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if not self.printed and trainer.global_rank == 0:
            x, y = batch[0], batch[1]
            print("=" * 70, flush=True)
            print("ACTUAL TENSOR SHAPES IN TRAINING", flush=True)
            print("=" * 70, flush=True)
            print(f"Input (x) shape: {x.shape}", flush=True)
            print(f"Target (y) shape: {y.shape}", flush=True)
            print(f"Expected input:  [batch, {args.history * 40}, 128, 256]", flush=True)
            print(f"Expected target: [batch, 4, 128, 256]", flush=True)
            print("=" * 70, flush=True)
            self.printed = True


class MemoryLogger(pl.Callback):
    """Log detailed GPU memory usage"""
    def __init__(self, log_every_n_steps=50):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.log_every_n_steps == 0 and trainer.global_rank == 0:
            if torch.cuda.is_available():
                mem_allocated = torch.cuda.memory_allocated() / 1024**3
                mem_reserved = torch.cuda.memory_reserved() / 1024**3
                mem_max = torch.cuda.max_memory_allocated() / 1024**3

                print(f"[Step {trainer.global_step}] GPU Memory - "
                      f"Allocated: {mem_allocated:.2f}GB, "
                      f"Reserved: {mem_reserved:.2f}GB, "
                      f"Peak: {mem_max:.2f}GB", flush=True)

                pl_module.log("memory/allocated_gb", mem_allocated, on_step=True, rank_zero_only=True)
                pl_module.log("memory/reserved_gb", mem_reserved, on_step=True, rank_zero_only=True)
                pl_module.log("memory/peak_gb", mem_max, on_step=True, rank_zero_only=True)


def latest_ckpt(ckpt_dir):
    if not os.path.isdir(ckpt_dir):
        return None
    cands = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    return max(cands, key=os.path.getmtime) if cands else None


# ══════════════════════════════════════════════════════════════════════════════
# Variables — EXACTLY matching the res_slimvit 1.40625° runs
# ══════════════════════════════════════════════════════════════════════════════
# 10 base variables (5 pressure-level + 5 surface/static) → 40 input channels
# Pressure levels: [50, 250, 500, 600, 700, 850, 925]
variables = [
    "temperature",
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "land_sea_mask",
    "orography",
    "specific_humidity",
]

in_vars = []
for var in variables:
    if var in PRESSURE_LEVEL_VARS:
        for level in DEFAULT_PRESSURE_LEVELS:
            in_vars.append(var + "_" + str(level))
    else:
        in_vars.append(var)

# 4 output variables — same as res_slimvit
out_variables = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "geopotential_500",
    "temperature_850",
]

out_vars = []
for var in out_variables:
    if var in PRESSURE_LEVEL_VARS:
        for level in DEFAULT_PRESSURE_LEVELS:
            out_vars.append(var + "_" + str(level))
    else:
        out_vars.append(var)

if args.checkpoint is None and world_rank == 0:
    print(f"Input variables ({len(in_vars)}): {in_vars}", flush=True)
    print(f"Output variables ({len(out_vars)}): {out_vars}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Data module — matching res_slimvit config
# ══════════════════════════════════════════════════════════════════════════════
if args.forecast_type in ("direct", "iterative"):
    dm = cl.data.IterDataModule(
        f"{args.forecast_type}-forecasting",
        args.era5_dir,
        args.era5_dir,
        in_vars,
        out_vars,
        src="era5",
        history=args.history,       # 1
        window=6,
        pred_range=args.pred_range,
        subsample=6,
        batch_size=args.batch_size,  # 16
        num_workers=args.num_workers,
    )
elif args.forecast_type == "continuous":
    dm = cl.data.IterDataModule(
        "continuous-forecasting",
        args.era5_dir,
        args.era5_dir,
        in_vars,
        out_vars,
        src="era5",
        history=args.history,
        window=6,
        pred_range=1,
        max_pred_range=120,
        random_lead_time=True,
        hrs_each_step=1,
        subsample=6,
        batch_size=args.batch_size,
        buffer_size=2000,
        num_workers=args.num_workers,
    )


# ══════════════════════════════════════════════════════════════════════════════
# No collate_resize needed — ViT operates on native 128x256 grid directly
# (res_slimvit also uses superres_factor=1 for 1.40625°)
# ══════════════════════════════════════════════════════════════════════════════

# ── Channel counts ──
in_channels = len(in_vars)  # 40
if args.forecast_type == "continuous":
    in_channels += 1
if args.forecast_type == "iterative":
    out_channels = in_channels
else:
    out_channels = len(out_vars)  # 4

lr_h, lr_w = args.img_size  # (128, 256)

if args.model == "resnet":
    model_kwargs = {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "history": args.history,
        "n_blocks": 28,
    }
elif args.model == "unet":
    model_kwargs = {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "history": args.history,
        "ch_mults": (1, 2, 2),
        "is_attn": (False, False, False),
    }
elif args.model == "vit":
    # ────────────────────────────────────────────────────────────────────
    # ViT baseline config — matched to res_slimvit for fair comparison
    #
    # res_slimvit @ 1.40625°: 12.18M params
    #   img_size=(128,256), patch_size=2, embed_dim=256, depth=12,
    #   num_heads=4, history=1, in_channels=40, out_channels=4
    #
    # ViT baseline: aim for comparable param count (~12M)
    #   img_size=(128,256), patch_size=2, embed_dim=256, depth=12,
    #   num_heads=4, history=1
    #   Tokens: (128/2)*(256/2) = 64*128 = 8192 tokens (same as res_slimvit)
    # ────────────────────────────────────────────────────────────────────
    model_kwargs = {
        "img_size": (lr_h, lr_w),        # (128, 256)
        "in_channels": in_channels,       # 40
        "out_channels": out_channels,     # 4
        "history": args.history,          # 1
        "patch_size": args.patch_size,    # 2
        "embed_dim": 256,
        "depth": 12,
        "decoder_depth": 4,
        "learn_pos_emb": True,
        "num_heads": 4,
    }
elif args.model == "res_slimvit":
    model_kwargs = {
        "img_size": (lr_h, lr_w),
        "in_channels": in_channels,
        "out_channels": out_channels,
        "history": args.history,
        "superres_factor": args.superres_factor,
        "cnn_ratio": 4,
        "patch_size": args.patch_size,
        "embed_dim": 256,
        "depth": 12,
        "decoder_depth": 4,
        "learn_pos_emb": True,
        "num_heads": 4,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Optimizer & scheduler — matching res_slimvit config
# ══════════════════════════════════════════════════════════════════════════════
num_gpus = int(os.environ.get('SLURM_NTASKS', 8))
base_lr = 5e-4
scaling_factor = (num_gpus ** 0.8)
scaled_lr = base_lr * scaling_factor

optim_kwargs = {
    "lr": scaled_lr,
    "weight_decay": 1e-5,
    "betas": (0.9, 0.99),
}
sched_kwargs = {
    "warmup_epochs": 5,
    "max_epochs": args.max_epochs,
    "warmup_start_lr": 1e-6,
    "eta_min": 1e-8,
}

if args.checkpoint is None and world_rank == 0:
    print("=" * 50, flush=True)
    print(f"[DEBUG] Model: {args.model}", flush=True)
    print(f"[DEBUG] Model kwargs: {model_kwargs}", flush=True)
    print(f"[DEBUG] Optim: lr={scaled_lr:.6f} (base={base_lr}, gpus={num_gpus})", flush=True)
    print(f"[DEBUG] Scheduler: {sched_kwargs}", flush=True)
    print("=" * 50, flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Create model (ONCE — fixed duplicate creation from original)
# ══════════════════════════════════════════════════════════════════════════════
model = cl.load_forecasting_module(
    data_module=dm,
    model=args.model,
    model_kwargs=model_kwargs,
    optim="adamw",
    optim_kwargs=optim_kwargs,
    sched="linear-warmup-cosine-annealing",
    sched_kwargs=sched_kwargs,
    device=device,
)

model.net = SyncBatchNorm.convert_sync_batchnorm(model.net)

if args.checkpoint is None and world_rank == 0:
    print(
        f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M",
        flush=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Trainer setup
# ══════════════════════════════════════════════════════════════════════════════
pl.seed_everything(0)
default_root_dir = args.output_dir
logger = TensorBoardLogger(save_dir=f"{default_root_dir}/logs")
early_stopping = "train/lat_mse:aggregate"

ckpt_dir = os.path.join(default_root_dir, "checkpoints")
os.makedirs(ckpt_dir, exist_ok=True)

callbacks = [
    ShapeDebugCallback(),
    MemoryLogger(log_every_n_steps=50),
    TQDMProgressBar(refresh_rate=20),
    RichModelSummary(max_depth=args.summary_depth),
    EarlyStopping(
        monitor=early_stopping,
        patience=args.patience,
        check_on_train_epoch_end=True,
        mode="min",
    ),
    ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="epoch_{epoch:03d}-step_{step}",
        monitor=early_stopping,
        mode="min",
        save_top_k=1,
        save_last=True,
        every_n_train_steps=500,
        auto_insert_metric_name=False,
    ),
    Timer(duration=datetime.timedelta(minutes=115), interval="epoch", verbose=True),
]

auto_wrap_policy = functools.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={Block},
)

strategy = FSDPStrategy(
    auto_wrap_policy=auto_wrap_policy,
    activation_checkpointing=Block,
)

trainer = pl.Trainer(
    logger=logger,
    callbacks=callbacks,
    default_root_dir=default_root_dir,
    accelerator="gpu",
    devices=8,
    num_nodes=num_nodes if args.checkpoint is None else 1,
    max_epochs=args.max_epochs,
    strategy=strategy,
    precision="bf16-mixed",
    limit_val_batches=0,
    num_sanity_val_steps=0,
    check_val_every_n_epoch=1,
)


# ══════════════════════════════════════════════════════════════════════════════
# Testing helpers
# ══════════════════════════════════════════════════════════════════════════════
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
            history=args.history,
            window=6,
            pred_range=lead_time,
            subsample=1,
        )
        if from_checkpoint:
            trainer.test(model, datamodule=test_dm)
        else:
            trainer.test(model, datamodule=test_dm, ckpt_path="best")


def continuous_testing(model, trainer, args, from_checkpoint=False):
    for lead_time in [6, 24, 72, 120, 240]:
        test_dm = cl.data.IterDataModule(
            "continuous-forecasting",
            args.era5_dir,
            args.era5_dir,
            in_vars,
            out_vars,
            src="era5",
            history=args.history,
            window=6,
            pred_range=lead_time,
            max_pred_range=lead_time,
            random_lead_time=False,
            hrs_each_step=1,
            subsample=1,
            batch_size=args.batch_size,
            buffer_size=2000,
            num_workers=args.num_workers,
        )
        if from_checkpoint:
            trainer.test(model, datamodule=test_dm)
        else:
            trainer.test(model, datamodule=test_dm, ckpt_path="best")


# ══════════════════════════════════════════════════════════════════════════════
# Train or evaluate
# ══════════════════════════════════════════════════════════════════════════════
if args.checkpoint is None:
    resume_path = latest_ckpt(ckpt_dir)
    if world_rank == 0:
        print(f">> Resuming from: {resume_path}" if resume_path else ">> Fresh run", flush=True)
    trainer.fit(model, datamodule=dm, ckpt_path=resume_path)
    if args.forecast_type == "direct":
        trainer.test(model, datamodule=dm)
    elif args.forecast_type == "iterative":
        iterative_testing(model, trainer, args)
    elif args.forecast_type == "continuous":
        continuous_testing(model, trainer, args)
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
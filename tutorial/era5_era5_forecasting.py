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
)
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
import os
import torch
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
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import os, glob, datetime


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


parser = ArgumentParser()
parser.add_argument("--summary_depth", type=int, default=1)
parser.add_argument("--max_epochs", type=int, default=50)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--gpu", type=int, default=-1)
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--output_dir", default=None)

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

def latest_ckpt(ckpt_dir):
    if not os.path.isdir(ckpt_dir):
        return None
    cands = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    return max(cands, key=os.path.getmtime) if cands else None

# Set up data
variables = [
    "geopotential",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    # "relative_humidity",
    "specific_humidity",
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    # "toa_incident_solar_radiation",
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
    out_variables = ["2m_temperature", 
                     "geopotential_500", 
                     "temperature_850", 
                     "total_precipitation", 
                     "10m_u_component_of_wind", 
                     "10m_v_component_of_wind"]
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
        history=3,
        window=6,
        pred_range=args.pred_range,
        subsample=6,
        batch_size=128,
        num_workers=8,
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
dm.setup()

# Set up deep learning model
in_channels = 42 ### was 49
if args.forecast_type == "continuous":
    in_channels += 1  # time dimension
if args.forecast_type == "iterative":  # iterative predicts every var
    out_channels = in_channels
else:
    out_channels = len(out_variables) ### was 3
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
        "img_size": (32, 64),
        "in_channels": in_channels,
        "out_channels": out_channels,
        "history": 3,
        "patch_size": 2,
        "embed_dim": 128,
        "depth": 8,
        "decoder_depth": 2,
        "learn_pos_emb": True,
        "num_heads": 4,
    }
elif args.model == "res_slimvit":
        model_kwargs = {  # override some of the defaults
        "img_size": (32, 64),
        "in_channels": in_channels,
        "out_channels": out_channels,
        "history": 3,
        "superres_factor": 1, # no upscaling 
        "cnn_ratio": 4,
        "patch_size": 2,
        "embed_dim": 128,
        "depth": 8,
        "decoder_depth": 2,
        "learn_pos_emb": True,
        "num_heads": 4,
    }
optim_kwargs = {"lr": 5e-4, "weight_decay": 1e-5, "betas": (0.9, 0.99)}
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
    device=device,
)

# Setup trainer
pl.seed_everything(0)
default_root_dir = f"{args.output_dir}/{args.model}_{args.forecast_type}_forecasting_{args.pred_range}"
logger = TensorBoardLogger(save_dir=f"{default_root_dir}/logs")
# early_stopping = "val/lat_mse:aggregate"
early_stopping = "train/lat_mse:aggregate"

gpu_stats = DeviceStatsMonitor()

callbacks = [
    RichProgressBar(),
    RichModelSummary(max_depth=args.summary_depth),
    EarlyStopping(monitor=early_stopping, 
                  patience=args.patience,
                #   check_on_train_epoch_end=False
                  ),
    gpu_stats,
    ModelCheckpoint(
        dirpath=f"{default_root_dir}/checkpoints",
        monitor=early_stopping,
        filename="epoch_{epoch:03d}",
        auto_insert_metric_name=False,
    ),
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

trainer = pl.Trainer(
    logger=logger,
    callbacks=callbacks,
    default_root_dir=default_root_dir,
    accelerator="gpu",
    devices=8,
    max_epochs=args.max_epochs,
    strategy=strategy, ### was "ddp"
    precision="bf16-mixed",
)

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
    resume_path = latest_ckpt(f"{default_root_dir}/checkpoints")
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
    # ckpt_model = cl.LitModule.load_from_checkpoint(args.checkpoint, strict=True)
    # trainer.test(ckpt_model, datamodule=dm)
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
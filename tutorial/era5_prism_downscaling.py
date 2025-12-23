# Standard library
from argparse import ArgumentParser

# Third party
import climate_learn as cl
from climate_learn.models.hub import VisionTransformer, Interpolation
from climate_learn.transforms import Mask, Denormalize
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
import torch.nn as nn
import torch
import os
import functools
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from pytorch_lightning.strategies import FSDPStrategy
from timm.models.vision_transformer import Block
from climate_learn.models.hub.components.cnn_blocks import (
    DownBlock,
    MiddleBlock,
    UpBlock,
    ResidualBlock
)


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
parser.add_argument("era5_cropped_dir")
parser.add_argument("prism_processed_dir")
parser.add_argument("preset", choices=["resnet", "unet", "vit","res_slimvit"])
parser.add_argument("--summary_depth", type=int, default=1)
parser.add_argument("--max_epochs", type=int, default=50)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--gpu", type=int, default=-1)
parser.add_argument("--checkpoint", default=None)
args = parser.parse_args()

# Set up data
dm = cl.data.ERA5toPRISMDataModule(
    args.era5_cropped_dir,
    args.prism_processed_dir,
    batch_size=32,
    num_workers=4,
)
dm.setup()

# Set up masking
mask = Mask(dm.get_out_mask().to(device=f"cuda:{args.gpu}"))
denorm = Denormalize(dm)
denorm_mask = lambda x: denorm(mask(x))

# Default ViT preset is optimized for ERA5 to ERA5 downscaling, so we
# modify the architecture for ERA5 to PRISM
if args.preset == "vit":
    net = nn.Sequential(
        Interpolation((32, 64), "bilinear"),
        VisionTransformer(
            img_size=(32, 64),
            in_channels=1,
            out_channels=1,
            history=1,
            patch_size=2,
            learn_pos_emb=True,
            embed_dim=128,
            depth=8,
            decoder_depth=2,
            num_heads=4,
        ),
    )
    optim_kwargs = {"lr": 1e-5, "weight_decay": 1e-5, "betas": (0.9, 0.99)}
    sched_kwargs = {
        "warmup_epochs": 5,
        "max_epochs": 50,
        "warmup_start_lr": 1e-8,
        "eta_min": 1e-8,
    }
    model = cl.load_downscaling_module(
        device=device,
        data_module=dm,
        model=net,
        optim="adamw",
        optim_kwargs=optim_kwargs,
        sched="linear-warmup-cosine-annealing",
        sched_kwargs=sched_kwargs,
        train_target_transform=mask,
        val_target_transform=[denorm_mask, denorm_mask, denorm_mask, mask],
        test_target_transform=[denorm_mask, denorm_mask, denorm_mask],
    )
# Default presets for ResNet and U-net are ready to use out of the box
else:
    model = cl.load_downscaling_module(
        device=device,
        data_module=dm,
        architecture=args.preset,
        train_target_transform=mask,
        val_target_transform=[denorm_mask, denorm_mask, denorm_mask, mask],
        test_target_transform=[denorm_mask, denorm_mask, denorm_mask],
    )

# Setup trainer
pl.seed_everything(0)
default_root_dir = f"{args.preset}_downscaling_prism"
logger = TensorBoardLogger(save_dir=f"{default_root_dir}/logs")
early_stopping = "val/mse:aggregate"
callbacks = [
    RichModelSummary(max_depth=args.summary_depth),
    EarlyStopping(monitor=early_stopping, patience=args.patience),
    ModelCheckpoint(
        dirpath=f"{default_root_dir}/checkpoints",
        monitor=early_stopping,
        filename="epoch_{epoch:03d}",
        auto_insert_metric_name=False,
        save_top_k=-1,
        every_n_epochs=1
    ),
]


if args.preset =="vit" or args.preset=="res_slimvit":
   
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            Block  # < ---- Your Transformer layer class
        },
    )

    strategy = FSDPStrategy(auto_wrap_policy=auto_wrap_policy,activation_checkpointing=Block)

elif args.preset =="unet":
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            UpBlock,DownBlock,MiddleBlock  # < ---- Your Transformer layer class
        },
    )

    strategy = FSDPStrategy(auto_wrap_policy=auto_wrap_policy,activation_checkpointing=[UpBlock,DownBlock,MiddleBlock])
else: #resnet
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            ResidualBlock  # < ---- Your Transformer layer class
        },
    )

    strategy = FSDPStrategy(auto_wrap_policy=auto_wrap_policy,activation_checkpointing=ResidualBlock)


trainer = pl.Trainer(
    logger=logger,
    callbacks=callbacks,
    default_root_dir=default_root_dir,
    accelerator="gpu",
    devices = 8,
    num_nodes = num_nodes,
    max_epochs=args.max_epochs,
    strategy=strategy,
    precision="bf16-mixed",
)

# Train and evaluate model from scratch
if args.checkpoint is None:
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)
# Resume training from saved model checkpoint
else:
    trainer.fit(model, datamodule=dm, ckpt_path=args.checkpoint)
    trainer.test(model, datamodule=dm)

# Evaluate only from saved model checkpoint
#else:
#    model = cl.LitModule.load_from_checkpoint(
#        args.checkpoint,
#        net=model.net,
#        optimizer=model.optimizer,
#        lr_scheduler=None,
#        train_loss=None,
#        val_loss=None,
#        test_loss=model.test_loss,
#        test_target_transforms=model.test_target_transforms,
#    )
#    trainer.test(model, datamodule=dm)

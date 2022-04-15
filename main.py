import argparse
from asyncio.log import logger
import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    TQDMProgressBar,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import WandbLogger
import wandb
from datamodules import DATAMODULE_TABLE
from transforms import TRANSFORMS_TABLE
from gans import MODEL_TABLE

# from gans import


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


def hyperparameters():
    parser = argparse.ArgumentParser()
    # project
    parser = pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--project_name", type=str, default="GANs")

    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--logdir", type=str, default="experiment")

    # data
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DATAMODULE_TABLE.keys()),
        default="MNIST",
    )
    parser.add_argument(
        "--transform",
        type=str,
        choices=list(TRANSFORMS_TABLE.keys()),
        default="BASE",
    )
    parser.add_argument("--root_dir", type=str)
    parser.add_argument("--image_size", type=int, default=28)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)

    # model
    parser.add_argument(
        "--GAN",
        type=str,
        choices=list(MODEL_TABLE.keys()),
        default="GAN",
    )
    parser.add_argument("--g_dim", type=int, default=16)
    parser.add_argument("--d_dim", type=int, default=16)
    parser.add_argument("--z_dim", type=int, default=100)
    parser.add_argument("--image_channels", type=int, default=3)

    # training
    parser.add_argument("--sample_count", type=int, default=64)
    parser.add_argument("--g_lr", type=float, default=1e-4)
    parser.add_argument("--d_lr", type=float, default=1e-4)
    parser.add_argument("--b1", type=float, default=0.5)
    parser.add_argument("--b2", type=float, default=0.99)

    # logger
    parser.add_argument("--upload_artifacts", type=str2bool, default=True)
    args = pl.Trainer.parse_argparser(parser.parse_args())

    return args


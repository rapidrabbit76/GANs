import argparse
import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    TQDMProgressBar,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
import wandb
from datamodules import build_datamodule, DATAMODULE_LIST
from transforms import build_transform, TRANSFORMS_LIST
from gans import build_model, MODEL_LIST

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
        choices=DATAMODULE_LIST,
        default="MNIST",
    )
    parser.add_argument(
        "--transform",
        type=str,
        choices=TRANSFORMS_LIST,
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
        choices=MODEL_LIST,
        default="GAN",
    )
    parser.add_argument("--g_dim", type=int, default=16)
    parser.add_argument("--d_dim", type=int, default=16)
    parser.add_argument("--z_dim", type=int, default=100)
    parser.add_argument("--image_channels", type=int, default=3)

    # training
    parser.add_argument("--sample_count", type=int, default=100)
    parser.add_argument("--g_lr", type=float, default=1e-4)
    parser.add_argument("--d_lr", type=float, default=1e-4)
    parser.add_argument("--b1", type=float, default=0.5)
    parser.add_argument("--b2", type=float, default=0.99)

    # WGAN
    parser.add_argument("--n_critic", type=int, default=5)
    parser.add_argument("--critic_clip_value", type=float, default=0.01)
    # WGAN-GP
    parser.add_argument("--LAMBDA_GP", type=float, default=10.0)

    # logger
    parser.add_argument("--upload_artifacts", type=str2bool, default=True)
    args = pl.Trainer.parse_argparser(parser.parse_args())

    return args


def main(args):
    ############# SETUP ################
    pl.seed_everything(args.seed)

    ############# DATAMODULE ################
    transform = build_transform(
        args.transform,
        image_shape=[args.image_channels, args.image_size, args.image_size],
    )
    datamodule = build_datamodule(
        args.dataset,
        root_dir=args.root_dir,
        train_transforms=transform,
        val_transforms=transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    ############# MODEL ################
    model = build_model(args)

    ############# LOGGER ###############
    name = f"{args.GAN}-{args.dataset}"
    tb_logger = TensorBoardLogger("tensorboard", name=name)
    wandb_logger = WandbLogger(project=args.project_name, name=name, id=name)
    wandb_logger.watch(model, log="all", log_freq=args.log_every_n_steps)
    save_dir = wandb_logger.experiment.dir

    ############# CALLBACKS ############
    callbacks = [
        TQDMProgressBar(refresh_rate=5),
        LearningRateMonitor(logging_interval="epoch"),
        ModelCheckpoint(save_dir, save_last=True),
    ]

    ############### TRAINER ##############
    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=[tb_logger, wandb_logger],
        callbacks=callbacks,
    )
    ########### TRAINING START ###########
    trainer.fit(model, datamodule=datamodule)

    ############### Artifacts ################
    model = model.cpu()
    torchscript_path = os.path.join(
        save_dir,
        f"{args.GAN}-{args.dataset}.jit",
    )
    model.to_torchscript(torchscript_path)

    if args.upload_artifacts:
        artifacts = wandb.Artifact(f"{args.GAN}-{args.dataset}", type="model")
        artifacts.add_file(torchscript_path)
        wandb.log_artifact(artifacts)


if __name__ == "__main__":
    args = hyperparameters()
    info = main(args)

from typing import Dict
import pytorch_lightning as pl
from datamodules.MNIST import (
    MnistDataModule,
    FashionMnistDataModule,
    EmnistDataModule,
    KMnistDataModule,
)
from datamodules.CIFAR import CIFAR10DataModule, CIFAR100DataModule

DATAMODULE_TABLE: Dict["str", pl.LightningDataModule] = {
    "MNIST": MnistDataModule,
    "FMNIST": FashionMnistDataModule,
    "EMNIST": EmnistDataModule,
    "KMNIST": KMnistDataModule,
    "CIFAR10": CIFAR10DataModule,
    "CIFAR100": CIFAR100DataModule,
}

DATAMODULE_LIST = list(DATAMODULE_TABLE.keys())


def build_datamodule(dataset: str, **kwargs) -> pl.LightningDataModule:
    DATAMODULE = DATAMODULE_TABLE[dataset]
    datamodule = DATAMODULE(
        root_dir=kwargs.get("root_dir"),
        train_transforms=kwargs.get("train_transforms"),
        val_transforms=kwargs.get("val_transforms"),
        batch_size=kwargs.get("batch_size"),
        num_workers=kwargs.get("num_workers"),
    )
    return datamodule

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

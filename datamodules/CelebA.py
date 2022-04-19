from typing import Callable, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


class CelebADataModule(pl.LightningDataModule):
    def __init__(
        self,
        root_dir: str,
        train_transforms: Callable,
        val_transforms: Callable,
        batch_size: int = 8,
        num_workers: int = 8,
    ):
        super().__init__()
        self.save_hyperparameters(
            {
                "root_dir": root_dir,
                "batch_size": batch_size,
                "num_workers": num_workers,
            },
        )
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            # split dataset to train, val
            self.train_ds = ImageFolder(
                self.hparams.root_dir,
                transform=self.train_transforms,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
        )

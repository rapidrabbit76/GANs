import pytorch_lightning as pl

import torch
from torch import nn

Tensor = torch.Tensor


class Discriminator(pl.LightningModule):
    def __init__(self, inp: int) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(inp, 128),
            nn.LeakyReLU(0.02),
            nn.Linear(128, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.net(x)
        return x

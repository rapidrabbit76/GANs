import os
from torchvision.utils import make_grid, save_image
import torch
import torch.nn as nn


def build_sample(
    samples: torch.Tensor,
    gs: int,
    epoch: int,
    save_dir: str,
):
    sample = make_grid(
        samples,
        int(len(samples) ** 0.5),
        normalize=True,
        value_range=(-1, 1),
    )
    filename = f"GS:{gs}.png"
    filepath = os.path.join(save_dir, filename)
    save_image(sample, filepath)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

import torch.nn as nn
import torch.nn.functional as F
import torch

Tensor = torch.Tensor


class GAN:
    @classmethod
    def calc_real_loss(cls, x: Tensor) -> Tensor:
        x = F.binary_cross_entropy_with_logits(x, torch.ones_like(x))
        return x

    @classmethod
    def calc_fake_loss(cls, x: Tensor) -> Tensor:
        x = F.binary_cross_entropy_with_logits(x, torch.zeros_like(x))
        return x


class WGAN:
    @classmethod
    def calc_real_loss(cls, x: Tensor) -> Tensor:
        x = -torch.mean(x)
        return x

    @classmethod
    def calc_fake_loss(cls, x: Tensor) -> Tensor:
        x = torch.mean(x)
        return x

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

    @classmethod
    def calc_disc_loss(cls, real: Tensor, fake: Tensor) -> Tensor:
        real_loss = cls.calc_real_loss(real)
        fake_loss = cls.calc_fake_loss(fake)
        loss = (real_loss + fake_loss) * 0.5
        return loss

    @classmethod
    def calc_gen_loss(cls, fake: Tensor) -> Tensor:
        loss = cls.calc_real_loss(fake)
        return loss

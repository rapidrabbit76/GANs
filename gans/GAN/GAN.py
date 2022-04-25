from random import sample

import losses
import pytorch_lightning as pl
import torch
import wandb
from torchvision.utils import make_grid

from .model import Discriminator, Generator

Tensor = torch.Tensor


class GAN(pl.LightningModule):
    def __init__(self, args) -> None:
        super().__init__()
        self.save_hyperparameters(args)
        inp = (
            self.hparams.image_channels
            * self.hparams.image_size
            * self.hparams.image_size
        )
        self.generator: Generator = Generator(self.hparams.z_dim, inp)
        self.discriminator = Discriminator(inp)
        self.gan_losses = losses.GAN
        self.val_z = self.get_noise().reshape([self.hparams.sample_count, -1])

    def get_noise(self, N: int = 0) -> Tensor:
        N = self.hparams.sample_count if N == 0 else N
        z = torch.randn(N, self.hparams.z_dim, 1, 1)
        return z

    def forward(self, z: Tensor) -> Tensor:
        x = self.generator(z)
        return x

    def configure_optimizers(self):
        g_lr = self.hparams.g_lr
        d_lr = self.hparams.d_lr
        b1 = self.hparams.b1
        b2 = self.hparams.b2

        opt_g = torch.optim.Adam(
            self.generator.parameters(), lr=g_lr, betas=(b1, b2)
        )
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=d_lr, betas=(b1, b2)
        )
        return [opt_g, opt_d], []

    def training_step(self, batch, batch_idx, optimizer_idx):
        images, _ = batch

        # train generator
        if optimizer_idx == 0:
            loss = self.generator_step(images, batch_idx)
            self.log_dict(loss)

        # train discriminator
        if optimizer_idx == 1:
            loss = self.discriminator_step(images, batch_idx)
            self.log_dict(loss)
        return loss["loss"]

    def generator_step(self, images, batch_idx):
        N, *_ = images.shape
        z = self.get_noise(N).type_as(images).reshape([N, -1])
        self.fake = self(z)
        output = self.discriminator(self.fake)
        loss = self.gan_losses.calc_real_loss(output)
        loss = {"loss": loss, "g_loss": loss.item()}
        return loss

    def discriminator_step(self, images, batch_idx):
        N, *_ = images.shape
        images = torch.reshape(images, [N, -1])
        real_logits = self.discriminator(images)
        fake_logits = self.discriminator(self.fake.detach())

        real_loss = self.gan_losses.calc_real_loss(real_logits)
        fake_loss = self.gan_losses.calc_fake_loss(fake_logits)
        loss = (real_loss + fake_loss) * 0.5
        loss = {
            "loss": loss,
            "d_loss": loss.item(),
            "prob/real_prob": torch.mean(torch.sigmoid(real_logits)).item(),
            "prob/fake_prob": torch.mean(torch.sigmoid(fake_logits)).item(),
        }
        return loss

    def training_epoch_end(self, outputs):
        self.sample_step()

    def sample_step(self):
        torch.set_grad_enabled(False)
        self.eval()
        z = self.val_z.to(self.generator.device)
        samples = self(z)
        samples = torch.reshape(
            samples,
            [
                self.hparams.sample_count,
                self.hparams.image_channels,
                self.hparams.image_size,
                self.hparams.image_size,
            ],
        )
        # tensorboard logger
        sample = make_grid(
            samples,
            int(self.hparams.sample_count**0.5),
            normalize=True,
            value_range=(-1, 1),
        )
        self.logger[0].experiment.add_image(
            "sample", sample, self.current_epoch
        )

        # wandb logger
        sample = wandb.Image(samples)
        self.logger[1].experiment.log({"sample": sample}, step=self.global_step)
        self.train(True)
        torch.set_grad_enabled(True)

    def to_torchscript(self, file_path: str):
        return super().to_torchscript(file_path, "trace", self.val_z)

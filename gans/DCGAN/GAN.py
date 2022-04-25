import os

import losses
import pytorch_lightning as pl
import torch
import wandb
from gans import GAN
from opt import initialize_weights
from torchvision.utils import make_grid, save_image

from .model import Discriminator, Generator

Tensor = torch.Tensor


class DCGAN(GAN):
    def __init__(self, args) -> None:
        super(GAN, self).__init__()
        self.save_hyperparameters(args)
        self.generator: Generator = Generator(
            self.hparams.z_dim, self.hparams.image_channels, self.hparams.g_dim
        )
        self.discriminator = Discriminator(
            self.hparams.image_channels, self.hparams.d_dim
        )
        self.gan_losses = losses.GAN
        self.val_z = self.get_noise()
        initialize_weights(self.generator)
        initialize_weights(self.discriminator)

    def generator_step(self, images, batch_idx):
        N, *_ = images.shape
        z = self.get_noise(N).type_as(images)
        self.fake = self(z)
        output = self.discriminator(self.fake)
        loss = self.gan_losses.calc_real_loss(output)
        loss = {"loss": loss, "g_loss": loss.item()}
        return loss

    def discriminator_step(self, images, batch_idx):
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

    def sample_step(self):
        torch.set_grad_enabled(False)
        self.eval()
        z = self.val_z.to(self.generator.device)
        samples = self(z)
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

        filename = f"GS:{self.global_step}.png"
        filepath = os.path.join(self.logger[0].experiment.log_dir, filename)
        save_image(sample, filepath)

        # wandb logger
        sample = wandb.Image(filepath)
        self.logger[1].experiment.log(
            {"sample": sample},
            step=self.global_step,
        )
        self.train(True)
        torch.set_grad_enabled(True)

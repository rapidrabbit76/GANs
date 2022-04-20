import pytorch_lightning as pl


import torch
from torch import nn
from torchvision.utils import make_grid
from .model import Generator, Discriminator as Critic
import losses
import wandb

Tensor = torch.Tensor


class WGAN(pl.LightningModule):
    def __init__(self, args) -> None:
        super().__init__()
        self.save_hyperparameters(args)
        self.generator: Generator = Generator(
            self.hparams.z_dim, self.hparams.image_channels, self.hparams.g_dim
        )
        self.critic = Critic(self.hparams.image_channels, self.hparams.d_dim)
        self.gan_losses = losses.WGAN
        self.val_z = torch.randn(
            self.hparams.sample_count, self.hparams.z_dim, 1, 1
        )
        initialize_weights(self.generator)
        initialize_weights(self.critic)

    def forward(self, z: Tensor) -> Tensor:
        x = self.generator(z)
        return x

    def configure_optimizers(self):
        g_lr = self.hparams.g_lr
        d_lr = self.hparams.d_lr
        opt_g = torch.optim.RMSprop(self.generator.parameters(), lr=g_lr)
        opt_d = torch.optim.RMSprop(self.critic.parameters(), lr=d_lr)
        return (
            {"optimizer": opt_g, "frequency": 1},
            {"optimizer": opt_d, "frequency": 5},
        )

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
        z = torch.randn(N, self.hparams.z_dim, 1, 1).type_as(images)
        self.fake = self(z)
        output = self.critic(self.fake)
        loss = self.gan_losses.calc_real_loss(output)
        loss = {"loss": loss, "g_loss": loss.item()}
        return loss

    def discriminator_step(self, images, batch_idx):
        real_logits = self.critic(images)
        fake_logits = self.critic(self.fake.detach())

        real_loss = self.gan_losses.calc_real_loss(real_logits)
        fake_loss = self.gan_losses.calc_fake_loss(fake_logits)
        loss = real_loss + fake_loss
        loss = {"loss": loss, "d_loss": loss.item()}
        return loss

    def on_train_batch_end(self, *args, **kwargs):
        # weight clipping for Lipschitz-continuous
        c = self.hparams.critic_clip_value
        for p in self.critic.parameters():
            p.data.clamp_(-c, c)

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
            int(self.hparams.sample_count ** 0.5),
            normalize=True,
            value_range=(-1, 1),
        )
        self.logger[0].experiment.add_image(
            "sample", sample, self.current_epoch
        )

        # wandb logger
        sample = wandb.Image(samples)
        self.logger[1].experiment.log(
            {"sample": sample},
            step=self.global_step,
        )
        self.train(True)
        torch.set_grad_enabled(True)

    def to_torchscript(self, file_path: str):
        return super().to_torchscript(
            file_path, "trace", torch.rand([1, self.hparams.z_dim, 1, 1])
        )


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

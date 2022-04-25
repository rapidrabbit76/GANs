import losses
import torch

from gans import DCGAN

Tensor = torch.Tensor


class WGAN(DCGAN):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.gan_losses = losses.WGAN

    def configure_optimizers(self):
        g_lr = self.hparams.g_lr
        d_lr = self.hparams.d_lr
        opt_g = torch.optim.RMSprop(self.generator.parameters(), lr=g_lr)
        opt_d = torch.optim.RMSprop(self.discriminator.parameters(), lr=d_lr)
        return (
            {"optimizer": opt_g, "frequency": 1},
            {"optimizer": opt_d, "frequency": 5},
        )

    def discriminator_step(self, images, batch_idx):
        real_logits = self.discriminator(images)
        fake_logits = self.discriminator(self.fake.detach())

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

import losses
import torch

from gans import DCGAN

Tensor = torch.Tensor


class WGANGP(DCGAN):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.gan_losses = losses.WGAN

    def gradient_penalty(
        self, fake: torch.Tensor, real: torch.Tensor
    ) -> torch.Tensor:
        N, *_ = real.shape
        alpha = torch.rand(N, 1, 1, 1).type_as(real).expand_as(real)
        x = real * alpha + fake * (1 - alpha)
        x = torch.autograd.Variable(x, requires_grad=True)

        output = self.discriminator(x)
        gradient = torch.autograd.grad(
            inputs=x,
            outputs=output,
            grad_outputs=torch.ones_like(output),
            create_graph=True,
            retain_graph=True,
        )[0]

        gradient = gradient.view(gradient.shape[0], -1)
        gradient_norm = torch.norm(gradient, 2, dim=1)
        gradient_penalty = torch.mean(torch.square(gradient_norm - 1))
        return gradient_penalty

    def discriminator_step(self, images, batch_idx):
        fake = self.fake.detach()
        real_logits = self.discriminator(images)
        fake_logits = self.discriminator(fake)

        real_loss = self.gan_losses.calc_real_loss(real_logits)
        fake_loss = self.gan_losses.calc_fake_loss(fake_logits)
        gp = self.gradient_penalty(fake, images)
        loss = (real_loss + fake_loss) + (gp * self.hparams.LAMBDA_GP)

        loss = {
            "loss": loss,
            "d_loss": loss.item(),
            "gradient_penalty": gp.item(),
        }
        return loss

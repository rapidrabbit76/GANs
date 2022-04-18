import pytest
from easydict import EasyDict
import torch
from tests.conftest import tensor
from gans import GAN


class TestGAN:
    @pytest.fixture(scope="class")
    def args(self):
        return EasyDict(
            {
                "batch_size": 2,
                "image_size": 28,
                "image_channels": 3,
                "z_dim": 100,
                "g_dim": 16,
                "d_dim": 16,
                "sample_count": 64,
            }
        )

    @pytest.fixture(scope="class")
    def model(self, args):
        model = GAN(args)
        return model

    @pytest.fixture(scope="class")
    def z(self, args):
        return torch.rand(size=[args.batch_size, args.z_dim])

    @pytest.fixture(scope="class")
    def images(self, args):
        return tensor(args.batch_size, args.image_size, args.image_channels)

    def test_gen_disc_inference(self, model, z, images):
        N, C, H, W = images.shape
        x = model.generator(z)
        assert list(x.shape) == [N, C * H * W]
        x_ = model.discriminator(x)
        x = model.discriminator(images.view([N, -1]))
        assert list(x_.shape) == [N, 1]
        assert list(x.shape) == [N, 1]

    def test_gen_step(self, model: GAN, images):
        x = model.training_step((images, 0), 0, 0)
        assert list(x.shape) == []

    def test_disc_step(self, model: GAN, images):
        x = model.training_step((images, 0), 0, 1)
        assert list(x.shape) == []

import os

import pytest
import torch
from easydict import EasyDict
from gans.WGANGP import WGANGP

from tests.conftest import tensor


class TestWGANGP:
    @pytest.fixture(scope="class")
    def args(self):
        return EasyDict(
            {
                "batch_size": 2,
                "image_size": 64,
                "image_channels": 3,
                "z_dim": 100,
                "g_dim": 64,
                "d_dim": 64,
                "sample_count": 64,
                "LAMBDA_GP": 10.0,
            }
        )

    @pytest.fixture(scope="class")
    def model(self, args):
        model = WGANGP(args)
        return model

    @pytest.fixture(scope="class")
    def z(self, args):
        return torch.rand(size=[args.batch_size, args.z_dim, 1, 1])

    @pytest.fixture(scope="class")
    def images(self, args):
        return tensor(args.batch_size, args.image_size, args.image_channels)

    def test_gen_disc_inference(self, args, model, z, images):
        x = model.generator(z)
        assert x.shape == images.shape
        x_ = model.critic(x).view([args.batch_size, -1])
        x = model.critic(images).view([args.batch_size, -1])
        assert list(x_.shape) == [args.batch_size, 1]
        assert list(x.shape) == [args.batch_size, 1]

    def test_training_step(self, model, images):
        x = model.training_step((images, 0), 0, 0)
        assert list(x.shape) == []

        x = model.training_step((images, 0), 0, 1)
        assert list(x.shape) == []

    def test_torchscript(self, model, save_dir, z, images):
        torchscript_path = os.path.join(save_dir.name, "temp.jit")
        model.to_torchscript(torchscript_path)
        assert os.path.exists(torchscript_path)

        model = torch.jit.load(torchscript_path)
        x = model(z)
        assert x.shape == images.shape
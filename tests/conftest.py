import tempfile
import pytest
import torch


@pytest.fixture(scope="session")
def save_dir():
    return tempfile.TemporaryDirectory()


def tensor(b, size, c):
    return torch.rand([b, c, size, size])

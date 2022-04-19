from typing import Dict
import pytorch_lightning as pl
from gans.GAN.GAN import GAN
from gans.DCGAN.GAN import DCGAN


MODEL_TABLE: Dict["str", "pl.LightningModule"] = {
    "GAN": GAN,
    "DCGAN": DCGAN,
}
MODEL_LIST = list(MODEL_TABLE.keys())


def build_model(args) -> pl.LightningModule:
    datamodule = MODEL_TABLE[args.GAN](args)
    return datamodule

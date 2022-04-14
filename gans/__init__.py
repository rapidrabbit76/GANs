from typing import Dict
from gans.GAN.GAN import GAN


MODEL_TABLE: Dict["str", "pl.LightningModule"] = {
    "GAN": GAN,
}

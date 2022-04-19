from typing import Dict, Callable
from .base import BaseTransforms


TRANSFORMS_TABLE: Dict["str", Callable] = {
    "BASE": BaseTransforms,
}
TRANSFORMS_LIST = list(TRANSFORMS_TABLE.keys())


def build_transform(transform: str, **kwargs) -> Callable:
    datamodule = TRANSFORMS_TABLE[transform](**kwargs)
    return datamodule

from typing import Dict, Callable
from .base import BaseTransforms


TRANSFORMS_TABLE: Dict["str", Callable] = {
    "BASE": BaseTransforms,
}

from typing import Union, List, Tuple
import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2 as ToTensor
from PIL import Image
import cv2


class CelebATransforms:
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    def __init__(
        self,
        image_shape: List[int],
        train: Union[int, bool, str] = False,
        mean=None,
        std=None,
    ) -> None:
        self.image_shape = image_shape
        c, image_size, _ = image_shape
        train = True if isinstance(train, str) and train == "train" else False

        mean = self.mean if mean is None else mean
        std = self.std if std is None else std
        mean = mean[:c]
        std = std[:c]

        assert c == len(mean)
        assert c == len(std)

        crop_size = image_size
        image_size = int(image_size * 1.3)
        self.transforms = A.Compose(
            [
                A.Resize(image_size, image_size),
                A.CenterCrop(crop_size, crop_size),
                A.HorizontalFlip(p=1.0 if train else 0.0),
                A.Normalize(
                    mean=mean,
                    std=std,
                    max_pixel_value=255.0,
                ),
                ToTensor(),
            ]
        )

    def __call__(self, image: Union[np.ndarray, Image.Image]) -> torch.Tensor:
        image = np.array(image)
        if self.image_shape[0] == 3 and len(image.shape) < 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        image = self.transforms(image=image)["image"]
        return image

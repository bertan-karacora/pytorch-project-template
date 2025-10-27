import logging

import torch
import torchvision.transforms.v2 as tv_transforms


_LOGGER = logging.getLogger(__name__)


class ToDtype(tv_transforms.Transform):
    def __init__(self, dtype, scale=False):
        super().__init__()

        self.dtype = getattr(torch, dtype)
        self.scale = scale
        self.transform_tv = None

        self._init()

    def _init(self):
        self.transform_tv = tv_transforms.ToDtype(
            dtype=self.dtype,
            scale=self.scale,
        )

    def _transform(self, inpt):
        output = self.transform_tv(inpt)
        return output

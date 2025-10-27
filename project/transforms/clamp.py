import logging

import torch
import torchvision.transforms.v2 as tv_transforms


_LOGGER = logging.getLogger(__name__)


class Clamp(tv_transforms.Transform):
    def __init__(self, min, max):
        super().__init__()

        self.min = min
        self.max = max

    def _transform(self, inpt):
        output = torch.clamp(inpt, min=self.min, max=self.max)
        return output

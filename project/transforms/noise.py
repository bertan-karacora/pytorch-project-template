import logging

import torch
import torchvision.transforms.v2 as tv_transforms


_LOGGER = logging.getLogger(__name__)


def add_noise(inpt, noise, inplace=False):
    if not inplace:
        inpt = inpt.clone()

    output = inpt.add_(noise)
    return output


class AddNoiseGaussian(tv_transforms.Transform):
    def __init__(self, std, inplace=False):
        super().__init__()

        self.inplace = inplace
        self.std = std

    @staticmethod
    def get_params(shape, std):
        noise = torch.randn(shape) * std
        return noise

    def _transform(self, inpt):
        noise = AddNoiseGaussian.get_params(inpt.shape, self.std)
        output = add_noise(inpt, noise, inplace=False)
        return output

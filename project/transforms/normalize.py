import logging

import torch
import torchvision.transforms.v2 as tv_transforms


_LOGGER = logging.getLogger(__name__)


def denormalize(inpt, mean, std, dim=1, inplace=False):
    if not inplace:
        inpt = inpt.clone()

    mean = torch.as_tensor(mean, dtype=inpt.dtype, device=inpt.device)
    std = torch.as_tensor(std, dtype=inpt.dtype, device=inpt.device)
    if mean.ndim == 1:
        shape = [1] * inpt.ndim
        shape[dim] = -1
        mean = mean.view(*shape)
    if std.ndim == 1:
        shape = [1] * inpt.ndim
        shape[dim] = -1
        std = std.view(*shape)

    output = inpt.mul_(std).add_(mean)
    return output


class Denormalize(tv_transforms.Transform):
    def __init__(self, mean, std, dim=1, inplace=False):
        super().__init__()

        self.dim = dim
        self.inplace = inplace
        self.mean = mean
        self.std = std

    def _transform(self, inpt):
        output = denormalize(inpt, self.mean, self.std, dim=self.dim, inplace=self.inplace)
        return output

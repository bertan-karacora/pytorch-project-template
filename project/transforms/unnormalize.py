import torch
import torchvision.transforms.v2 as tv_transforms


def unnormalize(input, mean, std, inplace=False):
    if not inplace:
        input = input.clone()

    mean = torch.as_tensor(mean, dtype=input.dtype, device=input.device)
    std = torch.as_tensor(std, dtype=input.dtype, device=input.device)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)

    output = input.mul_(std).add(mean)
    return output


class Unnormalize(tv_transforms.Transform):
    def __init__(self, mean, std):
        super().__init__()

        self.mean = mean
        self.std = std

    def _transform(self, input, params):
        output = unnormalize(input, self.mean, self.std)
        return output

import logging

import torch

import project.config as config
import project.libs.utils_data as utils_data
import project.libs.utils_torch as utils_torch


_LOGGER = logging.getLogger(__name__)


@torch.no_grad()
def sample(model, split, num_samples=None):
    device = utils_torch.get_device(config._DEVICE)

    inpt, target = utils_data.sample(split=split, num_samples=num_samples)

    model = model.to(device)
    inpt = utils_data.move_batch(inpt, device)
    target = utils_data.move_batch(target, device)

    output = model(inpt)

    return output, target


def freeze(model):
    for param in model.parameters():
        param.requires_grad = False

    _LOGGER.info(f"Froze parameters of model: {type(model).__name__}")

    return model


def unfreeze(model):
    for param in model.parameters():
        param.requires_grad = True

    _LOGGER.info(f"Unfroze parameters of model: {type(model).__name__}")

    return model

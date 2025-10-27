import logging

import torch

_LOGGER = logging.getLogger(__name__)


def polar_to_cartesian(rs, phis):
    xs = rs * torch.cos(phis)
    ys = rs * torch.sin(phis)
    return xs, ys


def complex_to_quaternion(real, imag):
    # Assume rotation in xy-plane
    quaternion = torch.tensor([0.0, 0.0, real, imag])
    return quaternion

import logging

import torch


_LOGGER = logging.getLogger(__name__)


def elbo_gaussian(inpt, target):
    """Evidence lower bound assuming a Gaussian distribution.
    A derivation of the closed form solution: https://johfischer.com/2022/05/21/closed-form-solution-of-kullback-leibler-divergence-between-two-gaussians"""

    mean, var_log = inpt["mean"], inpt["var_log"]

    loss = torch.mean(torch.sum(-0.5 * (1 + var_log - mean**2 - torch.exp(var_log)), dim=1), dim=0)
    return loss


class ELBOGaussian(torch.nn.Module):
    """Evidence lower bound assuming a Gaussian distribution."""

    def __init__(self):
        super().__init__()

    def forward(self, inpt, target):
        loss = elbo_gaussian(inpt, target)
        return loss

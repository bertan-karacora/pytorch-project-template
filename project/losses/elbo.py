import torch


class ELBOGaussian(torch.nn.Module):
    """Evidence lower bound assuming a Gaussian distribution.
    A derivation of the closed form solution: https://johfischer.com/2022/05/21/closed-form-solution-of-kullback-leibler-divergence-between-two-gaussians"""

    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        mean, log_var = input["mean"], input["log_var"]

        loss = torch.mean(torch.sum(-0.5 * (1 + log_var - mean**2 - torch.exp(log_var)), dim=1), dim=0)
        return loss

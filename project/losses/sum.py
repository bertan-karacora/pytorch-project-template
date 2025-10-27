import logging

import torch

_LOGGER = logging.getLogger(__name__)


class SumWeighted(torch.nn.Module):
    def __init__(self, modules, weights):
        super().__init__()

        self.modules_sum = torch.nn.ModuleList(modules)
        self.weights = weights

    def forward(self, inpt, target):
        output = self.weights[0] * self.modules_sum[0](inpt, target)
        for i in range(1, len(self.modules_sum)):
            output += self.weights[i] * self.modules_sum[i](inpt, target)

        return output

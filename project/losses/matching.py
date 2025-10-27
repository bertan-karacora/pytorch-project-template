import logging

import numpy as np
import scipy as sp
import torch


_LOGGER = logging.getLogger(__name__)


def outer(a, b):
    size_a = tuple(a.size()) + (b.size()[-1],)
    size_b = tuple(b.size()) + (a.size()[-1],)
    a = a.unsqueeze(dim=-1).expand(*size_a)
    b = b.unsqueeze(dim=-2).expand(*size_b)
    return a, b


def match_hungarian(predictions, targets):
    predictions = predictions.permute(0, 2, 1)
    targets = targets.permute(0, 2, 1)

    predictions, targets = outer(predictions, targets)
    error = torch.sqrt(torch.mean((predictions - targets) ** 2, dim=1))

    # Detach, we only need matching
    costs = error.detach().cpu().numpy()
    costs[costs == np.inf] = 0.0
    indices = [sp.optimize.linear_sum_assignment(c) for c in costs]

    return indices, error


def loss_matching(indices, squared_error):
    losses = [sample[row_idx, col_idx].mean() for sample, (row_idx, col_idx) in zip(squared_error, indices)]
    loss = torch.mean(torch.stack(list(losses)))
    return loss


class Matching(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        # Multiple samples
        samples_loss = []
        for sample in input:
            # Multiple layers
            loss = 0.0
            for sample_layer in sample:
                indices, errors = sample_layer
                loss += loss_matching(indices, errors)

            samples_loss.append(loss)
        output = torch.mean(torch.stack(samples_loss))

        return output

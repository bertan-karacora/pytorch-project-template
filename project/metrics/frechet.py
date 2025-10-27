import logging

import torch
import torchmetrics.image.fid as fid_tm

import project.transforms.denormalize as denormalize


_LOGGER = logging.getLogger(__name__)


class FrechetInceptionDistance(torch.nn.Module):
    def __init__(self, use_denormalize=False, kwargs_denormalize=None, **kwargs):
        super().__init__()

        self.kwargs_denormalize = kwargs_denormalize or {}
        self.metric_tm = fid_tm.FrechetInceptionDistance(**kwargs)
        self.use_denormalize = use_denormalize

    @torch.no_grad()
    def forward(self, inpt, target):
        if self.use_denormalize:
            inpt = denormalize.denormalize(inpt, **self.kwargs_denormalize)
            target = denormalize.denormalize(target, **self.kwargs_denormalize)

        self.metric_tm.update(inpt, real=False)
        self.metric_tm.update(target, real=True)
        output = self.metric_tm.compute()

        return output

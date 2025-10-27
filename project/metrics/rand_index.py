import logging

import torch
from torchmetrics.clustering import AdjustedRandScore

from sklearn.cluster import KMeans


_LOGGER = logging.getLogger(__name__)


# Note: Only working on cpu for now
class AdjustedRandIndexKMeans(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.metric_tm = AdjustedRandScore()

    @torch.no_grad()
    def forward(self, inpt, target):
        latents_anchor = inpt["anchor"]
        target_anchor = target["anchor"]

        num_classes = len(torch.unique(target_anchor))
        clusters = KMeans(n_clusters=num_classes).fit_predict(latents_anchor.numpy(force=True))

        output = self.metric_tm(torch.as_tensor(clusters, device="cpu"), target_anchor.cpu())

        return output

import torch
from torchmetrics.clustering import AdjustedRandScore

from sklearn.cluster import KMeans


class AdjustedRandIndexKMeans(torch.nn.Module):
    """Note: Only working on cpu for now"""

    def __init__(self):
        super().__init__()

        self.metric_tm = AdjustedRandScore()

    @torch.no_grad()
    def forward(self, input, targets):
        latents_anchor = input["anchor"]
        targets_anchor = targets["anchor"]

        num_classes = len(torch.unique(targets_anchor))
        clusters = KMeans(n_clusters=num_classes).fit_predict(latents_anchor.numpy(force=True))

        output = self.metric_tm(torch.as_tensor(clusters, device="cpu"), targets_anchor.cpu())

        return output

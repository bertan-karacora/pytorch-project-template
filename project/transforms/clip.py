import torch
import torchvision.transforms.v2 as tv_transforms


class Clip(tv_transforms.Transform):
    def __init__(self, min, max):
        super().__init__()

        self.min = min
        self.max = max

    def _transform(self, features, params):
        features_transformed = torch.clip(features, min=self.min, max=self.max)
        return features_transformed

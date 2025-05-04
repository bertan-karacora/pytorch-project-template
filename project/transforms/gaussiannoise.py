import torch
import torchvision.transforms.v2 as tv_transforms


class GaussianNoise(tv_transforms.Transform):
    def __init__(self, mean=0.0, std=1.0):
        super().__init__()

        self.mean = torch.as_tensor(mean)
        self.std = torch.as_tensor(std)

    def _transform(self, features, params):
        features_transformed = features + torch.normal(self.mean, self.std, size=features.shape)
        return features_transformed

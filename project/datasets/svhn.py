from pathlib import Path

from torch.utils.data import Dataset
import torchvision as tv

import assignment.libs.utils_data as utils_data


class SVHN(Dataset):
    def __init__(self, path, split, transform=None, transform_target=None, use_download=True, ratio_validation_to_training=0.2):
        self.dataset_tv = None
        self.path = Path(path)
        self.ratio_validation_to_training = ratio_validation_to_training
        self.split = split
        self.transform = transform
        self.transform_target = transform_target
        self.use_download = use_download

        self._init()

    def _init(self):
        self.dataset_tv = tv.datasets.SVHN(
            root=self.path,
            split=self.split if self.split not in ["training", "validation"] else "train",
            transform=self.transform,
            target_transform=self.transform_target,
            download=self.use_download,
        )

        if self.split in ["training", "validation"]:
            subset_training, subset_validation = utils_data.split_into_training_and_validation(self.dataset_tv, self.ratio_validation_to_training)
            self.dataset_tv = subset_training if self.split == "training" else subset_validation

    def __len__(self):
        length = len(self.dataset_tv)
        return length

    def __getitem__(self, index):
        features, target = self.dataset_tv[index]
        return features, target

    def __str__(self):
        s = f"""Dataset {self.__class__.__name__}
    Number of samples: {self.__len__()}
    Path: {self.path}
    Split: {self.split}
    Transform of samples: {self.transform}
    Transform of targets: {self.transform_target}"""
        return s

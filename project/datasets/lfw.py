from pathlib import Path
import random

import numpy as np
import torch
import torchvision as tv

import assignment.libs.utils_data as utils_data


class LFWPeople(torch.utils.data.Dataset):
    split_to_split_tv = {
        "training": "train",
        "validation": "train",
        "test": "test",
    }

    def __init__(self, path, split, transform=None, transform_target=None, use_download=False, type_funneling="deepfunneled", ratio_validation_to_training=0.2, use_stratification=False):
        self.dataset_tv = None
        self.labelset = None
        self.path = Path(path)
        self.ratio_validation_to_training = ratio_validation_to_training
        self.split = split
        self.transform = transform
        self.transform_target = transform_target
        self.type_funneling = type_funneling
        self.use_download = use_download
        self.use_stratification = use_stratification

        self._init()

    def _init(self):
        self.dataset_tv = tv.datasets.LFWPeople(
            root=self.path,
            split=self.split_to_split_tv[self.split],
            image_set=self.type_funneling,
            transform=self.transform,
            target_transform=self.transform_target,
            download=self.use_download,
        )
        self.labelset = np.asarray(list(self.dataset_tv.class_to_idx.keys()))

        if self.split in ["training", "validation"]:
            labels = self.dataset_tv.targets if self.use_stratification else None
            subset_training, subset_validation = utils_data.split_into_training_and_validation(self.dataset_tv, self.ratio_validation_to_training, labels=labels)
            self.dataset_tv = subset_training if self.split == "training" else subset_validation

    def __len__(self):
        length = len(self.dataset_tv)
        return length

    def __getitem__(self, index):
        features, target = self.dataset_tv[index]
        return features, target

    def __str__(self):
        s = f"""Dataset {self.__class__.__name__}
    Number of samples: {len(self)}
    Path: {self.path}
    Split: {self.split}
    Transform of samples: {self.transform}
    Transform of targets: {self.transform_target}"""
        return s


class LFWTriplets(LFWPeople):
    def __init__(self, *args, use_remove_single_occurances=True, **kwargs):
        self.indices = None
        self.targets = None
        self.use_remove_single_occurances = use_remove_single_occurances

        super().__init__(*args, **kwargs)

    def _init(self):
        super()._init()

        if self.use_remove_single_occurances:
            self._remove_single_occurances()

        self.targets = torch.as_tensor([target for _, target in self.dataset_tv])
        self.indices = torch.arange(len(self))

    def _remove_single_occurances(self):
        indices = torch.arange(len(self))
        targets = torch.as_tensor([target for _, target in self.dataset_tv])
        counts_labels = torch.bincount(targets, minlength=len(self.labelset))

        indices_valid = indices[counts_labels[targets] > 1]
        self.dataset_tv = torch.utils.data.Subset(self.dataset_tv, indices_valid)

    def __getitem__(self, index):
        features_anchor, target_anchor = self.dataset_tv[index]

        indices_positive_all = self.indices[self.targets == target_anchor]
        indices_positive = indices_positive_all[indices_positive_all != index] if self.use_remove_single_occurances else indices_positive_all
        indices_negative = self.indices[self.targets != target_anchor]

        index_positive = random.choice(indices_positive).item()
        index_negative = random.choice(indices_negative).item()
        features_positive, target_positive = self.dataset_tv[index_positive]
        features_negative, target_negative = self.dataset_tv[index_negative]

        features = dict(anchor=features_anchor, positive=features_positive, negative=features_negative)
        targets = dict(anchor=target_anchor, positive=target_positive, negative=target_negative)
        return features, targets

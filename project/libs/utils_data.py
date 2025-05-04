import sklearn.model_selection as model_selection
import torch

import self_supervised_learning_of_depth_and_motion.config as config
import self_supervised_learning_of_depth_and_motion.libs.factory as factory
import self_supervised_learning_of_depth_and_motion.transforms.unnormalize as unnorm


def split_into_training_and_validation(dataset, ratio_validation_to_training, labels=None):
    indices = list(range(len(dataset)))

    if ratio_validation_to_training == 0.0:
        indices_training = indices
        indices_validation = []
    elif ratio_validation_to_training == 1.0:
        indices_training = []
        indices_validation = indices
    else:
        indices_training, indices_validation = model_selection.train_test_split(indices, test_size=ratio_validation_to_training, stratify=labels)

    subset_training = torch.utils.data.Subset(dataset, indices_training)
    subset_validation = torch.utils.data.Subset(dataset, indices_validation)
    return subset_training, subset_validation


def sample(split, num_samples=None, use_unnormalize=False, use_labelset=False):
    num_samples = num_samples or config.DATA[split]["dataloader"]["kwargs"]["batch_size"]

    dataset, dataloader = factory.create_dataset_and_dataloader(split=split)

    features, target = sample_dataloader(dataloader, split=split, num_samples=num_samples, use_unnormalize=use_unnormalize)

    if use_labelset:
        target = targets_to_labels(target, dataset)

    return features, target


def sample_dataloader(dataloader, split="test", num_samples=None, use_unnormalize=False):
    num_samples = num_samples or config.DATA[split]["dataloader"]["kwargs"]["batch_size"]

    features, target = next(iter(dataloader))
    features = slice_items(features, num_samples)
    target = slice_items(target, num_samples)

    if use_unnormalize:
        features = unnormalize(features, split=split)
        if "use_features_as_target" in config.DATA[split]["dataset"]["kwargs"] and config.DATA[split]["dataset"]["kwargs"]["use_features_as_target"]:
            target = unnormalize(target, split=split)

    return features, target


def sample_dataset(dataset, indices):
    list_features, list_targets = map(list, zip(*[dataset[i] for i in indices]))
    return list_features, list_targets


def targets_to_labels(items, dataset):
    if isinstance(items, dict):
        for key in items.keys():
            items[key] = dataset.labelset[items[key]]
    else:
        items = dataset.labelset[items]

    return items


def move_items(items, device):
    if isinstance(items, dict):
        for key in items.keys():
            items[key] = items[key].to(device)
    else:
        items = items.to(device)

    return items


def count_items(items):
    if isinstance(items, dict):
        key = next(iter(items))
        count = len(items[key])
    else:
        count = len(items)

    return count


def slice_items(items, num_samples):
    if isinstance(items, dict):
        items = {key: value[:num_samples] for key, value in items.items()}
    else:
        items = items[:num_samples]

    return items


def unnormalize(items, split="test"):
    if isinstance(items, dict):
        for key in items.keys():
            items[key] = unnorm.unnormalize(items[key], mean=config.DATA[split]["dataset"]["mean"], std=config.DATA[split]["dataset"]["std"])
    else:
        items = unnorm.unnormalize(items, mean=config.DATA[split]["dataset"]["mean"], std=config.DATA[split]["dataset"]["std"])

    return items

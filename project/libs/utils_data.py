import logging

import sklearn.model_selection as model_selection
import torch

import project.config as config
import project.libs.factory as factory
import project.transforms.normalize as normalize


_LOGGER = logging.getLogger(__name__)


def split_into_training_and_validation(dataset, ratio_validation_to_training, labels=None):
    idxs = list(range(len(dataset)))

    if ratio_validation_to_training == 0.0:
        idxs_training = idxs
        idxs_validation = []
    elif ratio_validation_to_training == 1.0:
        idxs_training = []
        idxs_validation = idxs
    else:
        idxs_training, idxs_validation = model_selection.train_test_split(idxs, test_size=ratio_validation_to_training, stratify=labels)

    subset_training = torch.utils.data.Subset(dataset, idxs_training)
    subset_validation = torch.utils.data.Subset(dataset, idxs_validation)

    _LOGGER.info(f"Split dataset into validation and training subsets: len_dataset={len(dataset)}, len_dataset_validation={len(subset_validation)}, len_dataset_training={len(dataset_training)}")

    return subset_training, subset_validation


def sample(split, num_samples=None, use_denormalize=False, use_classes=False):
    num_samples = num_samples or config.DATA[split]["dataloader"]["kwargs"]["batch_size"]

    dataset, dataloader = factory.create_dataset_and_dataloader(split)

    inpt, target = sample_dataloader(dataloader, split, num_samples, use_denormalize)

    if use_classes:
        target = target_to_labels(target, dataset)

    return inpt, target


def sample_dataloader(dataloader, split="test", num_samples=None, use_denormalize=False):
    num_samples = num_samples or config.DATA[split]["dataloader"]["kwargs"]["batch_size"]

    inpt, target = next(iter(dataloader))
    inpt = slice_items(inpt, num_samples)
    target = slice_items(target, num_samples)

    if use_denormalize:
        inpt = denormalize(inpt, split=split)
        if "use_inpt_as_target" in config.DATA[split]["dataset"]["kwargs"] and config.DATA[split]["dataset"]["kwargs"]["use_inpt_as_target"]:
            target = denormalize(target, split=split)

    return inpt, target


def sample_dataset(dataset, idxs):
    list_inpts, list_targets = map(list, zip(*[dataset[i] for i in idxs]))
    return list_inpts, list_targets


def target_to_labels(target, dataset):
    if isinstance(target, dict):
        for key in target.keys():
            target[key] = dataset.classes[target[key]]
    else:
        target = dataset.classes[target]

    return target


def move_batch(batch, device):
    if isinstance(batch, dict):
        for key in batch.keys():
            batch[key] = batch[key].to(device)
    else:
        batch = batch.to(device)

    return batch


def count_items(batch):
    if isinstance(batch, dict):
        key = next(iter(batch))
        count = len(batch[key])
    else:
        count = len(batch)

    return count


def slice_items(items, num_samples):
    if isinstance(items, dict):
        items = {key: value[:num_samples] for key, value in items.items()}
    else:
        items = items[:num_samples]

    return items


def denormalize(items, split="test"):
    if isinstance(items, dict):
        for key in items.keys():
            items[key] = normalize.denormalize(items[key], mean=config.DATA[split]["dataset"]["mean"], std=config.DATA[split]["dataset"]["std"])
    else:
        items = normalize.denormalize(items, mean=config.DATA[split]["dataset"]["mean"], std=config.DATA[split]["dataset"]["std"])

    return items

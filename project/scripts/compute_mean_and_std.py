import argparse
import logging

import torch
from tqdm import tqdm

import self_supervised_learning_of_depth_and_motion.config as config
import self_supervised_learning_of_depth_and_motion.libs.factory as factory

_LOGGER = logging.getLogger(__name__)


def parse_args():
    names_configs_available = config.list_available()

    parser = argparse.ArgumentParser(description="Compute mean and standard deviation of a dataset.")
    parser.add_argument("--config", help=f"Config name selected from {names_configs_available}", choices=names_configs_available, required=True)
    parser.add_argument("--split", help=f"Dataset split", default="training")
    args = parser.parse_args()

    return args.config, args.split


def compute_mean(dataloader):
    sum = torch.Tensor(0.0)

    progressbar = tqdm(dataloader, total=len(dataloader), disable=config.LOGGING["tqdm"]["period"] == 0)
    for i, item in enumerate(progressbar, start=1):
        features = item["image"]
        features = features.view(*features.shape[:2], -1)

        sum += torch.sum(torch.mean(features, dim=2), dim=0)

        if i % config.LOGGING["tqdm"]["period"] == 0:
            mean_sum = torch.mean(sum)
            progressbar.set_description(f"Computing mean: batch {i:03d} | mean_sum {mean_sum.item():.3f}")

    mean = sum / len(dataloader.dataset)

    return mean


def compute_std(dataloader, mean):
    sum_variance = torch.Tensor(0.0)

    progressbar = tqdm(dataloader, total=len(dataloader), disable=config.LOGGING["tqdm"]["period"] == 0)
    for i, item in enumerate(progressbar, start=1):
        features = item["image"]
        features = features.view(*features.shape[:2], -1)

        # Beware numerical problems. Use a biased estimate.
        sum_variance += torch.sum(torch.mean(features - mean[:, None] ** 2, dim=2), dim=0)

        if i % config.LOGGING["tqdm"]["period"] == 0:
            mean_sum_variance = torch.mean(sum_variance)
            progressbar.set_description(f"Computing std: batch {i:03d} | mean_sum_variance {mean_sum_variance.item():.3f}")

    std = torch.sqrt(sum_variance / len(dataloader.dataset))

    return std


def compute_mean_and_std(name_config, split="training"):
    _LOGGER.info(f"Computing mean and standard deviation ...")

    config.apply_config_preset(name_config)

    dataset, dataloader = factory.create_dataset_and_dataloader(split=split)

    _LOGGER.info("Dataset")
    _LOGGER.info(dataset)

    # mean = compute_mean(dataloader)
    mean = torch.Tensor([0.5, 0.5, 0.5])
    std = compute_std(dataloader, mean)

    _LOGGER.info(f"Mean: {mean}")
    _LOGGER.info(f"Standard deviation: {std}")

    _LOGGER.info(f"Computing mean and standard deviation finished")


def main():
    name_config, split = parse_args()
    compute_mean_and_std(name_config, split)


if __name__ == "__main__":
    main()

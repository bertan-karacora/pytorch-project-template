import logging
from pathlib import Path

import torch
import torchinfo

import project.config as config
import project.libs.collations as collations
import project.libs.utils_import as utils_import
import project.libs.utils_model as utils_model


_LOGGER = logging.getLogger(__name__)


def create_transform(dict_transform):
    kwargs_transform = dict_transform["kwargs"] if "kwargs" in dict_transform else {}

    class_transform = utils_import.import_transform(dict_transform["name"])
    if dict_transform["name"] in ["Compose", "RandomApply", "RandomChoice", "RandomOrder"]:
        transform = class_transform(
            transforms=create_transforms(dict_transform["transforms"]),
            **kwargs_transform,
        )
    else:
        transform = class_transform(**kwargs_transform)

    _LOGGER.info(f"Created transform: '{type(transform).__name__}'")
    _LOGGER.debug(f"Transform: '{transform}'")

    return transform


def create_transforms(dicts_transforms):
    transforms = []
    for dict_transform in dicts_transforms:
        transform = create_transform(dict_transform)
        transforms.append(transform)

    return transforms


def create_transform_dataset(split, thing):
    dict_transform = config.DATA[split]["dataset"]["transform"][thing]
    transform = create_transform(dict_transform) if dict_transform else None

    return transform


def create_dataset(split):
    dict_dataset = config.DATA[split]["dataset"]
    kwargs_dataset = dict_dataset["kwargs"] if "kwargs" in dict_dataset else {}

    class_dataset = utils_import.import_dataset(dict_dataset["name"])
    dataset_split = class_dataset(
        path=Path(config._PATH_DIR_DATA) / dict_dataset["name"].lower(),
        split=split,
        transform_features=create_transform_dataset(split, "features"),
        transform_targets=create_transform_dataset(split, "targets"),
        **kwargs_dataset,
    )

    _LOGGER.info(f"Created dataset: '{type(dataset_split).__name__}'")
    _LOGGER.debug(f"Dataset: '{dataset_split}'")

    return dataset_split


def create_collation(split):
    dict_collation = config.DATA[split]["dataloader"]["collation"]
    kwargs_collation = dict_collation["kwargs"] if "kwargs" in dict_collation else {}

    create_collation_fn = getattr(collations, dict_collation["name"])
    collate_fn = create_collation_fn(**kwargs_collation)

    _LOGGER.info(f"Created collation: '{collate_fn.__name__}'")
    _LOGGER.debug(f"Collation: '{collate_fn}'")

    return collate_fn


def create_dataloader(dataset_split, split):
    dict_dataloader = config.DATA[split]["dataloader"]
    kwargs_dataloader = dict_dataloader["kwargs"] if "kwargs" in dict_dataloader else {}

    dataloader_split = torch.utils.data.DataLoader(
        dataset_split,
        num_workers=config._NUM_WORKERS_DATALOADING,
        collate_fn=create_collation(split) if "collation" in dict_dataloader else None,
        **kwargs_dataloader,
    )

    _LOGGER.info(f"Created dataloader: '{type(dataloader_split).__name__}'")
    _LOGGER.debug(f"Dataloader: '{dataloader_split}'")

    return dataloader_split


def create_dataset_and_dataloader(split):
    dataset_split = create_dataset(split)
    dataloader_split = create_dataloader(dataset_split, split)

    return dataset_split, dataloader_split


def create_model():
    dict_model = config.MODEL
    kwargs_model = dict_model["kwargs"] if "kwargs" in dict_model else {}

    class_model = utils_import.import_model(dict_model["name"])
    model = class_model(**kwargs_model).eval()

    _LOGGER.info(f"Created model: '{type(model).__name__}'")
    _LOGGER.debug(f"Model: '{model}'")

    # TODO: This is bad.
    if "transfer" in dict_model:
        if "epochs_freeze" in dict_model["transfer"] and dict_model["transfer"]["epochs_freeze"] > 0:
            model = utils_model.freeze(model)

        for dict_layer in dict_model["transfer"]["layers"]:
            dict_model_layer = dict_layer["model"]
            class_model_layer = utils_import.import_model(dict_model_layer["name"])
            model_layer = class_model_layer(**dict_model_layer["kwargs"]).eval()

            setattr(model, dict_layer["name"], model_layer)

    # TODO: This isn't very good either
    try:
        _LOGGER.info(torchinfo.summary(model, [config.MODEL["shape_input"]], verbose=0))
    except Exception as e:
        _LOGGER.exception(e)

    return model


def create_criterion(dict_criterion=None):
    dict_criterion = dict_criterion or config.CRITERION
    kwargs_criterion = dict_criterion["kwargs"] if "kwargs" in dict_criterion else {}

    class_criterion = utils_import.import_criterion(dict_criterion["name"])
    if dict_criterion["name"] in ["SumWeighted"]:
        criterion = class_criterion(
            modules=create_criteria(dict_criterion["modules"]),
            **kwargs_criterion,
        )
    else:
        criterion = class_criterion(**kwargs_criterion)

    _LOGGER.info(f"Created criterion: '{type(criterion).__name__}'")
    _LOGGER.debug(f"Criterion: '{criterion}'")

    return criterion


def create_criteria(dicts_criteria=None):
    criteria = []
    for dict_criterion in dicts_criteria:
        criterion = create_criterion(dict_criterion)
        criteria.append(criterion)

    return criteria


def create_measurer(dict_measurer):
    kwargs_measurer = dict_measurer["kwargs"] if "kwargs" in dict_measurer else {}

    class_measurer = utils_import.import_metric(dict_measurer["name"])
    measurer = class_measurer(**kwargs_measurer)

    _LOGGER.info(f"Created measurer: '{type(measurer).__name__}'")
    _LOGGER.debug(f"Measurer: '{measurer}'")

    return measurer


def create_measurers(split):
    measurers = []
    dicts_measurers = config.MEASURERS[split] if split in config.MEASURERS else config.MEASURERS
    for dict_measurer in dicts_measurers:
        measurer = create_measurer(dict_measurer)
        measurers.append(measurer)

    return measurers


def create_optimizer(params):
    dict_optimizer = config.TRAINING["optimizer"]
    kwargs_optimizer = dict_optimizer["kwargs"] if "kwargs" in dict_optimizer else {}

    class_optimizer = getattr(torch.optim, dict_optimizer["name"])
    optimizer = class_optimizer(params, **kwargs_optimizer)

    _LOGGER.info(f"Created optimizer: '{type(optimizer).__name__}'")
    _LOGGER.debug(f"Optimizer: '{optimizer}'")

    return optimizer


def create_scheduler(optimizer, dict_scheduler=None):
    dict_scheduler = dict_scheduler or config.TRAINING["scheduler"]
    kwargs_scheduler = dict_scheduler["kwargs"] if "kwargs" in dict_scheduler else {}

    class_scheduler = getattr(torch.optim.lr_scheduler, dict_scheduler["name"])
    if dict_scheduler["name"] in ["SequentialLR"]:
        scheduler = class_scheduler(
            optimizer=optimizer,
            schedulers=create_schedulers(optimizer, dict_scheduler["schedulers"]),
            **kwargs_scheduler,
        )
    else:
        scheduler = class_scheduler(optimizer, **kwargs_scheduler)

    _LOGGER.info(f"Created scheduler: '{type(scheduler).__name__}'")
    _LOGGER.debug(f"Scheduler: '{scheduler}'")

    return scheduler


def create_schedulers(optimizer, dicts_schedulers):
    schedulers = []
    for dict_scheduler in dicts_schedulers:
        scheduler = create_scheduler(optimizer, dict_scheduler)
        schedulers.append(scheduler)

    return schedulers

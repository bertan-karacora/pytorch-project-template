"""Utility module to handle dynamic imports.
This module allows to automatically search for datasets, models, etc., in custom locations and in external libraries such as torchvision and torchmetrics.
E.g., import_dataset(MNIST) will import the dataset class MNIST from torchvision except its definition is overwritten in this package.
Overwriting is as simple as adding an object definition of the targeted type (class, function, etc.) somewhere in the directories of this package,
i.e., creating a class MNIST in any file in the datasets subpackage without the need of changing names in configs or in imports throughout the codebase.
"""

import inspect
import logging


_LOGGER = logging.getLogger(__name__)


def import_dataset(name):
    """Return dataset class if it exists in custom datasets or torchvision."""
    import project.datasets as custom_datasets
    import torchvision.datasets as tv_datasets

    modules = [custom_datasets, tv_datasets]

    for module in modules:
        if hasattr(module, name):
            class_found = getattr(module, name)
            if inspect.isclass(class_found):
                _LOGGER.debug(f"Found dataset: name={name}, module={module.__name__}")
                return class_found

    raise ImportError(f"Dataset '{name}' not found")


def import_model(name):
    """Return model class or factory function if it exists in custom models or torchvision."""
    import project.models as custom_models
    import torchvision.models as tv_models

    modules = [custom_models, tv_models]

    for module in modules:
        if hasattr(module, name):
            class_or_function_found = getattr(module, name)
            if inspect.isclass(class_or_function_found) or inspect.isfunction(class_or_function_found):
                _LOGGER.debug(f"Found model: name={name}, module={module.__name__}")
                return class_or_function_found

    raise ImportError(f"Model '{name}' not found")


def import_module(name):
    """Return module class or factory function if it exists in custom models or torchvision."""
    import project.models as custom_models
    import torch.nn as torch_nn

    modules = [custom_models, torch_nn]

    for module in modules:
        if hasattr(module, name):
            class_or_function_found = getattr(module, name)
            if inspect.isclass(class_or_function_found) or inspect.isfunction(class_or_function_found):
                _LOGGER.debug(f"Found torch module: name={name}, module={module.__name__}")
                return class_or_function_found

    raise ImportError(f"Torch module '{name}' not found")


def import_transform(name):
    """Return transform class if it exists in custom transform or torchvision."""
    import project.transforms as custom_transforms
    import torchvision.transforms.v2 as tv_transforms

    modules = [custom_transforms, tv_transforms]

    for module in modules:
        if hasattr(module, name):
            class_found = getattr(module, name)
            if inspect.isclass(class_found):
                _LOGGER.debug(f"Found transform: name={name}, module={module.__name__}")
                return class_found

    raise ImportError(f"Transform '{name}' not found")


def import_criterion(name):
    """Return loss class if it exists in custom losses or Pytorch."""
    import project.losses as custom_losses
    import torch.nn as torch_nn

    modules = [custom_losses, torch_nn]

    for module in modules:
        if hasattr(module, name):
            class_found = getattr(module, name)
            if inspect.isclass(class_found):
                _LOGGER.debug(f"Found criterion: name={name}, module={module.__name__}")
                return class_found

    raise ImportError(f"Loss '{name}' not found")


def import_metric(name):
    """Return metric class if it exists in custom metrics, custom losses, Pytorch, or Torchmetrics."""
    import project.losses as custom_losses
    import project.metrics as custom_metrics
    import torch.nn as torch_nn
    import torchmetrics as tm
    import torchmetrics.classification as tm_classification

    modules = [custom_metrics, custom_losses, torch_nn, tm, tm_classification]

    for module in modules:
        if hasattr(module, name):
            class_found = getattr(module, name)
            if inspect.isclass(class_found):
                _LOGGER.debug(f"Found metric: name={name}, module={module.__name__}")
                return class_found

    raise ImportError(f"Metric '{name}' not found")

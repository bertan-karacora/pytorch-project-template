"""Utility module to handle dynamic imports.
    This module allows to automatically search for datasets, models, etc., in custom locations and in external libraries such as torchvision.
    Names in torchvision may be overwritten without the need of changing names in configs or in imports throughout the codebase.
"""

import inspect


def import_dataset(name):
    """Return dataset class if it exists in custom datasets or torchvision."""
    import self_supervised_learning_of_depth_and_motion.datasets as custom_datasets
    import torchvision.datasets as tv_datasets

    modules = [custom_datasets, tv_datasets]

    for module in modules:
        if hasattr(module, name):
            class_found = getattr(module, name)
            if inspect.isclass(class_found):
                return class_found

    raise ImportError(f"Dataset '{name}' not found")


def import_model(name):
    """Return model class or factory function if it exists in custom models or torchvision."""
    import self_supervised_learning_of_depth_and_motion.models as custom_models
    import torchvision.models as tv_models

    modules = [custom_models, tv_models]

    for module in modules:
        if hasattr(module, name):
            class_or_function_found = getattr(module, name)
            if inspect.isclass(class_or_function_found) or inspect.isfunction(class_or_function_found):
                return class_or_function_found

    raise ImportError(f"Model '{name}' not found")


def import_module(name):
    """Return module class or factory function if it exists in custom models or torchvision."""
    import self_supervised_learning_of_depth_and_motion.models as custom_models
    import torch.nn as torch_nn

    modules = [custom_models, torch_nn]

    for module in modules:
        if hasattr(module, name):
            class_or_function_found = getattr(module, name)
            if inspect.isclass(class_or_function_found) or inspect.isfunction(class_or_function_found):
                return class_or_function_found

    raise ImportError(f"Module '{name}' not found")


def import_transform(name):
    """Return transform class if it exists in custom transform or torchvision."""
    import self_supervised_learning_of_depth_and_motion.transforms as custom_transforms
    import torchvision.transforms.v2 as tv_transforms

    modules = [custom_transforms, tv_transforms]

    for module in modules:
        if hasattr(module, name):
            class_found = getattr(module, name)
            if inspect.isclass(class_found):
                return class_found

    raise ImportError(f"Transform '{name}' not found")


def import_criterion(name):
    """Return loss class if it exists in custom losses or Pytorch."""
    import self_supervised_learning_of_depth_and_motion.losses as custom_losses
    import torch.nn as torch_nn

    modules = [custom_losses, torch_nn]

    for module in modules:
        if hasattr(module, name):
            class_found = getattr(module, name)
            if inspect.isclass(class_found):
                return class_found

    raise ImportError(f"Loss '{name}' not found")


def import_metric(name):
    """Return metric class if it exists in custom metrics, custom losses, Pytorch, or Torchmetrics."""
    import self_supervised_learning_of_depth_and_motion.losses as custom_losses
    import self_supervised_learning_of_depth_and_motion.metrics as custom_metrics
    import torch.nn as torch_nn
    import torchmetrics as tm
    import torchmetrics.classification as tm_classification

    modules = [custom_metrics, custom_losses, torch_nn, tm, tm_classification]

    for module in modules:
        if hasattr(module, name):
            class_found = getattr(module, name)
            if inspect.isclass(class_found):
                return class_found

    raise ImportError(f"Metric '{name}' not found")

"""
Global package configuration module.

Config attributes are represented as module-level globals.
Package-wide static config settings are defined in a top-level config file within the package and indicated by an underscore prefix.
E.g., the setting 'num_workers' in the 'config.yaml' file will be accessible as 'config._NUM_WORKERS'.
Loading the package-wide settings and the package configuration (e.g. seeding RNGs) is only done when the module is first imported.
Config settings that may change during runtime (such as settings for training a model) are dynamically loaded from separate config files.
E.g., the nested setting 'data.training.dataset.name' in a selected config will be accessible as 'config.DATA["training"]["dataset"]["name"]'.

Note:
    While generally discouraged, using module-level globals is considered acceptable for setting up a package-wide singleton configuration.
    See https://stackoverflow.com/questions/5055042/whats-the-best-practice-using-a-settings-file-in-python
    See https://stackoverflow.com/questions/30556857/creating-a-static-class-with-no-instances
"""

import importlib.resources as resources
import logging
from pathlib import Path
import pprint
import random

import numpy as np
import torch

import project.libs.utils_io as utils_io


_PATH_CONFIG_PACKAGE = str(resources.files(__package__) / "config.yaml")
_SEED_RNGS = 42
_LEVEL_LOGGING_DEFAULT = "INFO"


def _init():
    """Initialize this module and apply package-wide config"""
    apply(Path(_PATH_CONFIG_PACKAGE), use_private=True)

    setup_logging()
    seed_rngs()

    logging.getLogger(__name__).info(f"Initialized package config from file: '{_PATH_CONFIG_PACKAGE}'")


def get_attributes(use_private: bool = False) -> dict:
    attributes = {}
    for key, value in globals().items():
        is_attribute = key.isupper()
        is_private = key[0] == "_"

        if is_attribute and (use_private or not is_private):
            attributes[key.lower()] = value

    return attributes


def set_attributes(attributes: dict, use_private: bool = False):
    for key, value in attributes.items():
        key = key.upper() if not use_private else f"_{key.upper()}"

        globals()[key] = value

        logging.getLogger(__name__).info(f"Set config attribute: '{key}' = {value}")


def apply(path: Path, use_private: bool = False):
    if not path.exists():
        message = f"Config file not found: '{path}'"
        logging.getLogger(__name__).error(message)
        raise FileNotFoundError(message)

    attributes = utils_io.load_yaml(path)
    set_attributes(attributes, use_private=use_private)

    logging.getLogger(__name__).info(f"Loaded config from file: '{path}'")


def save(path_dir: Path, use_private: bool = False):
    """Save the current config.
    The relative path within the directory is always 'config.yaml'.
    """
    if not path_dir.exists():
        message = f"Destination directory not found: '{path_dir}'"
        logging.getLogger(__name__).error(message)
        raise FileNotFoundError(message)

    attributes = get_attributes(use_private=use_private)
    path_config = path_dir / "config.yaml"

    utils_io.save_yaml(attributes, path_config)

    logging.getLogger(__name__).info(f"Saved config to file: '{path_config}'")


def dump(use_private: bool = True):
    attributes = get_attributes(use_private=use_private)

    pprint.pprint(attributes)


def setup_logging():
    if not logging.getLogger(__name__).hasHandlers():
        logging.basicConfig(
            level=getattr(logging, _LEVEL_LOGGING_DEFAULT),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )


def seed_rngs():
    random.seed(_SEED_RNGS)
    np.random.seed(_SEED_RNGS)
    torch.manual_seed(_SEED_RNGS)

    logging.getLogger(__name__).info(f"Seeded RNGs with seed: {_SEED_RNGS}")


def apply_preset(name: str):
    """Apply a config from a set of preset configs integrated in the package.
    The config name may be prepended by directory names but does not include the yaml suffix.
    """
    path_config = resources.files(__package__) / "configs" / f"{name}.yaml"

    apply(path_config)


def apply_experiment(path_dir_experiment: Path):
    """Apply a config from an experiment directory.
    The relative path within the directory is always 'config.yaml'.
    """
    path_config = path_dir_experiment / "config.yaml"

    apply(path_config)


def list_available() -> list:
    """List all available preset configs integrated in the package.
    The config names may be prepended by directory names but do not include the yaml suffix.
    """
    path_dir_configs = resources.files(__package__) / "configs"
    paths_config = sorted(path_dir_configs.glob("**/*.yaml"))

    names_available = [str(path_config.parent.relative_to(path_dir_configs) / path_config.stem) for path_config in paths_config]

    return names_available


_init()

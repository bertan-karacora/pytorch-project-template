import json
import logging

import torchvision as tv
import yaml


_LOGGER = logging.getLogger(__name__)
INDENT_TAB = 4


def load_yaml(path):
    try:
        with open(path, "r") as file:
            data = yaml.safe_load(file)
    except yaml.YAMLError as exception:
        _LOGGER.exception(exception)
        raise exception

    _LOGGER.debug(f"Loaded yaml from path: {path}")

    return data


def save_yaml(dict, path):
    class DumperIndent(yaml.Dumper):
        def increase_indent(self, flow=False, indentless=False):
            return super().increase_indent(flow, False)

    try:
        with open(path, "w") as file:
            yaml.dump(dict, file, Dumper=DumperIndent, default_flow_style=False, indent=INDENT_TAB)
    except yaml.YAMLError as exception:
        _LOGGER.exception(exception)

        raise exception

    _LOGGER.debug(f"Saved yaml to path: {path}")


def load_json(path):
    try:
        with open(path, "r") as file:
            data = json.load(file)
    except json.JSONDecodeError as exception:
        _LOGGER.exception(exception)

        raise exception

    _LOGGER.debug(f"Loaded json from path: {path}")

    return data


def load_image(path, mode="unchanged", use_exif_orientation=False):
    mode = getattr(tv.io.ImageReadMode, mode.upper())
    image = tv.io.read_image(path, mode=mode, apply_exif_orientation=use_exif_orientation)

    _LOGGER.debug(f"Loaded image from path: {path}")

    return image

import logging

import torch


_LOGGER = logging.getLogger(__name__)


def get_device(name):
    if name == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            _LOGGER.warning("CUDA not available. Falling back to CPU.")
            device = torch.device("cpu")
    elif name == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            _LOGGER.warning("MPS not available. Falling back to CPU.")
            device = torch.device("cpu")
    elif name == "cpu":
        device = torch.device("cpu")
    else:
        message = f"Unknown device: '{name}'"
        _LOGGER.error(message)
        raise ValueError(message)

    _LOGGER.info(f"Selected device: '{device}'")

    return device

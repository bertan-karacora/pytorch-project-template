import logging
import random

import numpy as np
import torch

import project.config as config
import project.libs.factory as factory

_LOGGER = logging.getLogger(__name__)


def save(trainer, epoch=None, num_epochs=None, name=None):
    path_dir_checkpoints = trainer.path_dir_experiment / "checkpoints"
    if name is not None:
        filename = f"checkpoint_{name}.pth"
    elif epoch is not None:
        filename = f"checkpoint_epoch{f"{epoch:0{len(str(num_epochs))}d}" if num_epochs is not None else epoch}.pth"
    else:
        filename = "checkpoint.pth"
    path_checkpoint = path_dir_checkpoints / filename

    torch.save(
        {
            "epoch": epoch if epoch is not None else None,
            "state_model": trainer.model.state_dict() if trainer.model else None,
            "state_optimizer": trainer.optimizer.state_dict() if trainer.optimizer else None,
            "state_scaler": trainer.scaler.state_dict() if trainer.scaler else None,
            "state_scheduler": trainer.scheduler.state_dict() if trainer.scheduler else None,
            "state_random": {
                "random": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        },
        path_checkpoint,
    )

    _LOGGER.info(f"Saved checkpoint to file: '{path_checkpoint}'")


def load(path):
    checkpoint = torch.load(path)

    epoch = None
    if "epoch" in checkpoint:
        epoch = checkpoint["epoch"]

    model = None
    if "state_model" in checkpoint:
        model = factory.create_model()
        model.load_state_dict(checkpoint["state_model"])

    optimizer = None
    if "state_optimizer" in checkpoint and model is not None:
        optimizer = factory.create_optimizer(model.parameters())
        optimizer.load_state_dict(checkpoint["state_optimizer"])

    scaler = None
    if "state_scaler" in checkpoint:
        scaler = torch.cuda.amp.GradScaler(enabled=config.TRAINING["use_amp"])
        scaler.load_state_dict(checkpoint["state_scaler"])

    scheduler = None
    if "state_scheduler" in checkpoint and optimizer is not None:
        scheduler = factory.create_scheduler(optimizer)
        scheduler.load_state_dict(checkpoint["state_scheduler"])

    if "state_random" in checkpoint:
        if "random" in checkpoint["state_random"] and checkpoint["state_random"]["random"] is not None:
            random.setstate(checkpoint["state_random"]["random"])

        if "numpy" in checkpoint["state_random"] and checkpoint["state_random"]["numpy"] is not None:
            np.random.set_state(checkpoint["state_random"]["numpy"])

        if "torch" in checkpoint["state_random"] and checkpoint["state_random"]["torch"] is not None:
            torch.set_rng_state(checkpoint["state_random"]["torch"])

        if torch.cuda.is_available() and checkpoint["state_random"].get("cuda") is not None:
            torch.cuda.set_rng_state_all(checkpoint["state_random"]["cuda"])

    _LOGGER.info(f"Loaded checkpoint from file: '{path}'")

    return epoch, model, optimizer, scaler, scheduler

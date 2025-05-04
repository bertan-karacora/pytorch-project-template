import torch

import self_supervised_learning_of_depth_and_motion.config as config
import self_supervised_learning_of_depth_and_motion.libs.factory as factory


def save(trainer, epoch, name=None):
    path_dir_checkpoints = trainer.path_dir_exp / "checkpoints"

    filename = f"{name}.pth" if name is not None else f"epoch_{epoch}.pth"
    path_checkpoint = path_dir_checkpoints / filename
    torch.save(
        {
            "epoch": epoch,
            "state_dict_model": trainer.model.state_dict(),
            "state_dict_optimizer": trainer.optimizer.state_dict(),
            "state_dict_scaler": trainer.scaler.state_dict() if trainer.scaler is not None else "",
            "state_dict_scheduler": trainer.scheduler.state_dict() if trainer.scheduler is not None else "",
        },
        path_checkpoint,
    )


def load(path):
    checkpoint = torch.load(path)

    epoch = None
    if "epoch" in checkpoint:
        epoch = checkpoint["epoch"]

    model = None
    if "state_dict_model" in checkpoint:
        model = factory.create_model()
        model.load_state_dict(checkpoint["state_dict_model"])

    optimizer = None
    if "state_dict_optimizer" in checkpoint and model is not None:
        optimizer = factory.create_optimizer(model.parameters())
        optimizer.load_state_dict(checkpoint["state_dict_optimizer"])

    scaler = None
    if "state_dict_scaler" in checkpoint:
        scaler = torch.cuda.amp.GradScaler(enabled=config.TRAINING["use_amp"])
        scaler.load_state_dict(checkpoint["state_dict_scaler"])

    scheduler = None
    if "state_dict_scheduler" in checkpoint and optimizer is not None:
        scheduler = factory.create_scheduler(optimizer)
        scheduler.load_state_dict(checkpoint["state_dict_scheduler"])

    return epoch, model, optimizer, scaler, scheduler


def load_model(path):
    checkpoint = torch.load(path)

    model = factory.create_model()
    model.load_state_dict(checkpoint["state_dict_model"])
    return model

import collections
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torchinfo
from tqdm import tqdm

import self_supervised_learning_of_depth_and_motion.config as config
import self_supervised_learning_of_depth_and_motion.libs.factory as factory
import self_supervised_learning_of_depth_and_motion.libs.utils_checkpoints as utils_checkpoints
import self_supervised_learning_of_depth_and_motion.libs.utils_data as utils_data


class Trainer:
    def __init__(self, name_exp, quiet=False):
        self.criterion = None
        self.dataloader_training = None
        self.dataloader_validation = None
        self.dataset_training = None
        self.dataset_validation = None
        self.device = None
        self.log = None
        self.measurers_training = None
        self.measurers_validation = None
        self.model = None
        self.name_exp = name_exp
        self.optimizer = None
        self.path_dir_exp = None
        self.scaler = None
        self.scheduler = None
        self.quiet = quiet
        self.writer_tensorboard = None

        self._init()

    def __str__(self):
        s = f"""Trainer for experiment {self.name_exp}
    Path: {self.path_dir_exp}
    Dataset (training): {self.dataset_training}
    Dataset (validation): {self.dataset_validation}
    Model: {self.model}
    Criterion: {self.criterion}
    Optimizer: {self.optimizer}
    Scheduler: {self.scheduler}
    Measurers (training): {self.measurers_training}
    Measurers (validation): {self.measurers_validation}"""
        return s

    def _init(self):
        self.path_dir_exp = Path(config._PATH_DIR_EXPS) / self.name_exp
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        path_tensorboard = self.path_dir_exp / "tensorboard" / time.strftime("%Y_%m_%d-%H_%M_%S")
        path_tensorboard.mkdir(parents=True, exist_ok=True)
        self.writer_tensorboard = SummaryWriter(path_tensorboard)

        self.scaler = torch.cuda.amp.GradScaler(enabled=config.TRAINING["use_amp"])
        self.log = {
            "training": {
                "batches": {
                    "epoch": [],
                    "num_samples": [],
                    "learning_rate": [],
                    "loss": [],
                    "metrics": collections.defaultdict(list),
                },
                "epochs": {
                    "learning_rate": [],
                    "loss": [],
                    "metrics": collections.defaultdict(list),
                },
            },
            "validation": {
                "batches": {
                    "epoch": [],
                    "num_samples": [],
                    "learning_rate": [],
                    "loss": [],
                    "metrics": collections.defaultdict(list),
                },
                "epochs": {
                    "learning_rate": [],
                    "loss": [],
                    "metrics": collections.defaultdict(list),
                },
            },
        }
        self.dataset_training, self.dataloader_training = factory.create_dataset_and_dataloader(split="training")
        self.dataset_validation, self.dataloader_validation = factory.create_dataset_and_dataloader(split="validation")
        self.model = factory.create_model()
        self.criterion = factory.create_criterion()
        self.optimizer = factory.create_optimizer(self.model.parameters())
        if "scheduler" in config.TRAINING:
            self.scheduler = factory.create_scheduler(self.optimizer)
        self.measurers_training = factory.create_measurers(split="training")
        self.measurers_validation = factory.create_measurers(split="validation")

        self.print(self)
        try:
            self.print(torchinfo.summary(self.model, [config.MODEL["shape_input"]], verbose=0))
        except Exception as e:
            self.print(e)

    def print(self, s):
        if not self.quiet:
            print(s)

    @torch.no_grad()
    def log_batch(self, pass_loop, iteration, epoch, num_samples, loss, lr, output, targets):
        self.log[pass_loop]["batches"]["epoch"] += [epoch]
        self.log[pass_loop]["batches"]["num_samples"] += [num_samples]
        self.log[pass_loop]["batches"]["loss"] += [loss.item()]
        self.log[pass_loop]["batches"]["learning_rate"] += [lr]

        for measurer in getattr(self, f"measurers_{pass_loop}"):
            name_metric = measurer.name_module if hasattr(measurer, "name_module") else type(measurer).__name__
            metric = measurer(output, targets)
            self.log[pass_loop]["batches"]["metrics"][name_metric] += [metric.item()]
            self.writer_tensorboard.add_scalar(f"{pass_loop}|Batches|{name_metric}", metric.item(), iteration)
        self.writer_tensorboard.add_scalar(f"{pass_loop}|Batches|Loss", loss.item(), iteration)
        self.writer_tensorboard.add_scalar(f"{pass_loop}|Batches|Learning rate", lr, iteration)

    @torch.no_grad()
    def log_epoch(self, pass_loop, epoch, num_samples, num_batches):
        nums_samples = np.asarray(self.log[pass_loop]["batches"]["num_samples"][-num_batches:])

        losses = np.asarray(self.log[pass_loop]["batches"]["loss"][-num_batches:])
        loss_epoch = np.sum(losses * nums_samples) / num_samples
        self.log[pass_loop]["epochs"]["loss"] += [loss_epoch]

        lrs = np.asarray(self.log[pass_loop]["batches"]["learning_rate"][-num_batches:])
        lr_epoch = np.sum(lrs * nums_samples) / num_samples
        self.log[pass_loop]["epochs"]["learning_rate"] += [lr_epoch]

        for name, metrics in self.log[pass_loop]["batches"]["metrics"].items():
            metrics_epoch = np.asarray(metrics[-num_batches:])
            metric_epoch = np.sum(metrics_epoch * nums_samples) / num_samples
            self.log[pass_loop]["epochs"]["metrics"][name] += [metric_epoch]
            self.writer_tensorboard.add_scalar(f"{pass_loop}|Epochs|{name}", metric_epoch, epoch)
        self.writer_tensorboard.add_scalar(f"{pass_loop}|Epochs|Loss", loss_epoch, epoch)
        self.writer_tensorboard.add_scalar(f"{pass_loop}|Epochs|Learning rate", lr_epoch, epoch)

    @torch.no_grad()
    def validate_epoch(self, epoch):
        progress_bar = tqdm(self.dataloader_validation, total=len(self.dataloader_validation), disable=self.quiet)
        for i, (features, targets) in enumerate(progress_bar, start=1):
            features = utils_data.move_items(features, self.device)
            targets = utils_data.move_items(targets, self.device)
            num_samples = utils_data.count_items(targets)

            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.scaler is not None):
                output = self.model(features)
                loss = self.criterion(output, targets)
            self.optimizer.zero_grad()
            self.scaler.scale(loss)

            lr = self.optimizer.param_groups[0]["lr"]
            self.log_batch("validation", len(self.dataloader_validation) * epoch + i, epoch, num_samples, loss, lr, output, targets)
            if i % config.LOGGING["tqdm"]["frequency"] == 0 and not self.quiet:
                progress_bar.set_description(f"Validating: Epoch {epoch:03d} | Batch {i:03d} | LR {lr:.6f} | Loss {loss.item():.5f}")

        self.log_epoch("validation", epoch, len(self.dataset_validation), len(self.dataloader_validation))

    def train_epoch(self, epoch):
        progress_bar = tqdm(self.dataloader_training, total=len(self.dataloader_training), disable=self.quiet)
        for i, (features, targets) in enumerate(progress_bar, start=1):
            features = utils_data.move_items(features, self.device)
            targets = utils_data.move_items(targets, self.device)
            num_samples = utils_data.count_items(targets)

            with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.scaler is not None):
                output = self.model(features)
                loss = self.criterion(output, targets)
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            lr = self.optimizer.param_groups[0]["lr"]
            self.log_batch("training", len(self.dataloader_training) * epoch + i, epoch, num_samples, loss, lr, output, targets)
            if i % config.LOGGING["tqdm"]["frequency"] == 0 and not self.quiet:
                progress_bar.set_description(f"Training: Epoch {epoch:03d} | Batch {i:03d} | LR {lr:.6f} | Loss {loss.item():.5f}")

        self.log_epoch("training", epoch, len(self.dataset_training), len(self.dataloader_training))

    def loop(self, num_epochs, save_checkpoints=True):
        self.print("Looping ...")

        loss_best = float("inf")
        epoch_loss_best = 0
        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)
        for i in range(len(self.measurers_training)):
            self.measurers_training[i] = self.measurers_training[i].to(self.device)
        for i in range(len(self.measurers_validation)):
            self.measurers_validation[i] = self.measurers_validation[i].to(self.device)

        self.validate_epoch(0)
        for epoch in range(1, num_epochs + 1):
            self.model.train()
            self.train_epoch(epoch)

            self.model.eval()
            self.validate_epoch(epoch)

            loss_epoch = self.log["validation"]["epochs"]["loss"][-1]
            if self.scheduler is not None:
                self.scheduler.step()

            if loss_epoch < loss_best:
                loss_best = loss_epoch
                epoch_loss_best = epoch
                if save_checkpoints:
                    utils_checkpoints.save(self, epoch, name="best")

            if save_checkpoints:
                utils_checkpoints.save(self, epoch, name="latest")
                if epoch % config.LOGGING["checkpoint"]["frequency"] == 0 and epoch != 1:
                    utils_checkpoints.save(self, epoch)
                if epoch == num_epochs:
                    utils_checkpoints.save(self, epoch, name="final")

            if "early_stopping" in config.TRAINING and epoch - epoch_loss_best > config.TRAINING["early_stopping"]["patience"]:
                self.print("Looping stopped early")
                break

        self.print("Looping finished")

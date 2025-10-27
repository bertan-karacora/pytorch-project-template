from contextlib import contextmanager
import logging
from pathlib import Path
import time

import torch
from tqdm.auto import tqdm

import project.config as config
import project.libs.factory as factory
import project.libs.utils_checkpoints as utils_checkpoints
import project.libs.utils_data as utils_data
import project.libs.utils_torch as utils_torch
from project.training.log import Log


_LOGGER = logging.getLogger(__name__)


class Trainer:
    def __init__(self, name_experiment):
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
        self.name_experiment = name_experiment
        self.optimizer = None
        self.path_dir_experiment = None
        self.scaler = None
        self.scheduler = None

        self._init()

        _LOGGER.info(f"Initialized trainer for experiment: '{self.name_experiment}'")

    def _init(self):
        self.path_dir_experiment = Path(config._PATH_DIR_EXPS) / self.name_experiment
        self.log = Log(self.path_dir_experiment)
        self.device = utils_torch.get_device(config._DEVICE)
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.TRAINING["use_amp"])
        self.dataset_training, self.dataloader_training = factory.create_dataset_and_dataloader(split="training")
        self.dataset_validation, self.dataloader_validation = factory.create_dataset_and_dataloader(split="validation")
        self.model = factory.create_model()
        self.criterion = factory.create_criterion()
        self.optimizer = factory.create_optimizer(self.model.parameters())
        if "scheduler" in config.TRAINING:
            self.scheduler = factory.create_scheduler(self.optimizer)
        if hasattr(config, "MEASURERS"):
            if "training" in config.MEASURERS:
                self.measurers_training = factory.create_measurers(split="training")
            if "validation" in config.MEASURERS:
                self.measurers_validation = factory.create_measurers(split="validation")

    def loop(self, num_epochs, use_save_checkpoints=True):
        _LOGGER.info("Looping...")

        self.validate_epoch(epoch=0, num_epochs=num_epochs)

        loss_best = float("inf")
        epoch_loss_best = 0
        for epoch in range(1, num_epochs + 1):
            self.train_epoch(epoch, num_epochs=num_epochs)

            self.validate_epoch(epoch, num_epochs=num_epochs)

            if self.scheduler:
                self.scheduler.step()

            loss_epoch = self.log["validation"]["epochs"]["loss"][-1]
            if loss_epoch < loss_best:
                loss_best = loss_epoch
                epoch_loss_best = epoch

            if use_save_checkpoints:
                utils_checkpoints.save(self, epoch, num_epochs=num_epochs, name="latest")
                if epoch % config.LOGGING["checkpoints"]["frequency"] == 1:
                    utils_checkpoints.save(self, epoch, num_epochs=num_epochs)
                if epoch == num_epochs:
                    utils_checkpoints.save(self, epoch, num_epochs=num_epochs, name="final")
                if epoch == epoch_loss_best:
                    utils_checkpoints.save(self, epoch, num_epochs=num_epochs, name="best")

            if "early_stopping" in config.TRAINING and epoch - epoch_loss_best > config.TRAINING["early_stopping"]["patience"]:
                _LOGGER.info("Looping stopped early")
                break

        _LOGGER.info("Looping finished")

    def predict(self, inpt, target):
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.scaler is not None):
            output = self.model(inpt)
            loss = self.criterion(output, target)
        self.optimizer.zero_grad()
        loss_scaled = self.scaler.scale(loss)

        return output, loss, loss_scaled

    def measure(self, measurers, output, target):
        metrics = {}
        for measurer in measurers:
            name_metric = measurer.name_module if hasattr(measurer, "name_module") else type(measurer).__name__
            metric = measurer(output, target)
            metrics[name_metric] = metric

        return metrics

    def to(self, device):
        self.model.to(device)
        self.criterion.to(device)
        for measurer in self.measurers_training:
            measurer.to(device)
        for measurer in self.measurers_validation:
            measurer.to(device)

    def train(self):
        self.model.train()
        self.criterion.train()
        for measurer in self.measurers_training:
            measurer.train()
        for measurer in self.measurers_validation:
            measurer.train()

    def eval(self):
        self.model.eval()
        self.criterion.eval()
        for measurer in self.measurers_training:
            measurer.eval()
        for measurer in self.measurers_validation:
            measurer.eval()

    @contextmanager
    def progress(stage, dataloader, epoch, num_batches=None, num_epochs=None):
        _progress = tqdm(
            iterable=dataloader,
            total=num_batches,
            disable=not _LOGGER.isEnabledFor(logging.INFO),
            desc=f"{stage.capitalize()}: Epoch {f'{epoch:0{len(str(num_epochs))}d}' if num_epochs is not None else epoch}",
            dynamic_ncols=True,
            leave=False,
        )

        def update(iteration_epoch, loss, loss_epoch, duration):
            if iteration_epoch % config.LOGGING["tqdm"]["frequency"] == 1 or iteration_epoch == num_batches:
                _progress.set_postfix(
                    {
                        "Batch": f"{f"{iteration_epoch:0{len(str(num_batches))}d}" if num_batches is not None else iteration_epoch} ",
                        "Duration": f"{duration:.1f}",
                        "Loss (batch)": f"{loss:.5f}",
                        "Loss (epoch)": f"{loss_epoch:.5f}",
                    }
                )
                _LOGGER.info(
                    "".join(
                        [
                            f"{stage.capitalize()}: ",
                            f"Epoch={f"{epoch:0{len(str(num_epochs))}d}" if num_epochs is not None else epoch}, ",
                            f"Batch={f"{iteration_epoch:0{len(str(num_batches))}d}" if num_batches is not None else iteration_epoch}, ",
                            f"Duration={duration:.1f}, ",
                            f"Loss_batch={loss:.5f}, ",
                            f"Loss_epoch={loss_epoch:.5f}",
                        ]
                    )
                )

        try:
            yield _progress, update
        finally:
            _progress.close()

    @torch.no_grad()
    def validate_epoch(self, epoch, num_epochs=None):
        time_start_epoch = time.time()

        self.to(self.device)
        self.eval()

        num_batches = len(self.dataloader_validation)
        count_samples = 0
        loss_total = 0.0
        with progress("validation", self.dataloader_validation, epoch, num_batches, num_epochs) as (progress, update_progress):
            for iteration_epoch, (inpt, target) in enumerate(progress, start=1):
                time_start = time.time()

                inpt = utils_data.move_batch(inpt, self.device)
                target = utils_data.move_batch(target, self.device)

                output, loss, loss_scaled = self.predict(inpt, target)

                metrics = self.measure(self.measurers_validation, output, target)
                iteration = num_batches * epoch + iteration_epoch
                num_samples = utils_data.count_items(target)
                loss = loss.item()
                time_end = time.time()
                duration = time_end - time_start
                self.log.add_batch("validation", iteration, epoch, num_samples, inpt, target, output, loss, metrics, duration)

                count_samples += num_samples
                loss_total += loss * num_samples
                loss_epoch = loss_total / count_samples

                update_progress(iteration_epoch, loss, loss_epoch, duration)

        time_end_epoch = time.time()
        duration_epoch = time_end_epoch - time_start_epoch
        self.log.add_epoch("validation", epoch, len(self.dataset_validation), num_batches, duration_epoch)

    def train_epoch(self, epoch, num_epochs=None):
        time_start_epoch = time.time()

        self.to(self.device)
        self.train()

        num_batches = len(self.dataloader_training)
        count_samples = 0
        loss_total = 0.0
        with progress("training", self.dataloader_training, epoch, num_batches, num_epochs) as (progress, update_progress):
            for iteration_epoch, (inpt, target) in enumerate(progress, start=1):
                time_start = time.time()

                inpt = utils_data.move_batch(inpt, self.device)
                target = utils_data.move_batch(target, self.device)

                output, loss, loss_scaled = self.predict(inpt, target)

                loss_scaled.backward()

                if "norm_gradient_max" in config.TRAINING["norm_gradient_max"]:
                    norm_gradient = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=config.TRAINING["norm_gradient_max"])

                self.scaler.step(self.optimizer)
                self.scaler.update()

                if not "norm_gradient_max" in config.TRAINING["norm_gradient_max"]:
                    norm_gradient = torch.nn.utils.get_total_norm([parameter.grad for parameter in self.model.parameters() if parameter.grad is not None])

                metrics = self.measure(self.measurers_training, output, target)
                iteration = num_batches * epoch + iteration_epoch
                num_samples = utils_data.count_items(target)
                loss = loss.item()
                time_end = time.time()
                duration = time_end - time_start
                learning_rate = self.optimizer.param_groups[0]["lr"]
                self.log.add_batch("training", iteration, epoch, num_samples, inpt, target, output, loss, metrics, duration, learning_rate, norm_gradient)

                count_samples += num_samples
                loss_total += loss * num_samples
                loss_epoch = loss_total / count_samples

                update_progress(iteration_epoch, loss, loss_epoch, duration)

        time_end_epoch = time.time()
        duration_epoch = time_end_epoch - time_start_epoch
        self.log.add_epoch("training", epoch, len(self.dataset_validation), num_batches, duration_epoch)

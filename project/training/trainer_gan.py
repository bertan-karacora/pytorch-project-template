import collections
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torchinfo
import torchvision as tv
from tqdm import tqdm

import self_supervised_learning_of_depth_and_motion.config as config
import self_supervised_learning_of_depth_and_motion.libs.factory as factory
import self_supervised_learning_of_depth_and_motion.libs.utils_checkpoints as utils_checkpoints
import self_supervised_learning_of_depth_and_motion.libs.utils_data as utils_data


class Trainer:
    def __init__(self, name_exp, quiet=False):
        self.criterion_discriminator_fake = None
        self.criterion_discriminator_real = None
        self.criterion_generator = None
        self.dataloader_training = None
        self.dataloader_validation = None
        self.dataset_training = None
        self.dataset_validation = None
        self.device = None
        self.log_discriminator = None
        self.log_generator = None
        self.measurers_training_discriminator = None
        self.measurers_training_generator = None
        self.measurers_validation_discriminator = None
        self.measurers_validation_generator = None
        self.model_discriminator = None
        self.model_generator = None
        self.name_exp = name_exp
        self.optimizer_discriminator = None
        self.optimizer_generator = None
        self.path_dir_exp = None
        self.scheduler_discriminator = None
        self.scheduler_generator = None
        self.quiet = quiet
        self.writer_tensorboard = None

        self._init()

    def __str__(self):
        s = f"""Trainer for experiment {self.name_exp}
    Path: {self.path_dir_exp}
    Dataset (training): {self.dataset_training}
    Dataset (validation): {self.dataset_validation}
    Model (discriminator): {self.model_discriminator}
    Model (generator): {self.model_generator}
    Criterion (discriminator, fake): {self.criterion_discriminator_fake}
    Criterion (discriminator, real): {self.criterion_discriminator_real}
    Criterion (generator): {self.criterion_generator}
    Optimizer (discriminator): {self.optimizer_discriminator}
    Optimizer (generator): {self.optimizer_generator}
    Scheduler (discriminator): {self.scheduler_discriminator}
    Scheduler (generator): {self.scheduler_generator}
    Measurers (discriminator, training): {self.measurers_training_discriminator}
    Measurers (generator, training): {self.measurers_training_generator}
    Measurers (discriminator, validation): {self.measurers_validation_discriminator}
    Measurers (generator, validation): {self.measurers_validation_generator}"""
        return s

    def _init(self):
        self.path_dir_exp = Path(config._PATH_DIR_EXPS) / self.name_exp
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.writer_tensorboard = SummaryWriter(self.path_dir_exp / "tensorboard")
        self.log_discriminator = {
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
        self.log_generator = {
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
        self.model_discriminator = factory.create_model(config.MODEL_DISCRIMINATOR)
        self.model_generator = factory.create_model(config.MODEL_GENERATOR)
        self.criterion_discriminator_fake = factory.create_criterion()
        self.criterion_discriminator_real = factory.create_criterion()
        self.criterion_generator = factory.create_criterion()
        self.optimizer_discriminator = factory.create_optimizer(self.model_discriminator.parameters())
        self.optimizer_generator = factory.create_optimizer(self.model_generator.parameters())
        if "scheduler" in config.TRAINING:
            self.scheduler_discriminator = factory.create_scheduler(self.optimizer_discriminator)
            self.scheduler_generator = factory.create_scheduler(self.optimizer_generator)
        self.measurers_training_discriminator = factory.create_measurers(config.MEASURERS_DISCRIMINATOR["training"])
        self.measurers_training_generator = factory.create_measurers(config.MEASURERS_GENERATOR["training"])
        self.measurers_validation_discriminator = factory.create_measurers(config.MEASURERS_DISCRIMINATOR["validation"])
        self.measurers_validation_generator = factory.create_measurers(config.MEASURERS_GENERATOR["validation"])

        self.print(self)
        try:
            self.print(torchinfo.summary(self.model_discriminator, [config.MODEL_DISCRIMINATOR["shape_input"]], verbose=0))
            self.print(torchinfo.summary(self.model_generator, [config.MODEL_GENERATOR["shape_input"]], verbose=0))
        except Exception as e:
            self.print(e)
            self.print("Failed to run torchinfo")

    def print(self, s):
        if not self.quiet:
            print(s)

    @torch.no_grad()
    def log_batch_discriminator(self, pass_loop, iteration, epoch, num_samples, loss, lr, output, targets):
        self.log_discriminator[pass_loop]["batches"]["epoch"] += [epoch]
        self.log_discriminator[pass_loop]["batches"]["num_samples"] += [num_samples]
        self.log_discriminator[pass_loop]["batches"]["loss"] += [loss.item()]
        self.log_discriminator[pass_loop]["batches"]["learning_rate"] += [lr]

        for measurer in getattr(self, f"measurers_{pass_loop}_discriminator"):
            name_metric = measurer.name_module if hasattr(measurer, "name_module") else type(measurer).__name__
            metric = measurer(output, targets)
            self.log_discriminator[pass_loop]["batches"]["metrics"][name_metric] += [metric.item()]
            self.writer_tensorboard.add_scalar(f"Discriminator|{pass_loop}|Batches|{name_metric}", metric.item(), iteration)
        self.writer_tensorboard.add_scalar(f"Discriminator|{pass_loop}|Batches|Loss", loss.item(), iteration)
        self.writer_tensorboard.add_scalar(f"Discriminator|{pass_loop}|Batches|Learning rate", lr, iteration)

    @torch.no_grad()
    def log_batch_generator(self, pass_loop, iteration, epoch, num_samples, loss, lr, output, targets):
        self.log_generator[pass_loop]["batches"]["epoch"] += [epoch]
        self.log_generator[pass_loop]["batches"]["num_samples"] += [num_samples]
        self.log_generator[pass_loop]["batches"]["loss"] += [loss.item()]
        self.log_generator[pass_loop]["batches"]["learning_rate"] += [lr]

        for measurer in getattr(self, f"measurers_{pass_loop}_generator"):
            name_metric = measurer.name_module if hasattr(measurer, "name_module") else type(measurer).__name__
            metric = measurer(output, targets)
            self.log_generator[pass_loop]["batches"]["metrics"][name_metric] += [metric.item()]
            self.writer_tensorboard.add_scalar(f"Generator|{pass_loop}|Batches|{name_metric}", metric.item(), iteration)
        self.writer_tensorboard.add_scalar(f"Generator|{pass_loop}|Batches|Loss", loss.item(), iteration)
        self.writer_tensorboard.add_scalar(f"Generator|{pass_loop}|Batches|Learning rate", lr, iteration)
        if "frequency_images" in config.LOGGING["tensorboard"] and iteration % config.LOGGING["tensorboard"]["frequency_images"] == 0:
            images = self.model_generator.sample(num_samples=16)
            images = utils_data.unnormalize(images, split=pass_loop)
            grid = tv.utils.make_grid(images, nrow=4)
            self.writer_tensorboard.add_image(f"Generator|{pass_loop}|Images", grid, global_step=iteration)

    @torch.no_grad()
    def log_epoch_discriminator(self, pass_loop, epoch, num_samples, num_batches):
        nums_samples = np.asarray(self.log_discriminator[pass_loop]["batches"]["num_samples"][-num_batches:])

        losses = np.asarray(self.log_discriminator[pass_loop]["batches"]["loss"][-num_batches:])
        loss_epoch = np.sum(losses * nums_samples) / num_samples
        self.log_discriminator[pass_loop]["epochs"]["loss"] += [loss_epoch]

        lrs = np.asarray(self.log_discriminator[pass_loop]["batches"]["learning_rate"][-num_batches:])
        lr_epoch = np.sum(lrs * nums_samples) / num_samples
        self.log_discriminator[pass_loop]["epochs"]["learning_rate"] += [lr_epoch]

        for name, metrics in self.log_discriminator[pass_loop]["batches"]["metrics"].items():
            metrics_epoch = np.asarray(metrics[-num_batches:])
            metric_epoch = np.sum(metrics_epoch * nums_samples) / num_samples
            self.log_discriminator[pass_loop]["epochs"]["metrics"][name] += [metric_epoch]
            self.writer_tensorboard.add_scalar(f"Discriminator|{pass_loop}|Epochs|{name}", metric_epoch, epoch)
        self.writer_tensorboard.add_scalar(f"Discriminator|{pass_loop}|Epochs|Loss", loss_epoch, epoch)
        self.writer_tensorboard.add_scalar(f"Discriminator|{pass_loop}|Epochs|Learning rate", lr_epoch, epoch)

    @torch.no_grad()
    def log_epoch_generator(self, pass_loop, epoch, num_samples, num_batches):
        nums_samples = np.asarray(self.log_generator[pass_loop]["batches"]["num_samples"][-num_batches:])

        losses = np.asarray(self.log_generator[pass_loop]["batches"]["loss"][-num_batches:])
        loss_epoch = np.sum(losses * nums_samples) / num_samples
        self.log_generator[pass_loop]["epochs"]["loss"] += [loss_epoch]

        lrs = np.asarray(self.log_generator[pass_loop]["batches"]["learning_rate"][-num_batches:])
        lr_epoch = np.sum(lrs * nums_samples) / num_samples
        self.log_generator[pass_loop]["epochs"]["learning_rate"] += [lr_epoch]

        for name, metrics in self.log_generator[pass_loop]["batches"]["metrics"].items():
            metrics_epoch = np.asarray(metrics[-num_batches:])
            metric_epoch = np.sum(metrics_epoch * nums_samples) / num_samples
            self.log_generator[pass_loop]["epochs"]["metrics"][name] += [metric_epoch]
            self.writer_tensorboard.add_scalar(f"Generator|{pass_loop}|Epochs|{name}", metric_epoch, epoch)
        self.writer_tensorboard.add_scalar(f"Generator|{pass_loop}|Epochs|Loss", loss_epoch, epoch)
        self.writer_tensorboard.add_scalar(f"Generator|{pass_loop}|Epochs|Learning rate", lr_epoch, epoch)

    @torch.no_grad()
    def validate_epoch(self, epoch):
        progress_bar = tqdm(self.dataloader_validation, total=len(self.dataloader_validation), disable=self.quiet)
        for i, (features, targets) in enumerate(progress_bar, start=1):
            features = features.to(self.device)
            targets = targets.to(self.device)
            latent = self.model_generator.sample_latent(num_samples=features.shape[0])

            targets_real = torch.ones(features.shape[0], device=self.device)
            targets_fake = torch.zeros(features.shape[0], device=self.device)

            output_discriminator_real = self.model_discriminator(features, targets)
            loss_discriminator_real = self.criterion_discriminator_real(output_discriminator_real, targets_real)

            features_fake = self.model_generator(latent, targets)
            output_discriminator_fake = self.model_discriminator(features_fake.detach(), targets)
            loss_discriminator_fake = self.criterion_discriminator_fake(output_discriminator_fake, targets_fake)

            loss_discriminator = loss_discriminator_real + loss_discriminator_fake

            lr = self.optimizer_discriminator.param_groups[0]["lr"]
            output_discriminator = torch.concat((output_discriminator_real, output_discriminator_fake), dim=0)
            targets_discriminator = torch.concat((targets_real, targets_fake), dim=0)
            self.log_batch_discriminator("validation", len(self.dataloader_validation) * epoch + i, epoch, len(features), loss_discriminator, lr, output_discriminator, targets_discriminator)

            output_generator = self.model_discriminator(features_fake, targets)
            loss_generator = self.criterion_generator(output_generator, targets_real.clone())

            lr = self.optimizer_generator.param_groups[0]["lr"]
            self.log_batch_generator("validation", len(self.dataloader_validation) * epoch + i, epoch, len(features), loss_generator, lr, features_fake, features)

            if i % config.LOGGING["tqdm"]["frequency"] == 0 and not self.quiet:
                progress_bar.set_description(f"Validation: Epoch {epoch:03d} | Batch {i:03d} | LR {lr:.6f} | Loss (descriminator) {loss_discriminator.item():.5f} | Loss (generator) {loss_generator.item():.5f}")

        self.log_epoch_discriminator("validation", epoch, len(self.dataset_validation), len(self.dataloader_validation))
        self.log_epoch_generator("validation", epoch, len(self.dataset_validation), len(self.dataloader_validation))

    def train_epoch(self, epoch):
        progress_bar = tqdm(self.dataloader_training, total=len(self.dataloader_training), disable=self.quiet)
        for i, (features, targets) in enumerate(progress_bar, start=1):
            features = features.to(self.device)
            targets = targets.to(self.device)
            latent = self.model_generator.sample_latent(num_samples=features.shape[0])

            targets_real = torch.ones(features.shape[0], device=self.device)
            targets_fake = torch.zeros(features.shape[0], device=self.device)

            output_discriminator_real = self.model_discriminator(features, targets)
            loss_discriminator_real = self.criterion_discriminator_real(output_discriminator_real, targets_real)

            features_fake = self.model_generator(latent, targets)
            output_discriminator_fake = self.model_discriminator(features_fake.detach(), targets)
            loss_discriminator_fake = self.criterion_discriminator_fake(output_discriminator_fake, targets_fake)

            loss_discriminator = loss_discriminator_real + loss_discriminator_fake

            self.optimizer_discriminator.zero_grad()
            loss_discriminator.backward()
            self.optimizer_discriminator.step()

            lr = self.optimizer_discriminator.param_groups[0]["lr"]
            output_discriminator = torch.concat((output_discriminator_real, output_discriminator_fake), dim=0)
            targets_discriminator = torch.concat((targets_real, targets_fake), dim=0)
            self.log_batch_discriminator("training", len(self.dataloader_training) * epoch + i, epoch, len(features), loss_discriminator, lr, output_discriminator, targets_discriminator)

            output_generator = self.model_discriminator(features_fake, targets)
            loss_generator = self.criterion_generator(output_generator, targets_real.clone())

            self.optimizer_generator.zero_grad()
            loss_generator.backward()
            self.optimizer_generator.step()

            lr = self.optimizer_generator.param_groups[0]["lr"]
            self.log_batch_generator("training", len(self.dataloader_training) * epoch + i, epoch, len(features), loss_generator, lr, features_fake, features)

            if i % config.LOGGING["tqdm"]["frequency"] == 0 and not self.quiet:
                progress_bar.set_description(f"Training: Epoch {epoch:03d} | Batch {i:03d} | LR {lr:.6f} | Loss (descriminator) {loss_discriminator.item():.5f} | Loss (generator) {loss_generator.item():.5f}")

        self.log_epoch_discriminator("training", epoch, len(self.dataset_training), len(self.dataloader_training))
        self.log_epoch_generator("training", epoch, len(self.dataset_training), len(self.dataloader_training))

    def loop(self, num_epochs, save_checkpoints=True):
        self.print("Looping ...")

        loss_best = float("inf")

        self.model_discriminator = self.model_discriminator.to(self.device)
        self.model_generator = self.model_generator.to(self.device)
        self.criterion_discriminator_fake = self.criterion_discriminator_fake.to(self.device)
        self.criterion_discriminator_real = self.criterion_discriminator_real.to(self.device)
        self.criterion_generator = self.criterion_generator.to(self.device)
        for i in range(len(self.measurers_training_discriminator)):
            self.measurers_training_discriminator[i] = self.measurers_training_discriminator[i].to(self.device)
        for i in range(len(self.measurers_validation_discriminator)):
            self.measurers_validation_discriminator[i] = self.measurers_validation_discriminator[i].to(self.device)
        for i in range(len(self.measurers_training_generator)):
            self.measurers_training_generator[i] = self.measurers_training_generator[i].to(self.device)
        for i in range(len(self.measurers_validation_generator)):
            self.measurers_validation_generator[i] = self.measurers_validation_generator[i].to(self.device)

        self.validate_epoch(0)
        for epoch in range(1, num_epochs + 1):
            self.model_discriminator.train()
            self.model_generator.train()
            self.train_epoch(epoch)

            self.model_discriminator.eval()
            self.model_generator.eval()
            self.validate_epoch(epoch)

            if self.scheduler_discriminator is not None:
                self.scheduler_discriminator.step()
            if self.scheduler_generator is not None:
                self.scheduler_generator.step()

            loss_epoch = self.log_generator["validation"]["epochs"]["loss"][-1]
            if loss_epoch < loss_best:
                loss_best = loss_epoch
                if save_checkpoints:
                    utils_checkpoints.save(self, epoch, name="best")

            if save_checkpoints:
                utils_checkpoints.save(self, epoch, name="latest")
                if epoch % config.LOGGING["checkpoint"]["frequency"] == 0 and epoch != 1:
                    utils_checkpoints.save(self, epoch)
                if epoch == num_epochs:
                    utils_checkpoints.save(self, epoch, name="final")

        self.print("Looping finished")

import collections
import logging
import time

import numpy as np
from torch.utils.tensorboard import SummaryWriter


_LOGGER = logging.getLogger(__name__)


class Log:
    def __init__(self, path_dir_exp):
        self = None
        self.path_dir_exp = path_dir_exp
        self.path_tensorboard = None
        self.writer_tensorboard = None

        self._init()

    def _init(self):
        self = {
            "training": {
                "batches": {
                    "epoch": [],
                    "num_samples": [],
                    "loss": [],
                    "norm_gradient": [],
                    "learning_rate": [],
                    "metrics": collections.defaultdict(list),
                    "duration": [],
                },
                "epochs": {
                    "learning_rate": [],
                    "loss": [],
                    "norm_gradient": [],
                    "metrics": collections.defaultdict(list),
                    "duration": [],
                },
            },
            "validation": {
                "batches": {
                    "epoch": [],
                    "num_samples": [],
                    "loss": [],
                    "learning_rate": [],
                    "metrics": collections.defaultdict(list),
                    "duration": [],
                },
                "epochs": {
                    "loss": [],
                    "learning_rate": [],
                    "metrics": collections.defaultdict(list),
                    "duration": [],
                },
            },
        }

        self._init_tensorboard()

    def _init_tensorboard(self):
        self.path_tensorboard = self.path_dir_exp / "tensorboard" / time.strftime("%Y_%m_%d-%H_%M_%S")
        self.path_tensorboard.mkdir(parents=True, exist_ok=True)

        self.writer_tensorboard = SummaryWriter(self.path_tensorboard)

    @property
    def log(self):
        return self._log

    def __getitem__(self, key):
        item = self._log[key]
        return item

    def add_batch(self, stage, iteration, epoch, num_samples, inpt, targets, output, loss, metrics, duration, learning_rate=None, norm_gradient=None):
        self[stage]["batches"]["epoch"].append(epoch)
        self[stage]["batches"]["num_samples"].append(num_samples)
        self[stage]["batches"]["loss"].append(loss)
        if norm_gradient is not None:
            self[stage]["batches"]["norm_gradient"].append(norm_gradient)
        if learning_rate is not None:
            self[stage]["batches"]["learning_rate"].append(learning_rate)
        for name_metric, metric in metrics.items():
            self[stage]["batches"]["metrics"][name_metric].append(metric)
        self[stage]["batches"]["duration"].append(duration)

        self.writer_tensorboard.add_scalar(f"{stage}|Batches|Loss", loss, iteration)
        if norm_gradient is not None:
            self.writer_tensorboard.add_scalar(f"{stage}|Batches|Gradient norm", norm_gradient, iteration)
        if learning_rate is not None:
            self.writer_tensorboard.add_scalar(f"{stage}|Batches|Learning rate", learning_rate, iteration)
        for name_metric, metric in metrics.items():
            self.writer_tensorboard.add_scalar(f"{stage}|Batches|{name_metric.capitalize()}", metric, iteration)
        self.writer_tensorboard.add_scalar(f"{stage}|Batches|Duration", duration, iteration)

    def add_epoch(self, stage, epoch, num_samples, num_batches, duration):
        nums_samples_batch = np.asarray(self[stage]["batches"]["num_samples"][-num_batches:])

        losses_batch = np.asarray(self[stage]["batches"]["loss"][-num_batches:])
        loss_epoch = np.sum(losses_batch * nums_samples_batch) / num_samples
        self[stage]["epochs"]["loss"].append(loss_epoch)

        norms_gradient_batch = np.asarray(self[stage]["batches"]["norm_gradient"][-num_batches:])
        norm_gradient_epoch = np.sum(norms_gradient_batch * nums_samples_batch) / num_samples
        self[stage]["epochs"]["norm_gradient"].append(norm_gradient_epoch)

        learning_rates_batch = np.asarray(self[stage]["batches"]["learning_rate"][-num_batches:])
        learning_rate_epoch = np.sum(learning_rates_batch * nums_samples_batch) / num_samples
        self[stage]["epochs"]["learning_rate"].append(learning_rate_epoch)

        for name_metric, metrics in self[stage]["batches"]["metrics"].items():
            metrics_batch = np.asarray(metrics[-num_batches:])
            metric_epoch = np.sum(metrics_batch * nums_samples_batch) / num_samples
            self[stage]["epochs"]["metrics"][name_metric].append(metric_epoch)

        self[stage]["epochs"]["duration"].append(duration)

        self.writer_tensorboard.add_scalar(f"{stage}|Epochs|Loss", loss_epoch, epoch)
        self.writer_tensorboard.add_scalar(f"{stage}|Epochs|Gradient norm", norm_gradient_epoch, epoch)
        self.writer_tensorboard.add_scalar(f"{stage}|Epochs|Learning rate", learning_rate_epoch, epoch)
        for name_metric, metric_epoch in self[stage]["epochs"]["metrics"].items():
            self.writer_tensorboard.add_scalar(f"{stage}|Epochs|{name_metric}", metric_epoch, epoch)

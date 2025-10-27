import collections
import logging
import time

import numpy as np


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
            "test": {
                "batches": {
                    "num_samples": [],
                    "metrics": collections.defaultdict(list),
                    "duration": [],
                },
                "total": {
                    "metrics": {},
                    "duration": 0.0,
                },
            }
        }

    def _init_tensorboard(self):
        self.path_tensorboard = self.path_dir_exp / "tensorboard" / time.strftime("%Y_%m_%d-%H_%M_%S")
        self.path_tensorboard.mkdir(parents=True, exist_ok=True)

    @property
    def log(self):
        return self._log

    def __getitem__(self, key):
        item = self._log[key]
        return item

    def add_batch(self, num_samples, inpt, targets, output, metrics, duration):
        self["test"]["batches"]["num_samples"].append(num_samples)
        for name_metric, metric in metrics.items():
            self["test"]["batches"]["metrics"][name_metric].append(metric)
        self["test"]["batches"]["duration"].append(duration)

    def add_total(self, num_samples, num_batches, duration):
        nums_samples_batch = np.asarray(self["test"]["batches"]["num_samples"][-num_batches:])

        for name_metric, metrics in self["test"]["batches"]["metrics"].items():
            metrics_batch = np.asarray(metrics[-num_batches:])
            metric_total = np.sum(metrics_batch * nums_samples_batch) / num_samples
            self["test"]["total"]["metrics"][name_metric].append(metric_total)

        self["test"]["total"]["duration"].append(duration)

import collections
from pathlib import Path

import numpy as np
import torch
import torchinfo
from tqdm import tqdm

import self_supervised_learning_of_depth_and_motion.config as config
import self_supervised_learning_of_depth_and_motion.libs.factory as factory
import self_supervised_learning_of_depth_and_motion.libs.utils_checkpoints as utils_checkpoints


class Evaluator:
    def __init__(self, name_exp, name_checkpoint="best", quiet=False):
        self.dataloader_test = None
        self.dataset_test = None
        self.device = None
        self.log = None
        self.measurers = None
        self.model = None
        self.name_checkpoint = name_checkpoint
        self.name_exp = name_exp
        self.path_dir_exp = None
        self.quiet = quiet

        self._init()

    def __str__(self):
        s = f"""Evaluator for experiment {self.name_exp}
    Path: {self.path_dir_exp}
    Dataset: {self.dataset_test}
    Model: {self.model}
    Measurers: {self.measurers}"""
        return s

    def _init(self):
        self.path_dir_exp = Path(config._PATH_DIR_EXPS) / self.name_exp
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log = {
            "test": {
                "batches": {
                    "num_samples": [],
                    "metrics": collections.defaultdict(list),
                },
                "total": {
                    "metrics": {},
                },
            }
        }
        self.dataset_test, self.dataloader_test = factory.create_dataset_and_dataloader(split="test")
        self.model = utils_checkpoints.load_model(self.path_dir_exp / "checkpoints" / f"{self.name_checkpoint}.pth")
        self.measurers = factory.create_measurers(split="test")

        self.print(self)
        self.print(torchinfo.summary(self.model, [config.MODEL["shape_input"]], verbose=0))

    def print(self, s):
        if not self.quiet:
            print(s)

    def log_batch(self, num_samples, output, targets):
        self.log["test"]["batches"]["num_samples"] += [num_samples]
        for measurer in self.measurers:
            name_metric = measurer.name_module if hasattr(measurer, "name_module") else type(measurer).__name__
            metric = measurer(output, targets)
            self.log["test"]["batches"]["metrics"][name_metric] += [metric.item()]

    def log_total(self, num_batches, num_samples):
        nums_samples = np.asarray(self.log["test"]["batches"]["num_samples"][-num_batches:])
        for name, metrics in self.log["test"]["batches"]["metrics"].items():
            metrics_total = np.asarray(metrics[-num_batches:])
            metric_total = np.sum(metrics_total * nums_samples) / num_samples
            self.log["test"]["total"]["metrics"][name] = metric_total

    @torch.inference_mode()
    def evaluate(self):
        self.print("Evaluation ...")

        self.model = self.model.to(self.device)
        for i in range(len(self.measurers)):
            self.measurers[i] = self.measurers[i].to(self.device)

        progress_bar = tqdm(self.dataloader_test, total=len(self.dataloader_test), disable=self.quiet)
        for i, (features, targets) in enumerate(progress_bar, start=1):
            features = features.to(self.device)
            targets = targets.to(self.device)

            output = self.model(features)

            self.log_batch(len(targets), output, targets)
            if i % config.LOGGING["tqdm"]["frequency"] == 0 and not self.quiet:
                progress_bar.set_description(f"Validating: Batch {i:03d}")

        self.log_total(num_batches=len(self.dataloader_test), num_samples=len(self.dataset_test))

        self.print("Evaluation finished")

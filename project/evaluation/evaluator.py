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
from project.evaluation.log import Log


_LOGGER = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, name_experiment, name_checkpoint="best"):
        self.dataloader_test = None
        self.dataset_test = None
        self.device = None
        self.log = None
        self.measurers_test = None
        self.model = None
        self.name_checkpoint = name_checkpoint
        self.name_experiment = name_experiment
        self.path_dir_experiment = None

        self._init()

        _LOGGER.info(f"Initialized evaluator for experiment: '{self.name_experiment}'")

    def _init(self):
        self.path_dir_experiment = Path(config._PATH_DIR_EXPS) / self.name_experiment
        self.log = Log(self.path_dir_experiment)
        self.device = utils_torch.get_device(config._DEVICE)
        self.dataset_test, self.dataloader_test = factory.create_dataset_and_dataloader(split="test")
        self.model = utils_checkpoints.load_model(self.path_dir_exp / "checkpoints" / f"{self.name_checkpoint}.pth")
        if hasattr(config, "MEASURERS"):
            if "test" in config.MEASURERS:
                self.measurers = factory.create_measurers(split="test")

    def measure(self, measurers, output, target):
        metrics = {}
        for measurer in measurers:
            name_metric = measurer.name_module if hasattr(measurer, "name_module") else type(measurer).__name__
            metric = measurer(output, target)
            metrics[name_metric] = metric

        return metrics

    def to(self, device):
        self.model.to(device)
        for measurer in self.measurers_test:
            measurer.to(device)

    def eval(self):
        self.model.eval()
        for measurer in self.measurers_test:
            measurer.eval()

    @contextmanager
    def progress(dataloader, num_batches=None):
        _progress = tqdm(
            iterable=dataloader,
            total=num_batches,
            disable=not _LOGGER.isEnabledFor(logging.INFO),
            desc=f"Evaluation: ",
            dynamic_ncols=True,
            leave=False,
        )

        def update(iteration, duration):
            if iteration % config.LOGGING["tqdm"]["frequency"] == 1 or iteration == num_batches:
                _progress.set_postfix(
                    {
                        "Batch": f"{f"{iteration:0{len(str(num_batches))}d}" if num_batches is not None else iteration} ",
                        "Duration": f"{duration:.1f}",
                    }
                )
                _LOGGER.info(
                    "".join(
                        [
                            f"Evaluation: ",
                            f"Batch={f"{iteration:0{len(str(num_batches))}d}" if num_batches is not None else iteration}, ",
                            f"Duration={duration:.1f}, ",
                        ]
                    )
                )

        try:
            yield _progress, update
        finally:
            _progress.close()

    @torch.inference_mode()
    def evaluate(self):
        _LOGGER.info("Evaluating...")

        time_start_total = time.time()

        self.to(self.device)
        self.eval()

        num_batches = len(self.dataloader_test)
        with progress(self.dataloader_test, num_batches) as (progress, update_progress):
            for iteration, (inpt, target) in enumerate(progress, start=1):
                time_start = time.time()

                inpt = utils_data.move_batch(inpt, self.device)
                target = utils_data.move_batch(target, self.device)

                output = self.model(inpt)

                metrics = self.measure(self.measurers_validation, output, target)
                num_samples = utils_data.count_items(target)
                time_end = time.time()
                duration = time_end - time_start
                self.log.add_batch(num_samples, inpt, target, output, metrics, duration)

                update_progress(iteration, duration)

        time_end_total = time.time()
        duration_total = time_end_total - time_start_total
        self.log.add_total(len(self.dataset_test), num_batches, duration_total)

        _LOGGER.info("Evaluating finished")

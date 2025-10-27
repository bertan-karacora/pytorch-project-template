import copy
import logging
from pathlib import Path
import sys

import numpy as np
import optuna

import project.config as config
import project.libs.utils_optuna as utils_optuna
from project.training.trainer import Trainer


_LOGGER = logging.getLogger(__name__)


class OptimizerHyperparams:
    def __init__(self, name_experiment):
        self.name_experiment = name_experiment
        self.path_db = None
        self.path_dir_experiment = None
        self.study = None

        self._init()

        _LOGGER.info(f"Initialized hyperparameter optimizer")

    def _init(self):
        self.path_dir_experiment = Path(config._PATH_DIR_EXPS) / self.name_experiment
        self.path_db = self.path_dir_experiment / "optuna.db"

        self._init_study()

    def _init_study(self, use_load_if_exists=True):
        if not use_load_if_exists:
            self.path_db.unlink(missing_ok=True)
            _LOGGER.warning(f"Removed database with path: '{self.path_db}'")

        self.study = optuna.create_study(
            direction=config.OPTIMIZATION_HYPERPARAMS["direction"],
            study_name="study",
            storage=f"sqlite:///{self.path_db}",
            load_if_exists=use_load_if_exists,
        )

        if not use_load_if_exists:
            _LOGGER.info(f"Created database with path: '{self.path_db}'")

        _LOGGER.info(f"Loaded database from path: '{self.path_db}'")

    def optimize(self, num_epochs, num_trials):
        _LOGGER.info(f"{"Trials":<10}: {num_trials}")
        _LOGGER.info(f"{"Epochs":<10}: {num_epochs}")

        def objective(trial, num_epochs):
            try:
                config.apply_experiment(self.path_dir_experiment)
                config_trial = copy.deepcopy(config.get_attributes())
                config_trial["training"]["num_epochs"] = num_epochs
                config_trial["training"]["frequency_log"] = sys.maxsize
                params_to_optimize = config.OPTIMIZATION_HYPERPARAMS["params_to_optimize"]
                config_trial = utils_optuna.suggest_values(trial, config_trial, params_to_optimize)
                config.set_attributes(config_trial)

                trainer = Trainer(self.name_experiment)
                trainer.loop(num_epochs, save_checkpoints=False)
            except Exception as e:
                _LOGGER.exception(e)
                raise optuna.TrialPruned()

            config.apply_experiment(self.path_dir_experiment)

            func = np.min if config.OPTIMIZATION_HYPERPARAMS["direction"] == "minimize" else np.max
            metric = func(trainer.log["validation"]["epochs"]["metrics"][config.OPTIMIZATION_HYPERPARAMS["metric"]])

            return metric

        func_obj = lambda trial: objective(trial, num_epochs=num_epochs)

        level = _LOGGER.getEffectiveLevel()
        _LOGGER.setLevel(logging.WARNING)
        self.study.optimize(func_obj, callbacks=[optuna.study.MaxTrialsCallback(num_trials, states=(optuna.trial.TrialState.COMPLETE,))])
        _LOGGER.setLevel(level)

        utils_optuna.print_results(self.study)

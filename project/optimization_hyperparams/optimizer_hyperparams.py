import copy
from pathlib import Path
import sys

import numpy as np
import optuna

import self_supervised_learning_of_depth_and_motion.config as config
import self_supervised_learning_of_depth_and_motion.libs.utils_optuna as utils_optuna
from self_supervised_learning_of_depth_and_motion.training.trainer import Trainer


class OptimizerHyperparams:
    def __init__(self, name_exp):
        self.name_exp = name_exp
        self.path_db = None
        self.path_dir_exp = None
        self.study = None

        self._init()

    def _init(self):
        self.path_dir_exp = Path(config._PATH_DIR_EXPS) / self.name_exp
        self.path_db = self.path_dir_exp / "optuna.db"

    def create_study(self, load_if_exists=False):
        print(f"Creating study...")

        if load_if_exists is False:
            self.path_db.unlink(missing_ok=True)

        self.study = optuna.create_study(
            direction=config.OPTIMIZATION_HYPERPARAMS["direction"],
            study_name="study",
            storage=f"sqlite:///{self.path_db}",
            load_if_exists=load_if_exists if load_if_exists is not None else True,
        )

        print(f"Creating study finished")


    def optimize(self, num_epochs=10, num_trials=25):
        print(f"Optimizing...")
        print(f"    {"Trials":<10}: {num_trials}")
        print(f"    {"Epochs":<10}: {num_epochs}")

        def objective(trial, num_epochs):
            try:
                config.set_config_exp(self.path_dir_exp, quiet=True)
                params_to_optimize = config.OPTIMIZATION_HYPERPARAMS["params_to_optimize"]

                config_trial = copy.deepcopy(config.get_attributes())
                config_trial["training"]["num_epochs"] = num_epochs
                config_trial["training"]["frequency_log"] = sys.maxsize
                config_trial = utils_optuna.suggest_values(trial, config_trial, params_to_optimize)
                config.set_attributes(config_trial)

                trainer = Trainer(self.name_exp, quiet=True)
                trainer.loop(num_epochs, save_checkpoints=False)
            except Exception as e:
                raise optuna.TrialPruned()

            func = np.min if config.OPTIMIZATION_HYPERPARAMS["direction"] == "minimize" else np.max
            metric = func(trainer.log["validation"]["epochs"]["metrics"][config.OPTIMIZATION_HYPERPARAMS["metric"]])

            config.set_config_exp(self.path_dir_exp, quiet=True)

            return metric

        func_obj = lambda trial: objective(trial, num_epochs=num_epochs)
        self.study.optimize(func_obj, callbacks=[optuna.study.MaxTrialsCallback(num_trials, states=(optuna.trial.TrialState.COMPLETE,))])

        utils_optuna.print_results(self.study)

        print(f"Optimizing finished")

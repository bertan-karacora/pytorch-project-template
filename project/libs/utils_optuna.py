import functools
import logging
import operator

import optuna


_LOGGER = logging.getLogger(__name__)


def load_study(path_db, name_study="study"):
    study = optuna.load_study(study_name=name_study, storage=f"sqlite:///{str(path_db)}")

    _LOGGER.info(f"Loaded optuna study from path: {path_db}")

    return study


def print_results(study):
    _LOGGER.info("Study results")
    _LOGGER.info(f"    {"Trials finished":<10}: {len(study.trials)}")
    _LOGGER.info(f"    {"Trials completed":<10}: {len(study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE]))}")
    _LOGGER.info("Best trial")
    _LOGGER.info(f"    {"Number":<10}: {study.best_trial.number}")
    _LOGGER.info(f"    {"Value":<10}: {study.best_trial.value}")
    _LOGGER.info(f"    {"Params":<10}: {study.best_trial.params}")


def suggest_values(trial, config_trial, params_to_optimize):
    for param_to_optimize in params_to_optimize:
        # Ugly workaround to optimize lists or dicts
        if param_to_optimize["type"] in ["list", "dict"]:
            suggestion_str = trial.suggest_categorical(param_to_optimize["name"], [str(choice) for choice in param_to_optimize["kwargs"]["choices"]])
            suggestion = eval(suggestion_str)
        else:
            name_func = f"suggest_{param_to_optimize["type"]}"
            if not hasattr(trial, name_func):
                message = f"Trial has no attribute named {name_func}"
                _LOGGER.error(message)
                raise AttributeError(message)
            func_suggest = getattr(trial, f"suggest_{param_to_optimize["type"]}")
            suggestion = func_suggest(param_to_optimize["name"], **param_to_optimize["kwargs"])

        path_in_config = param_to_optimize["path_in_config"]
        functools.reduce(operator.getitem, path_in_config[:-1], config_trial)[path_in_config[-1]] = suggestion

    return config_trial

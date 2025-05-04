import functools
import operator

import optuna

def load_study(path_db, name_study="study"):
    study = optuna.load_study(study_name=name_study, storage=f"sqlite:///{str(path_db)}")
    return study

def print_results(study):
    print("Study results")
    print(f"    {"Trials finished":<10}: {len(study.trials)}")
    print(f"    {"Trials completed":<10}: {len(study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE]))}")
    print("Best trial")
    print(f"    {"Number":<10}: {study.best_trial.number}")
    print(f"    {"Value":<10}: {study.best_trial.value}")
    print(f"    {"Params":<10}: {study.best_trial.params}")

def suggest_values(trial, config_trial, params_to_optimize):
    for param_to_optimize in params_to_optimize:
        # Ugly workaround to optimize lists or dicts
        if param_to_optimize["type"] in ["list", "dict"]:
            suggestion_str = trial.suggest_categorical(param_to_optimize["name"], [str(choice) for choice in param_to_optimize["kwargs"]["choices"]])
            suggestion = eval(suggestion_str)
        else:
            name_func = f"suggest_{param_to_optimize["type"]}"
            if not hasattr(trial, name_func):
                raise AttributeError(f"Trial has no attribute named {name_func}")
            func_suggest = getattr(trial, f"suggest_{param_to_optimize["type"]}")
            suggestion = func_suggest(param_to_optimize["name"], **param_to_optimize["kwargs"])

        path_in_config = param_to_optimize["path_in_config"]
        functools.reduce(operator.getitem, path_in_config[:-1], config_trial)[path_in_config[-1]] = suggestion

    return config_trial

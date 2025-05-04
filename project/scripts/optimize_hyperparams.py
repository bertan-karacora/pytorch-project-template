import argparse
from pathlib import Path

import self_supervised_learning_of_depth_and_motion.config as config
from self_supervised_learning_of_depth_and_motion.optimization_hyperparams.optimizer_hyperparams import OptimizerHyperparams


def parse_args():
    parser = argparse.ArgumentParser(description="Optimize hyperparameters.")
    parser.add_argument("--name", help="Name of the experiment", required=True)
    parser.add_argument("--num_epochs", help="Number of epochs of each trial", type=int, default=10)
    parser.add_argument("--num_trials", help="Number of trials to execute", type=int, default=25)
    parser.add_argument("--load_if_exists", help="Load study if it exists", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    return args.name, args.config, args.num_epochs, args.num_trials, args.load_if_exists


def optimize_hyperparams(name_exp, num_epochs=10, num_trials=25, load_if_exists=False):
    print(f"Optimizing hyperparameters in {name_exp}...")

    optimizer_hyperparams = OptimizerHyperparams(name_exp, num_epochs, num_trials)
    optimizer_hyperparams.create_study(load_if_exists=load_if_exists)
    optimizer_hyperparams.optimize()

    print(f"Optimizing hyperparameters in {name_exp} finished")


def main():
    name_exp, num_epochs, num_trials, load_if_exists = parse_args()

    path_dir_exp = Path(config._PATH_DIR_EXPS) / name_exp
    config.set_config_exp(path_dir_exp)

    optimize_hyperparams(name_exp, num_epochs, num_trials, load_if_exists)


if __name__ == "__main__":
    main()

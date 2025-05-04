import argparse
from pathlib import Path

import self_supervised_learning_of_depth_and_motion.config as config
from self_supervised_learning_of_depth_and_motion.training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train model.")
    parser.add_argument("--name", help="Name of the experiment", required=True)
    args = parser.parse_args()

    return args.name


def train(name_exp):
    print("Training...")

    trainer = Trainer(name_exp)
    trainer.loop(config.TRAINING["num_epochs"])

    print("Training finished...")


def main():
    name_exp = parse_args()

    path_dir_exp = Path(config._PATH_DIR_EXPS) / name_exp
    config.set_config_exp(path_dir_exp)

    train(name_exp)


if __name__ == "__main__":
    main()

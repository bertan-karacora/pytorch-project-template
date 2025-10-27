import argparse
from pathlib import Path
import shutil
import time

import project.config as config

logging.getLogger().setLevel(logging.INFO)


def parse_args():
    configs_available = config.list_available()

    parser = argparse.ArgumentParser(description="Initialize_experiment.")
    parser.add_argument("--name", help="Name to give to the experiment", default=time.strftime("%Y_%m_%d-%H_%M_%S"))
    parser.add_argument("--config", help=f"Config name selected from {configs_available}", choices=configs_available, required=True)
    parser.add_argument("--use_remove_old", help=f"Remove old directory if it exists", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    return args.name, args.config


def create_dirs_exp(path, use_remove_old=False):
    dirs = [
        "checkpoints",
        "logs",
        "tensorboard",
        "plots",
        "visualizations",
    ]

    path.mkdir(parents=True, exist_ok=True)
    print(f"Created directory {path}")

    for dir in dirs:
        path_dir = path / dir
        if use_remove_old:
            shutil.rmtree(path_dir, ignore_errors=True)
        path_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created directory {path_dir}")


def init_exp(name_exp, name_config, use_remove_old):
    print(f"Initializing experiment {name_exp}...")

    path_dir_exp = Path(config._PATH_DIR_EXPS) / name_exp
    create_dirs_exp(path_dir_exp, use_remove_old)

    config.set_config_preset(name_config)
    config.save(path_dir_exp)

    print(f"Initializing experiment {name_exp} finished")


def main():
    name_exp, name_config, use_remove_old = parse_args()
    init_exp(name_exp, name_config, use_remove_old)


if __name__ == "__main__":
    main()

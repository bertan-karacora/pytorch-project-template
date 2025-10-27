import logging

import argparse
from pathlib import Path

import project.config as config
from project.training.trainer import Trainer


_LOGGER = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train model.")
    parser.add_argument("--name", help="Name of the experiment", required=True)
    parser.add_argument("--level_logging", help="Logging level", choices=["debug", "info", "warning", "error", "fatal"], default="info")
    parser.add_argument("--use_save_checkpoints", help="Logging level", choices=["debug", "info", "warning", "error", "fatal"], default="info")
    args = parser.parse_args()

    return args.name, args.level_logging, args.use_save_checkpoints


# TODO
# def configure_logging(path_dir_experiment, level="info"):
#     level = getattr(logging, level.upper())
#     _LOGGER.setLevel(level)

#     formatter = None
#     for h in _LOGGER.handlers:
#         if h.formatter:
#             formatter = h.formatter
#             break

#     for h in _LOGGER.handlers[:]:
#         if isinstance(h, logging.StreamHandler):
#             _LOGGER.removeHandler(h)

#     tqdm_handler = TqdmLoggingHandler(sys.stdout)
#     if formatter:
#         tqdm_handler.setFormatter(formatter)
#     _LOGGER.addHandler(tqdm_handler)

#     path_log = path_dir_experiment / "log.txt"
#     handler_file = logging.FileHandler(path_log, mode="w")
#     handler_file.setFormatter(formatter)
#     _LOGGER.addHandler(handler_file)


def train(name_experiment, use_save_checkpoints=True):
    _LOGGER.info(f"Training...")

    trainer = Trainer(name_experiment)
    trainer.loop(config.TRAINING["num_epochs"], use_save_checkpoints)

    _LOGGER.info("Training finished")


def main():
    name_experiment, level_logging, use_save_checkpoints = parse_args()

    path_dir_experiment = Path(config._PATH_DIR_EXPERIMENTS) / name_experiment
    config.apply_experiment(path_dir_experiment)

    train(name_experiment, use_save_checkpoints)


if __name__ == "__main__":
    main()

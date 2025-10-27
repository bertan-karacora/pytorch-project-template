import argparse

from project.evaluation.evaluator import Evaluator

logging.getLogger().setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="Optimize hyperparameters.")
    parser.add_argument("--name", help="Name to give to the experiment", required=True)
    args = parser.parse_args()

    return args.name


def evaluate():
    print("Evaluating")

    # TODO: change interface of evaluator to load best checkpoint
    # evaluator = Evaluator(name_exp)
    # evaluator.evaluate()

    print("Evaluating finished")


def main():
    name_exp, num_epochs, num_trials = parse_args()
    evaluate(name_exp, num_epochs, num_trials)


if __name__ == "__main__":
    main()

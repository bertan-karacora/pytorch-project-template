# Pytorch Project Template

A Python package that serves as a template for deep learning projects using Pytorch.

Note: The core of this package is the training, evaluation, hyperparameter optimization, the utilities, logging, visualization, and configuration, all of which have a high potential to be re-used in different projects. The models, datasets, transforms, losses, metrics have just been collected here in whatever state I previously left them in and will change for every project anyway. They just serve as backup of code snippets and inspiration for new code. Also, I have reworked some parts of this repository without testing in the end, so there are probably some bugs.

This template provides a powerful framework that allows for a quick and easy setup of experiments. The idea is to minimize the overhead. Optimally, you only need to do the following steps:

1. Adjust requirements to be installed in a containerized environment. You can use environment variables set in `env.sh` to adjust most components of the basic container image (versions of CUDA, CudNN, Python, etc.). The python environment can be controlled using the `requirements.txt` file. Alternatively, you may use the `setup.sh` script to setup your local environment but I really recommend using a container.
2. Configure the package using the `project/config.yaml` file. Configure the paths for your data and logging destination and set other package-wide settings like an RNG seed.
3. Add your dataset class, your model implementation, losses, metrics, and custom data transforms. You may use any pre-implemented modules from torch, torchvision, and other libraries by specifying them in your experiment config file.
4. Configure an experiment in the config directory using your dataset, model, etc., as well as all the hyperparameters. All the code in the template is highly dynamic and there is not a lot that cannot be adjusted directly in the experiment config.
5. Run a command in the container. You may start a jupyter server via the `scripts/start_jupyter.sh` script or use the python scripts in the package, e.g., `train.py`

You can find some examples of how earlier versions of this template have been used in my repository["lab-vision-systems-assignments"](https://github.com/bertan-karacora/lab-vision-systems-assignments). In the jupyter notebooks, there are also some nice plots and visualizations.

## Setup

```bash
git clone https://github.com/bertan-karacora/pytorch-project-template.git
cd pytorch-project-template
```

## Installation

You can configure your setup using `env.sh`, specifying the desired versions of Ubuntu, CUDA, CudNN, Python, Pip, Setuptools and Wheel.

### Local

Alternatively, you can setup your local system directly if you have sudo rights.

```bash
./setup.sh
```

```bash
rm -rf *.egg-info
pip install -e .
```

### Build container

```bash
container/build.sh
```

## Usage

### Run in container

```bash
container/run.sh
```

You may provide any command including arguments directly, e.g.:

```bash
container/run.sh echo "test" && scripts/start_jupyter.sh
```

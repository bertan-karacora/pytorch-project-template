import colorsys
import hashlib
import logging

import matplotlib.pyplot as plt
import numpy as np

import project.config as config


_LOGGER = logging.getLogger(__name__)
FIGSIZE_A4 = np.array([210, 297]) / 25.4
FIGSIZE_A4_LANDSCAPE = FIGSIZE_A4[::-1]


def subplot_simple(title, label_x, label_y, list_series, use_legend=True, use_log_x=False, use_log_y=False):
    """Create a subplot with given settings and plot values."""
    ax = plt.gca()

    ax.set_title(title, fontsize=config._SIZE_FONT)
    ax.set_xlabel(label_x, fontsize=config._SIZE_FONT)
    ax.set_ylabel(label_y, fontsize=config._SIZE_FONT)
    ax.tick_params(axis="both", which="major", labelsize=config._SIZE_LABEL_TICKS_MAJOR)
    ax.tick_params(axis="both", which="minor", labelsize=config._SIZE_LABEL_TICKS_MINOR)
    ax.grid(alpha=config._ALPHA_GRID)
    if use_log_x:
        ax.set_xscale("log")
    if use_log_y:
        ax.set_yscale("log")

    for series in list_series:
        ax.plot(series["values_x"], series["values_y"], label=series["label"], **series.get("style", {}))

    if use_legend:
        ax.legend(fontsize=config._SIZE_FONT)


def plot_simple(list_kwargs_subplot, use_landscape=False, num_rows=1, num_cols=1, path_save=None):
    """Create a figure with multiple subplots."""
    figsize = FIGSIZE_A4 if not use_landscape else FIGSIZE_A4_LANDSCAPE
    fig = plt.figure(figsize=figsize)

    for i, kwargs_subplot in enumerate(list_kwargs_subplot):
        fig.add_subplot(num_rows, num_cols, i + 1)

        if kwargs_subplot is None:
            continue

        subplot_simple(**kwargs_subplot)

    plt.tight_layout()

    if path_save:
        plt.savefig(path_save)
        _LOGGER.info(f"Plot saved to path: {path_save}")

    plt.show()


def create_epochs_batch_continuous(epochs_batch, nums_samples_batch):
    """Convert array of epoch number per iteration to a continuous epoch scale."""
    epochs_unique, idxs_start_batch = np.unique(epochs_batch, return_index=True)
    idxs = np.r_[idxs_start_batch, len(epochs_batch)]
    nums_samples_epoch = np.array([np.sum(nums_samples_batch[idx_start:idx_end]) for idx_start, idx_end in zip(idxs[:-1], idxs[1:])], dtype=int)

    offsets_epoch = {epoch: 0 for epoch in epochs_unique}
    offsets_batch = np.zeros_like(nums_samples_batch)
    for i, (epoch, num_samples) in enumerate(zip(epochs_batch, nums_samples_batch)):
        offsets_batch[i] = offsets_epoch[epoch]
        offsets_epoch[epoch] += num_samples

    epochs_batch_continuous = (epochs_batch - 1) + offsets_batch.astype(float) / nums_samples_epoch[epochs_batch - 1].astype(float)

    return epochs_batch_continuous


def smooth(f, k=5):
    """Smoothing a function using a low-pass filter (mean) of size K"""
    kernel = np.ones(k) / k
    f = np.concatenate([f[: int(k // 2)], f, f[int(-k // 2) :]])
    smooth_f = np.convolve(f, kernel, mode="same")
    smooth_f = smooth_f[k // 2 : -k // 2]
    return smooth_f


def id_to_color(id, saturation=0.75, value=0.95):
    """Convert an identifier to a RGB color using hashing."""
    hash_object = hashlib.sha256(str(id).encode())
    hash_digest = hash_object.hexdigest()

    hue = int(hash_digest[:16], 16) % 360 / 360.0
    rgb = colorsys.hsv_to_rgb(hue, saturation, value)

    return rgb


def get_dimensions(images, figsize):
    """Estimate a balanced grid layout (rows, cols) for displaying images.

    The layout is chosen heuristically to roughly match the aspect ratio
    of the figure while accounting for the aspect ratio of the images.
    This helps minimize whitespace and distortion when plotting.

    Assumes all images have the same shape.
    """
    # Assume all images have the same shape
    image_first = images[0]
    aspect_images = image_first.shape[1] / image_first.shape[0]
    aspect_figure = figsize[1] / figsize[0]

    num_images = len(images)
    num_cols = min(max(int(round(np.sqrt(num_images * aspect_figure / aspect_images))), 1), num_images)
    num_rows = max(int(np.ceil(num_images / num_cols)), 1)

    return num_rows, num_cols

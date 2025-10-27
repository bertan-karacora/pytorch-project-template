import logging

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch

import project.config as config
import project.libs.utils_data as utils_data
import project.libs.utils_visualization as utils_visualization


_LOGGER = logging.getLogger(__name__)


def visualize_images(images, labels=None, indices=None, path_save=None):
    fig = plt.figure(figsize=utils_visualization.FIGSIZE_A4)

    def subplot_image(image, i):
        ax = plt.gca()

        if labels is not None or indices is not None:
            title = ""
            if indices is not None:
                title += rf"#${indices[i]}$"
            if indices is not None and labels is not None:
                title += " | "
            if labels is not None:
                title += rf"{labels[i]}"
            ax.set_title(title, fontsize=config._SIZE_FONT)
        ax.set_axis_off()

        image_vis = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
        ax.imshow(image_vis, cmap="gray" if image.shape[-1] == 1 else None)

    images = images.numpy().transpose((0, 2, 3, 1))

    num_rows, num_cols = utils_visualization.get_dimensions(images, figsize=fig.get_size_inches())
    for i, image in enumerate(images):
        fig.add_subplot(num_rows, num_cols, i + 1)
        subplot_image(image, i)

    plt.tight_layout()

    if path_save:
        plt.savefig(path_save)
        _LOGGER.info(f"Plot saved to path: {path_save}")

    plt.show()


def visualize_images_pairs(images1, images2, labels=None, indices=None, path_save=None):
    din_a4 = np.array([210, 297]) / 25.4
    fig = plt.figure(figsize=din_a4)

    def subplot_image(image, i):
        ax = plt.gca()

        if labels is not None or indices is not None:
            title = ""
            if indices is not None:
                title += rf"#${indices[i]}$"
            if indices is not None and labels is not None:
                title += " | "
            if labels is not None:
                title += rf"label: {labels[i]}"
            ax.set_title(title, fontsize=config._SIZE_FONT)
        ax.set_axis_off()

        image_vis = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
        ax.imshow(image_vis, cmap="gray" if image.shape[-1] == 1 else None)

    images = torch.concat((images1, images2), dim=-1)
    images = images.numpy().transpose((0, 2, 3, 1))

    num_rows, num_cols = utils_visualization.get_dimensions(images, figsize=fig.get_size_inches())
    for i, image in enumerate(images):
        fig.add_subplot(num_rows, num_cols, i + 1)
        subplot_image(image, i)

    plt.tight_layout()

    if path_save:
        plt.savefig(path_save)
        _LOGGER.info(f"Plot saved to path: {path_save}")

    plt.show()


def visualize_kernels(kernels, channel=0):
    fig = plt.figure(figsize=utils_visualization.FIGSIZE_A4)

    def subfigure_kernels(subfig, name, kernels_single):
        subfig.suptitle(name)

        def subplot_kernel(kernel):
            ax = plt.gca()
            ax.set_axis_off()
            kernel -= torch.min(kernel)
            kernel /= torch.max(kernel)
            ax.imshow(kernel, cmap="gray")

        # Assume same shape for all images
        aspect_images = kernels_single[0].shape[1] / kernels_single[0].shape[0]
        figsize = fig.get_size_inches()
        aspect_figure = len(kernels) * figsize[0] / figsize[1]

        num_subplots = len(kernels_single)
        num_cols = max(int(np.sqrt(num_subplots * aspect_figure / aspect_images)), 1)
        num_rows = np.ceil(num_subplots / num_cols).astype(int)
        for j, kernel in enumerate(kernels_single):
            subfig.add_subplot(num_rows, num_cols, j + 1)
            subplot_kernel(kernel)

    gs = gridspec.GridSpec(len(kernels), 1)
    for i, (name, kernels_single) in enumerate(kernels.items()):
        subfig = fig.add_subfigure(gs[i, :])
        subfigure_kernels(subfig, name, torch.squeeze(kernels_single[:, channel, :, :]))

    plt.show()


def visualize_featuremaps(featuremaps):
    fig = plt.figure(figsize=utils_visualization.FIGSIZE_A4)

    def subfigure_featuremaps(subfig, name, featuremaps_single):
        subfig.suptitle(name)

        def subplot_featuremap(featuremap):
            ax = plt.gca()
            ax.set_axis_off()
            ax.imshow(featuremap)

        # Assume same shape for all images
        aspect_images = featuremaps_single[0].shape[1] / featuremaps_single[0].shape[0]
        figsize = fig.get_size_inches()
        aspect_figure = len(featuremaps) * figsize[0] / figsize[1]

        num_subplots = len(featuremaps_single)
        num_cols = max(int(np.sqrt(num_subplots * aspect_figure / aspect_images)), 1)
        num_rows = np.ceil(num_subplots / num_cols).astype(int)
        for j, featuremap in enumerate(featuremaps_single):
            subfig.add_subplot(num_rows, num_cols, j + 1)
            subplot_featuremap(featuremap)

    gs = gridspec.GridSpec(len(featuremaps), 1)
    for i, (name, featuremaps_single) in enumerate(featuremaps.items()):
        subfig = fig.add_subfigure(gs[i, :])
        subfigure_featuremaps(subfig, name, torch.squeeze(featuremaps_single))

    plt.show()


@torch.no_grad()
def visualize_interpolation_grid(model, shape_image, num_channels_latent, label_input, xrange=(-2, 2), yrange=(-2, 2), use_denormalize=False, resolution=12, path_save=None):
    fig = plt.figure(figsize=utils_visualization.FIGSIZE_A4)

    def subplot_interpolation_grid():
        ax = plt.gca()

        ax.set_title(f"Equispaced points from latent space (2D projection)", fontsize=config._SIZE_FONT)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        grid = np.empty((3, resolution * shape_image[0], resolution * shape_image[1]))
        for i, y in enumerate(np.linspace(*yrange, resolution)):
            for j, x in enumerate(np.linspace(*xrange, resolution)):
                embedding = torch.zeros(num_channels_latent, device=device)[None, ...]
                embedding[:, : embedding.shape[1] // 2] = x
                embedding[:, embedding.shape[1] // 2 :] = y

                output = model.decode(embedding, label_input).cpu()

                if use_denormalize:
                    output = utils_data.denormalize(output, split="test")

                grid[:, (resolution - 1 - i) * shape_image[0] : (resolution - i) * shape_image[0], j * shape_image[1] : (j + 1) * shape_image[1]] = output

        ax.imshow(grid.transpose((1, 2, 0)), extent=[*yrange, *xrange])
        ax.axis("off")

    fig.add_subplot(2, 1, 1)
    subplot_interpolation_grid()

    plt.tight_layout()

    if path_save:
        plt.savefig(path_save)
        _LOGGER.info(f"Plot saved to path: {path_save}")

    plt.show()

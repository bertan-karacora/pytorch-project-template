import logging

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sbn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import project.config as config
import project.libs.utils_visualization as utils_visualization


_LOGGER = logging.getLogger(__name__)


def plot_loss(log, path_save=None):
    fig = plt.figure(figsize=utils_visualization.FIGSIZE_A4)

    def subplot_loss(use_log_y=False):
        ax = plt.gca()

        ax.set_title(f"Training progress{" (log scale)" if use_log_y else ""}", fontsize=config._SIZE_FONT)
        ax.set_xlabel("Epoch", fontsize=config._SIZE_FONT)
        ax.set_ylabel("Loss", fontsize=config._SIZE_FONT)
        ax.tick_params(axis="both", which="major", labelsize=config._SIZE_LABEL_TICKS_MAJOR)
        ax.tick_params(axis="both", which="minor", labelsize=config._SIZE_LABEL_TICKS_MINOR)
        ax.grid(alpha=config._ALPHA_GRID)
        if use_log_y:
            ax.set_yscale("log")

        epochs_batch_training = np.asarray(log["training"]["batches"]["epoch"])
        nums_samples_batch_training = np.asarray(log["training"]["batches"]["num_samples"])
        epochs_batch_training_continuous = utils_visualization.create_epochs_batch_continuous(epochs_batch_training, nums_samples_batch_training)
        losses_batch_training = np.asarray(log["training"]["batches"]["loss"])
        ax.plot(
            epochs_batch_training_continuous,
            losses_batch_training,
            alpha=0.5,
        )

        num_iterations_epoch = len(epochs_batch_training[epochs_batch_training == epochs_batch_training[0]])
        losses_batch_training_smoothed = utils_visualization.smooth(losses_batch_training, k=int(0.5 * num_iterations_epoch))
        ax.plot(
            epochs_batch_training_continuous,
            losses_batch_training_smoothed,
            label="Loss (training)",
        )

        epochs_validation = np.arange(1, len(losses_epochs_validation) + 1)
        losses_epochs_validation = np.asarray(log["validation"]["epochs"]["loss"])
        epoch_validation_min_loss = np.argmin(losses_epochs_validation)
        loss_epoch_validation_min = losses_epochs_validation[epoch_validation_min_loss]
        ax.plot(
            epochs_validation,
            losses_epochs_validation,
            label=f"Loss (validation) [Min: {loss_epoch_validation_min:.3f} @ {epoch_validation_min_loss}]",
        )

        ax.legend(fontsize=config._SIZE_FONT)

    fig.add_subplot(3, 1, 1)
    subplot_loss()

    fig.add_subplot(3, 1, 2)
    subplot_loss(use_log_y=True)

    plt.tight_layout()

    if path_save:
        plt.savefig(path_save)
        _LOGGER.info(f"Plot saved to path: {path_save}")

    plt.show()


def plot_norm_gradient(log, path_save=None):
    fig = plt.figure(figsize=utils_visualization.FIGSIZE_A4)

    def subplot_norm_gradient(use_log_y=False):
        ax = plt.gca()

        ax.set_title(f"Gradient norm{" (log scale)" if use_log_y else ""}", fontsize=config._SIZE_FONT)
        ax.set_xlabel("Epoch", fontsize=config._SIZE_FONT)
        ax.set_ylabel("Gradient norm", fontsize=config._SIZE_FONT)
        ax.tick_params(axis="both", which="major", labelsize=config._SIZE_LABEL_TICKS_MAJOR)
        ax.tick_params(axis="both", which="minor", labelsize=config._SIZE_LABEL_TICKS_MINOR)
        ax.grid(alpha=config._ALPHA_GRID)
        if use_log_y:
            ax.set_yscale("log")

        epochs_batch_training = np.asarray(log["training"]["batches"]["epoch"])
        nums_samples_batch_training = np.asarray(log["training"]["batches"]["num_samples"])
        epochs_batch_training_continuous = utils_visualization.create_epochs_batch_continuous(epochs_batch_training, nums_samples_batch_training)
        norms_gradient_batch_training = np.asarray(log["training"]["batches"]["metrics"]["norm_gradient"])
        ax.plot(
            epochs_batch_training_continuous,
            norms_gradient_batch_training,
            alpha=0.5,
        )

        num_iterations_epoch = len(epochs_batch_training[epochs_batch_training == epochs_batch_training[0]])
        norms_gradient_batch_training_smoothed = utils_visualization.smooth(norms_gradient_batch_training, k=int(0.5 * num_iterations_epoch))
        ax.plot(
            epochs_batch_training_continuous,
            norms_gradient_batch_training_smoothed,
            label="Gradient norm",
        )

        ax.legend(fontsize=config._SIZE_FONT)

    fig.add_subplot(3, 1, 1)
    subplot_norm_gradient()

    fig.add_subplot(3, 1, 2)
    subplot_norm_gradient(use_log_y=True)

    plt.tight_layout()

    if path_save:
        plt.savefig(path_save)
        _LOGGER.info(f"Plot saved to path: {path_save}")

    plt.show()


def plot_learning_rate(log, path_save=None):
    fig = plt.figure(figsize=utils_visualization.FIGSIZE_A4)

    def subplot_learning_rate(use_log_y=False):
        ax = plt.gca()

        ax.set_title(f"Learning rate{" (log scale)" if use_log_y else ""}", fontsize=config._SIZE_FONT)
        ax.set_xlabel("Epoch", fontsize=config._SIZE_FONT)
        ax.set_ylabel("Learning rate", fontsize=config._SIZE_FONT)
        ax.tick_params(axis="both", which="major", labelsize=config._SIZE_LABEL_TICKS_MAJOR)
        ax.tick_params(axis="both", which="minor", labelsize=config._SIZE_LABEL_TICKS_MINOR)
        ax.grid(alpha=config._ALPHA_GRID)
        if use_log_y:
            ax.set_yscale("log")

        epochs_batch_training = np.asarray(log["training"]["batches"]["epoch"])
        nums_samples_batch_training = np.asarray(log["training"]["batches"]["num_samples"])
        epochs_batch_training_continuous = utils_visualization.create_epochs_batch_continuous(epochs_batch_training, nums_samples_batch_training)
        learning_rates_batch = np.asarray(log["training"]["batches"]["learning_rate"])
        ax.plot(epochs_batch_training_continuous, learning_rates_batch, label="Learning rate")

        ax.legend(fontsize=config._SIZE_FONT)

    fig.add_subplot(3, 1, 1)
    subplot_learning_rate()

    fig.add_subplot(3, 1, 2)
    subplot_learning_rate(use_log_y=True)

    plt.tight_layout()

    if path_save:
        plt.savefig(path_save)
        _LOGGER.info(f"Plot saved to path: {path_save}")

    plt.show()


def plot_metric(log, name_metric, path_save=None):
    fig = plt.figure(figsize=utils_visualization.FIGSIZE_A4)

    def subplot_metric(use_log_y=False):
        ax = plt.gca()

        ax.set_title(f"Training progress{" (log scale)" if use_log_y else ""}", fontsize=config._SIZE_FONT)
        ax.set_xlabel("Epoch", fontsize=config._SIZE_FONT)
        ax.set_ylabel("Metric", fontsize=config._SIZE_FONT)
        ax.tick_params(axis="both", which="major", labelsize=config._SIZE_LABEL_TICKS_MAJOR)
        ax.tick_params(axis="both", which="minor", labelsize=config._SIZE_LABEL_TICKS_MINOR)
        ax.grid(alpha=config._ALPHA_GRID)
        if use_log_y:
            ax.set_yscale("log")

        if name_metric in log["training"]["batches"]["metrics"]:
            epochs_batch_training = np.asarray(log["training"]["batches"]["epoch"])
            nums_samples_batch_training = np.asarray(log["training"]["batches"]["num_samples"])
            epochs_batch_training_continuous = utils_visualization.create_epochs_batch_continuous(epochs_batch_training, nums_samples_batch_training)
            metrics_batch_training = np.asarray(log["training"]["batches"]["metrics"][name_metric])
            ax.plot(
                epochs_batch_training_continuous,
                metrics_batch_training,
                alpha=0.5,
            )

            num_iterations_epoch = len(epochs_batch_training[epochs_batch_training == epochs_batch_training[0]])
            metrics_batch_training_smoothed = utils_visualization.smooth(metrics_batch_training, k=int(0.5 * num_iterations_epoch))
            ax.plot(
                epochs_batch_training_continuous,
                metrics_batch_training_smoothed,
                label=f"{name_metric.capitalize()} (training)",
            )
        else:
            _LOGGER.warning(f"Failed to find metric in training log: '{name_metric}'")
            return

        if name_metric in log["validation"]["batches"]["metrics"]:
            epochs_validation = np.arange(1, len(metrics_epochs_validation) + 1)
            metrics_epochs_validation = np.asarray(log["validation"]["epochs"]["metrics"][name_metric])
            epoch_validation_min_metric = np.argmin(metrics_epochs_validation)
            metric_epoch_validation_min = metrics_epochs_validation[epoch_validation_min_metric]
            epoch_validation_max_metric = np.argmax(metrics_epochs_validation)
            metric_epoch_validation_max = metrics_epochs_validation[epoch_validation_max_metric]
            ax.plot(
                epochs_validation,
                metrics_epochs_validation,
                label=f"{name_metric.capitalize()} (validation) [Min: {metric_epoch_validation_min:.3f} @ {epoch_validation_min_metric} | Max: {metric_epoch_validation_max:.3f} @ {epoch_validation_max_metric}]",
            )
        else:
            _LOGGER.warning(f"Failed to find metric in validation log: '{name_metric}'")
            return

        ax.legend(fontsize=config._SIZE_FONT)

    fig.add_subplot(3, 1, 1)
    subplot_metric()

    fig.add_subplot(3, 1, 2)
    subplot_metric(use_log_y=True)

    plt.tight_layout()

    if path_save:
        plt.savefig(path_save)
        _LOGGER.info(f"Plot saved to path: {path_save}")

    plt.show()


def plot_metrics(log, path_dir_save=None):
    for name_metric in log["validation"]["batches"]["metrics"].keys():
        path_save = path_dir_save / f"{name_metric.capitalize()}.png" if path_dir_save else None
        plot_metric(log, name_metric, path_save=path_save)


def plot_metrics_all(log, path_save=None):
    fig = plt.figure(figsize=utils_visualization.FIGSIZE_A4)

    def subplot_metrics(use_log_y=False):
        ax = plt.gca()

        ax.set_title(f"Training progress{" (log scale)" if use_log_y else ""}", fontsize=config._SIZE_FONT)
        ax.set_xlabel("Epoch", fontsize=config._SIZE_FONT)
        ax.set_ylabel("Metric", fontsize=config._SIZE_FONT)
        ax.tick_params(axis="both", which="major", labelsize=config._SIZE_LABEL_TICKS_MAJOR)
        ax.tick_params(axis="both", which="minor", labelsize=config._SIZE_LABEL_TICKS_MINOR)
        ax.grid(alpha=config._ALPHA_GRID)
        if use_log_y:
            ax.set_yscale("log")

        epochs_batch_training = np.asarray(log["training"]["batches"]["epoch"])
        nums_samples_batch_training = np.asarray(log["training"]["batches"]["num_samples"])
        epochs_batch_training_continuous = utils_visualization.create_epochs_batch_continuous(epochs_batch_training, nums_samples_batch_training)

        for name_metric, metrics_batch_training in log["training"]["batches"]["metrics"].items():
            metrics_batch_training = np.asarray(metrics_batch_training)
            ax.plot(
                epochs_batch_training_continuous,
                metrics_batch_training,
                alpha=0.5,
            )

            num_iterations_epoch = len(epochs_batch_training[epochs_batch_training == epochs_batch_training[0]])
            metrics_batch_training_smoothed = utils_visualization.smooth(metrics_batch_training, k=int(0.5 * num_iterations_epoch))
            ax.plot(
                epochs_batch_training_continuous,
                metrics_batch_training_smoothed,
                label=f"{name_metric.capitalize()} (training)",
            )

        for name_metric, metrics_epochs_validation in log["validation"]["epochs"]["metrics"].items():
            epochs_validation = np.arange(1, len(metrics_epochs_validation) + 1)
            metrics_epochs_validation = np.asarray(log["validation"]["epochs"]["metrics"][name_metric])
            epoch_validation_min_metric = np.argmin(metrics_epochs_validation)
            metric_epoch_validation_min = metrics_epochs_validation[epoch_validation_min_metric]
            epoch_validation_max_metric = np.argmax(metrics_epochs_validation)
            metric_epoch_validation_max = metrics_epochs_validation[epoch_validation_max_metric]
            ax.plot(
                epochs_validation,
                metrics_epochs_validation,
                label=f"{name_metric.capitalize()} (validation) [Min: {metric_epoch_validation_min:.3f} @ {epoch_validation_min_metric} | Max: {metric_epoch_validation_max:.3f} @ {epoch_validation_max_metric}]",
            )

        ax.legend(fontsize=config._SIZE_FONT)

    fig.add_subplot(3, 1, 1)
    subplot_metrics()

    fig.add_subplot(3, 1, 2)
    subplot_metrics(use_log_y=True)

    plt.tight_layout()

    if path_save:
        plt.savefig(path_save)
        _LOGGER.info(f"Plot saved to path: {path_save}")

    plt.show()


def plot_confusion(confusion, labelset, path_save=None):
    fig = plt.figure(figsize=utils_visualization.FIGSIZE_DIN_A4)

    def subplot_confusion():
        ax = plt.gca()

        ax.set_title("Confusion matrix", fontsize=config._SIZE_FONT)
        ax.set_xlabel("Target", fontsize=config._SIZE_FONT)
        ax.set_ylabel("Prediction", fontsize=config._SIZE_FONT)
        ax.tick_params(bottom=False, left=False)

        confusion_normalized = confusion / np.sum(confusion, axis=1)
        df_confusion = pd.DataFrame(confusion_normalized, index=labelset, columns=labelset)
        sbn.heatmap(df_confusion, annot=True)

    fig.add_subplot(2, 1, 1)
    subplot_confusion()

    plt.tight_layout()

    if path_save:
        plt.savefig(path_save)
        _LOGGER.info(f"Plot saved to path: {path_save}")

    plt.show()


def plot_projection_pca(embeddings, targets=None, classes=None, images_annotation=None, num_samples=2000, use_sampling_first=False, zoom_annotation=1.0, path_save=None):
    fig = plt.figure(figsize=utils_visualization.FIGSIZE_A4_LANDSCAPE)

    def subplot_projection_pca(embeddings_projected):
        ax = plt.gca()

        ax.set_title(f"Projection via PCA", fontsize=config._SIZE_FONT)
        ax.set_xlabel(r"$c_1$", fontsize=config._SIZE_FONT)
        ax.set_ylabel(r"$c_2$", fontsize=config._SIZE_FONT)
        ax.tick_params(axis="both", which="major", labelsize=config._SIZE_LABEL_TICKS_MAJOR)
        ax.tick_params(axis="both", which="minor", labelsize=config._SIZE_LABEL_TICKS_MINOR)
        ax.grid(alpha=config._ALPHA_GRID)

        if targets:
            targets_unique = np.unique(targets)
            for target in targets_unique:
                embeddings_projected_label = embeddings_projected[targets == target]
                ax.scatter(embeddings_projected_label[:, 0], embeddings_projected_label[:, 1], label=classes[target])
        else:
            ax.scatter(embeddings_projected[:, 0], embeddings_projected[:, 1])

        if targets and images_annotation:
            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            for embedding_projected, target, image_annotation in enumerate(zip(embeddings_projected, targets, images_annotation)):
                imagebox = OffsetImage(image_annotation.transpose((1, 2, 0)), zoom=zoom_annotation)
                imagebox.image.axes = ax

                ab = AnnotationBbox(
                    imagebox,
                    [embedding_projected[0], embedding_projected[1]],
                    xybox=(0, 0),
                    xycoords="data",
                    boxcoords="offset points",
                    pad=0.1,
                    bboxprops=dict(edgecolor=colors[int(np.where(target == targets_unique)[0])], lw=2),
                )
                ax.add_artist(ab)

        ax.legend(fontsize=config._SIZE_FONT)

    if use_sampling_first:
        idxs = np.random.choice(np.arange(len(embeddings)), size=num_samples, replace=False)
        embeddings = embeddings[idxs]
        if targets:
            targets = targets[idxs]
        if images_annotation:
            images_annotation = images_annotation[idxs]

    embeddings_projected = PCA(n_components=2).fit_transform(embeddings)

    if not use_sampling_first:
        idxs = np.random.choice(np.arange(len(embeddings_projected)), size=num_samples, replace=False)
        embeddings_projected = embeddings_projected[idxs]
        if targets:
            targets = targets[idxs]
        if images_annotation:
            images_annotation = images_annotation[idxs]

    fig.add_subplot(1, 1, 1)
    subplot_projection_pca(embeddings_projected)

    plt.tight_layout()

    if path_save:
        plt.savefig(path_save)
        _LOGGER.info(f"Plot saved to path: {path_save}")

    plt.show()


def plot_projection_tsne(embeddings, targets=None, classes=None, images_annotation=None, num_samples=2000, use_sampling_first=True, zoom_annotation=1.0, path_save=None):
    fig = plt.figure(figsize=utils_visualization.FIGSIZE_A4_LANDSCAPE)

    def subplot_projection_tsne(embeddings_projected):
        ax = plt.gca()

        ax.set_title(f"Projection via T-SNE", fontsize=config._SIZE_FONT)
        ax.set_xlabel(r"$t_1$", fontsize=config._SIZE_FONT)
        ax.set_ylabel(r"$t_2$", fontsize=config._SIZE_FONT)
        ax.tick_params(axis="both", which="major", labelsize=config._SIZE_LABEL_TICKS_MAJOR)
        ax.tick_params(axis="both", which="minor", labelsize=config._SIZE_LABEL_TICKS_MINOR)
        ax.grid(alpha=config._ALPHA_GRID)

        if targets:
            targets_unique = np.unique(targets)
            for target in targets_unique:
                embeddings_projected_label = embeddings_projected[targets == target]
                ax.scatter(embeddings_projected_label[:, 0], embeddings_projected_label[:, 1], label=classes[target])
        else:
            ax.scatter(embeddings_projected[:, 0], embeddings_projected[:, 1])

        if targets and images_annotation:
            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            for embedding_projected, target, image_annotation in enumerate(zip(embeddings_projected, targets, images_annotation)):
                imagebox = OffsetImage(image_annotation.transpose((1, 2, 0)), zoom=zoom_annotation)
                imagebox.image.axes = ax

                ab = AnnotationBbox(
                    imagebox,
                    [embedding_projected[0], embedding_projected[1]],
                    xybox=(0, 0),
                    xycoords="data",
                    boxcoords="offset points",
                    pad=0.1,
                    bboxprops=dict(edgecolor=colors[int(np.where(target == targets_unique)[0])], lw=2),
                )
                ax.add_artist(ab)

        ax.legend(fontsize=config._SIZE_FONT)

    if use_sampling_first:
        idxs = np.random.choice(np.arange(len(embeddings)), size=num_samples, replace=False)
        embeddings = embeddings[idxs]
        if targets:
            targets = targets[idxs]
        if images_annotation:
            images_annotation = images_annotation[idxs]

    embeddings_projected = TSNE(n_components=2).fit_transform(embeddings)

    if not use_sampling_first:
        idxs = np.random.choice(np.arange(len(embeddings_projected)), size=num_samples, replace=False)
        embeddings_projected = embeddings_projected[idxs]
        if targets:
            targets = targets[idxs]
        if images_annotation:
            images_annotation = images_annotation[idxs]

    fig.add_subplot(1, 1, 1)
    subplot_projection_tsne(embeddings_projected)

    plt.tight_layout()

    if path_save:
        plt.savefig(path_save)
        _LOGGER.info(f"Plot saved to path: {path_save}")

    plt.show()

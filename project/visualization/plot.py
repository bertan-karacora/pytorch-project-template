import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sbn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch

import self_supervised_learning_of_depth_and_motion.libs.utils_visualization as utils_visualization


def plot_loss(log, path_save=None):
    din_a4 = np.array([210, 297]) / 25.4
    fig = plt.figure(figsize=din_a4)

    def subplot_loss(use_logscale=False):
        ax = plt.gca()

        ax.set_title(f"Training progress ({'logscale' if use_logscale else 'linearscale'})", fontsize=9)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("Loss", fontsize=9)
        ax.tick_params(axis="both", which="major", labelsize=9)
        ax.tick_params(axis="both", which="minor", labelsize=8)
        ax.grid(alpha=0.4)
        if use_logscale:
            ax.set_yscale("log")

        epochs_training = np.asarray(log["training"]["batches"]["epoch"])
        nums_samples_training = np.asarray(log["training"]["batches"]["num_samples"])
        epochs_continuous = utils_visualization.create_epochs_continuous(epochs_training, nums_samples_training)

        loss_training = np.asarray(log["training"]["batches"]["loss"])

        num_iterations_per_epoch = len(epochs_training[epochs_training == epochs_training[0]])
        losses_training_smoothed = utils_visualization.smooth(loss_training, k=int(0.5 * num_iterations_per_epoch))

        ax.plot(epochs_continuous, loss_training, alpha=0.5)
        ax.plot(epochs_continuous, losses_training_smoothed, label="Loss (training)")
        loss_validation = log["validation"]["epochs"]["loss"]
        ax.plot(loss_validation, label=f"Loss (validation) [Min: {np.min(loss_validation):.3f} @ {np.argmin(loss_validation)}]")

        ax.legend(fontsize=9)

    fig.add_subplot(3, 1, 1)
    subplot_loss()

    fig.add_subplot(3, 1, 2)
    subplot_loss(use_logscale=True)

    plt.tight_layout()
    if path_save:
        plt.savefig(path_save)
    plt.show()


def plot_metrics_all(log, path_save=None):
    din_a4 = np.array([210, 297]) / 25.4
    fig = plt.figure(figsize=din_a4)

    def subplot_metrics(use_logscale=False):
        ax = plt.gca()

        ax.set_title(f"Training progress ({'logscale' if use_logscale else 'linearscale'})", fontsize=9)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("Metric", fontsize=9)
        ax.tick_params(axis="both", which="major", labelsize=9)
        ax.tick_params(axis="both", which="minor", labelsize=8)
        ax.grid(alpha=0.4)
        if use_logscale:
            ax.set_yscale("log")

        epochs_training = np.asarray(log["training"]["batches"]["epoch"])
        nums_samples_training = np.asarray(log["training"]["batches"]["num_samples"])
        epochs_continuous = utils_visualization.create_epochs_continuous(epochs_training, nums_samples_training)

        for name, metrics in log["training"]["batches"]["metrics"].items():
            metrics_training = np.asarray(metrics)

            num_iterations_per_epoch = len(epochs_training[epochs_training == epochs_training[0]])
            metrics_training_smoothed = utils_visualization.smooth(metrics_training, k=int(0.5 * num_iterations_per_epoch))

            ax.plot(epochs_continuous, metrics_training, alpha=0.5)
            ax.plot(epochs_continuous, metrics_training_smoothed, label=f"{name.capitalize()} (training)")

        for name, metrics in log["validation"]["epochs"]["metrics"].items():
            metrics_validation = np.asarray(metrics)
            ax.plot(
                metrics_validation,
                label=f"{name.capitalize()} (validation) [Min: {np.min(metrics_validation):.3f} @ {np.argmin(metrics_validation)} | Max: {np.max(metrics_validation):.3f} @ {np.argmax(metrics_validation)}]",
            )

        ax.legend(fontsize=9)

    fig.add_subplot(3, 1, 1)
    subplot_metrics()

    fig.add_subplot(3, 1, 2)
    subplot_metrics(use_logscale=True)

    plt.tight_layout()
    if path_save:
        plt.savefig(path_save)
    plt.show()


def plot_metric(log, name_metric, path_save=None):
    din_a4 = np.array([210, 297]) / 25.4
    fig = plt.figure(figsize=din_a4)

    def subplot_metric(use_logscale=False):
        ax = plt.gca()

        ax.set_title(f"Training progress ({'logscale' if use_logscale else 'linearscale'})", fontsize=9)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("Metric", fontsize=9)
        ax.tick_params(axis="both", which="major", labelsize=9)
        ax.tick_params(axis="both", which="minor", labelsize=8)
        ax.grid(alpha=0.4)
        if use_logscale:
            ax.set_yscale("log")

        epochs_training = np.asarray(log["training"]["batches"]["epoch"])
        nums_samples_training = np.asarray(log["training"]["batches"]["num_samples"])
        epochs_continuous = utils_visualization.create_epochs_continuous(epochs_training, nums_samples_training)

        if name_metric in log["training"]["batches"]["metrics"]:
            metrics = log["training"]["batches"]["metrics"][name_metric]
            metrics_training = np.asarray(metrics)

            num_iterations_per_epoch = len(epochs_training[epochs_training == epochs_training[0]])
            metrics_training_smoothed = utils_visualization.smooth(metrics_training, k=int(0.5 * num_iterations_per_epoch))

            ax.plot(epochs_continuous, metrics_training, alpha=0.5)
            ax.plot(epochs_continuous, metrics_training_smoothed, label=f"{name_metric.capitalize()} (training)")

        if name_metric in log["validation"]["batches"]["metrics"]:
            metrics = log["validation"]["epochs"]["metrics"][name_metric]
            metrics_validation = np.asarray(metrics)
            ax.plot(
                metrics_validation,
                label=f"{name_metric.capitalize()} (validation) [Min: {np.min(metrics_validation):.3f} @ {np.argmin(metrics_validation)} | Max: {np.max(metrics_validation):.3f} @ {np.argmax(metrics_validation)}]",
            )

        ax.legend(fontsize=9)

    fig.add_subplot(3, 1, 1)
    subplot_metric()

    fig.add_subplot(3, 1, 2)
    subplot_metric(use_logscale=True)

    plt.tight_layout()
    if path_save:
        plt.savefig(path_save)
    plt.show()


def plot_metrics(log, path_plots=None, suffix=None):
    for name_metric in log["validation"]["batches"]["metrics"].keys():
        path_save = path_plots / f"Metrics_{name_metric}{f"_{suffix}" if suffix else ""}.png"
        plot_metric(log, name_metric, path_save=path_save)


def plot_learning_rate(log, path_save=None):
    din_a4 = np.array([210, 297]) / 25.4
    fig = plt.figure(figsize=din_a4)

    def subplot_learning_rate():
        ax = plt.gca()

        ax.set_title(f"Learning rate during training", fontsize=9)
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("Learning rate", fontsize=9)
        ax.tick_params(axis="both", which="major", labelsize=9)
        ax.tick_params(axis="both", which="minor", labelsize=8)
        ax.grid(alpha=0.4)

        epochs_training = np.asarray(log["training"]["batches"]["epoch"])
        nums_samples_training = np.asarray(log["training"]["batches"]["num_samples"])
        epochs_continuous = utils_visualization.create_epochs_continuous(epochs_training, nums_samples_training)

        learning_rate_training = np.asarray(log["training"]["batches"]["learning_rate"])
        ax.plot(epochs_continuous, learning_rate_training, label="Learning rate (training)")

        ax.legend(fontsize=9)

    fig.add_subplot(3, 1, 1)
    subplot_learning_rate()

    plt.tight_layout()
    if path_save:
        plt.savefig(path_save)
    plt.show()


def plot_confusion(confusion, labelset):
    din_a4 = np.array([210, 297]) / 25.4
    fig = plt.figure(figsize=din_a4)

    def subplot_confusion():
        ax = plt.gca()

        confusion_normalized = confusion / np.sum(confusion, axis=1)
        df_confusion = pd.DataFrame(confusion_normalized, index=labelset, columns=labelset)

        sbn.heatmap(df_confusion, annot=True)

        ax.set_title("Confusion matrix", fontsize=9)
        ax.set_xlabel("Target", fontsize=9)
        ax.set_ylabel("Prediction", fontsize=9)
        ax.tick_params(bottom=False, left=False)

    fig.add_subplot(3, 1, 1)
    subplot_confusion()

    plt.tight_layout()
    plt.show()


# def plot_gradient_stats(iterations_training, stats, num_samples_training):
#     din_a4 = np.array([210, 297]) / 25.4
#     fig = plt.figure(figsize=din_a4)

#     def subplot_gradient_stat(name, stat):
#         ax = plt.gca()

#         ax.set_title(f"Gradient stat: {name.capitalize()}", fontsize=9)
#         ax.set_xlabel("Epoch", fontsize=9)
#         ax.set_ylabel(f"{name.capitalize()}", fontsize=9)
#         ax.tick_params(axis="both", which="major", labelsize=9)
#         ax.tick_params(axis="both", which="minor", labelsize=8)
#         ax.grid(alpha=0.4)

#         epochs_iterations_training = iterations2epochs(iterations_training, BATCHSIZE, num_samples_training)
#         for name_parameter, parameter in stat.items():
#             ax.plot(epochs_iterations_training, parameter, alpha=0.6, label=f"{name_parameter}")

#         ax.legend(fontsize=9)

#     for i, (name, stat) in enumerate(stats.items()):
#         fig.add_subplot(len(stats), 1, i + 1)
#         subplot_gradient_stat(name, stat)

#     plt.tight_layout()
#     plt.show()


def plot_projection_pca(features_flat, num_points=2000, targets=None, labelset=None, images_annotation=None, zoom_annotation=1.0, path_save=None):
    din_a4 = np.array([210, 297]) / 25.4
    din_a4_landscape = din_a4[::-1]
    fig = plt.figure(figsize=din_a4_landscape)

    def subplot_projection(points, targets):
        ax = plt.gca()

        ax.set_title(f"Projection via PCA", fontsize=9)
        ax.set_xlabel(r"$p_1$", fontsize=9)
        ax.set_ylabel(r"$p_2$", fontsize=9)
        ax.tick_params(axis="both", which="major", labelsize=9)
        ax.tick_params(axis="both", which="minor", labelsize=8)
        ax.grid(alpha=0.4)

        if targets is not None:
            labelset_targets = np.unique(targets)

            for label in labelset_targets:
                points_target = points[targets==label]
                ax.scatter(points_target[:, 0], points_target[:, 1], label=labelset[label])
        else:
            ax.scatter(points[:, 0], points[:, 1])

        if images_annotation is not None and targets is not None:
            from matplotlib.offsetbox import OffsetImage, AnnotationBbox

            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            for i, point in enumerate(points):
                imagebox = OffsetImage(images_annotation[i].transpose((1, 2, 0)), zoom=zoom_annotation)
                imagebox.image.axes = ax
                
                ab = AnnotationBbox(
                    imagebox,
                    [point[0], point[1]],
                    xybox=(0, 0),
                    xycoords="data",
                    boxcoords="offset points",
                    pad=0.1,
                    bboxprops=dict(edgecolor=colors[int(np.where(labelset_targets == targets[i])[0])], lw=2),
                )
                ax.add_artist(ab)

        ax.legend(fontsize=9)

    points = PCA(n_components=2).fit_transform(features_flat)
    points = points[:num_points]
    targets = targets[:num_points]

    fig.add_subplot(1, 1, 1)
    subplot_projection(points, targets)

    plt.tight_layout()
    if path_save:
        plt.savefig(path_save)
    plt.show()


def plot_projection_tsne(features_flat, num_points=2000, targets=None, labelset=None, images_annotation=None, zoom_annotation=1.0, path_save=None):
    din_a4 = np.array([210, 297]) / 25.4
    din_a4_landscape = din_a4[::-1]
    fig = plt.figure(figsize=din_a4_landscape)

    def subplot_projection(points, targets):
        ax = plt.gca()

        ax.set_title(f"Projection via T-SNE", fontsize=9)
        ax.set_xlabel(r"$t_1$", fontsize=9)
        ax.set_ylabel(r"$t_2$", fontsize=9)
        ax.tick_params(axis="both", which="major", labelsize=9)
        ax.tick_params(axis="both", which="minor", labelsize=8)
        ax.grid(alpha=0.4)

        if targets is not None:
            labelset_targets = np.unique(targets)

            for label in labelset_targets:
                points_target = points[targets==label]
                ax.scatter(points_target[:, 0], points_target[:, 1], label=labelset[label])
        else:
            ax.scatter(points[:, 0], points[:, 1])

        if images_annotation is not None and targets is not None:
            from matplotlib.offsetbox import OffsetImage, AnnotationBbox

            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            for i, point in enumerate(points):
                imagebox = OffsetImage(images_annotation[i].transpose((1, 2, 0)), zoom=zoom_annotation)
                imagebox.image.axes = ax
                
                ab = AnnotationBbox(
                    imagebox,
                    [point[0], point[1]],
                    xybox=(0, 0),
                    xycoords="data",
                    boxcoords="offset points",
                    pad=0.1,
                    bboxprops=dict(edgecolor=colors[int(np.where(labelset_targets == targets[i])[0])], lw=2),
                )
                ax.add_artist(ab)

        ax.legend(fontsize=9)

    points = TSNE(n_components=2).fit_transform(features_flat[:num_points])
    targets = targets[:num_points]

    fig.add_subplot(1, 1, 1)
    subplot_projection(points, targets)

    plt.tight_layout()
    if path_save:
        plt.savefig(path_save)
    plt.show()

"""Compare four CIFAR-10 experiments using their saved evaluation artifacts."""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, PercentFormatter
from sklearn.metrics import ConfusionMatrixDisplay

from train import OUTPUT_DIR

EXPERIMENTS = (
    ("exp0_baseline", "Exp0 Baseline"),
    ("exp1_scheduler", "Exp1 Scheduler"),
    ("exp2_deeper_cnn", "Exp2 Deeper CNN"),
    ("exp3_over_reg", "Exp3 Over-regularized"),
)
COLORS = ("#24578C", "#B34C1A", "#207A63", "#7F4F97")
LINESTYLES = ("-", "--", "-.", ":")
HARD_CLASSES = ("cat", "dog", "bird")


@dataclass
class Experiment:
    name: str
    label: str
    history: dict
    accuracy: float
    macro: list[float]
    class_f1: dict[str, float]
    matrix: np.ndarray
    inference_ms: float
    inference_device: str


def load_experiment(output_dir: Path, name: str, label: str) -> Experiment:
    directory = output_dir / name
    history = json.loads((directory / "history.json").read_text(encoding="utf-8"))
    report = (directory / "classification_report.txt").read_text(encoding="utf-8")
    metadata = dict(line.split(": ", 1) for line in report.splitlines() if ": " in line)
    fields = [line.split() for line in report.splitlines()]
    accuracy = next(float(row[1]) for row in fields if row and row[0] == "accuracy")
    scores = {
        " ".join(row[:-4]): [float(value) for value in row[-4:-1]]
        for row in fields if len(row) >= 5 and row[-1].isdecimal()
    }

    with (directory / "confusion_matrix.csv").open(encoding="utf-8", newline="") as file:
        header, *rows = csv.reader(file)
    class_names = header[1:]
    if [row[0] for row in rows] != class_names:
        raise ValueError(f"Invalid confusion matrix class order: {directory}")
    matrix = np.array([row[1:] for row in rows], dtype=int)

    return Experiment(
        name=name,
        label=label,
        history=history,
        accuracy=accuracy,
        macro=scores["macro avg"],
        class_f1={class_name: scores[class_name][2] for class_name in class_names},
        matrix=matrix,
        inference_ms=float(metadata["inference_ms_per_batch"]),
        inference_device=metadata["inference_device"],
    )


def save_summary(experiments: list[Experiment], path: Path) -> None:
    baseline = experiments[0]
    columns = [
        "Experiment", "Parameters", "Best epoch", "Best val acc (%)", "Test acc (%)",
        "Macro precision (%)", "Macro recall (%)", "Macro F1 (%)",
        *(f"{name} F1 delta (pp)" for name in HARD_CLASSES),
        "Train time (s)", "Inference (ms/batch)", "Inference device",
    ]
    rows = []
    for experiment in experiments:
        history = experiment.history
        rows.append([
            experiment.name,
            history["param_count"],
            history["best_epoch"],
            f"{history['best_val_acc'] * 100:.2f}",
            f"{experiment.accuracy * 100:.2f}",
            *(f"{value * 100:.2f}" for value in experiment.macro),
            *(f"{(experiment.class_f1[name] - baseline.class_f1[name]) * 100:+.2f}" for name in HARD_CLASSES),
            f"{history['elapsed_seconds']:.2f}",
            f"{experiment.inference_ms:.4f}",
            experiment.inference_device,
        ])
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        writer.writerows(rows)
    for row in (columns, ["---"] * len(columns), *rows):
        print("| " + " | ".join(map(str, row)) + " |")


def plot_curves(experiments: list[Experiment], path: Path) -> None:
    figure = Figure(figsize=(13, 4.2), layout="constrained")
    figure.suptitle("CIFAR-10 validation metrics and learning rates")
    axes = figure.subplots(1, 3)
    last_epoch = max(experiment.history["epochs"][-1]["epoch"] for experiment in experiments)
    for axis, key, ylabel in zip(
        axes, ("val_loss", "val_acc", "lr"), ("Validation loss", "Validation accuracy", "Learning rate"),
        strict=True,
    ):
        for experiment, color, linestyle in zip(experiments, COLORS, LINESTYLES, strict=True):
            epochs = experiment.history["epochs"]
            axis.plot([row["epoch"] for row in epochs], [row[key] for row in epochs],
                      color=color, linestyle=linestyle, marker=".", markersize=3, label=experiment.label)
        axis.set(xlabel="Epoch", ylabel=ylabel, xlim=(0.5, last_epoch + 0.5))
        axis.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True, min_n_ticks=1))
        axis.grid(alpha=0.22)
    axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=1))
    axes[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncols=4, frameon=False)
    figure.savefig(path, dpi=300, backend="agg")


def plot_class_f1(baseline: Experiment, best: Experiment, path: Path) -> None:
    classes = list(baseline.class_f1)
    positions = np.arange(len(classes))
    figure = Figure(figsize=(10, 4.4), layout="constrained")
    figure.suptitle("Test F1 by class")
    axis = figure.subplots()
    for offset, experiment, color in ((-0.19, baseline, COLORS[0]), (0.19, best, COLORS[2])):
        axis.bar(positions + offset, [experiment.class_f1[name] for name in classes], width=0.38,
                 color=color, label=experiment.label)
    axis.set(ylabel="F1 score", ylim=(0, 1), xticks=positions, xticklabels=classes)
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    axis.grid(axis="y", alpha=0.22)
    axis.set_axisbelow(True)
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncols=2, frameon=False)
    figure.savefig(path, dpi=300, backend="agg")


def plot_confusion_matrices(baseline: Experiment, best: Experiment, path: Path) -> None:
    figure = Figure(figsize=(13, 6.2), layout="constrained")
    figure.suptitle("CIFAR-10 test confusion matrices")
    axes = figure.subplots(1, 2)
    max_count = max(baseline.matrix.max(), best.matrix.max())
    for axis, experiment in zip(axes, (baseline, best), strict=True):
        display = ConfusionMatrixDisplay(experiment.matrix, display_labels=list(experiment.class_f1))
        display.plot(ax=axis, cmap="Blues", colorbar=False, values_format="d", xticks_rotation=40,
                     im_kw={"vmin": 0, "vmax": max_count})
        axis.set(title=experiment.label, xlabel="Predicted class", ylabel="True class")
    figure.colorbar(axes[0].images[0], ax=axes, shrink=0.8, label="Images")
    figure.savefig(path, dpi=300, backend="agg")


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总四组 CIFAR-10 实验并生成对比图")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="实验产物根目录，默认 outputs")
    args = parser.parse_args()
    try:
        experiments = [load_experiment(args.output_dir, name, label) for name, label in EXPERIMENTS]
    except FileNotFoundError as error:
        parser.error(f"找不到实验产物 {error.filename}。请先运行四组训练和评估。")

    baseline = experiments[0]
    if any(list(experiment.class_f1) != list(baseline.class_f1) for experiment in experiments[1:]):
        parser.error("四组实验的类别顺序应一致")
    best = max(experiments[1:], key=lambda experiment: experiment.history["best_val_acc"])
    save_summary(experiments, args.output_dir / "comparison_summary.csv")
    plot_curves(experiments, args.output_dir / "comparison_curves.png")
    plot_class_f1(baseline, best, args.output_dir / "comparison_class_f1.png")
    plot_confusion_matrices(baseline, best, args.output_dir / "comparison_cm_diff.png")
    print(f"最佳改进候选 {best.name}，依据验证集准确率。结果已保存到 {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

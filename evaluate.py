"""Evaluate the saved CIFAR-10 checkpoint and generate experiment figures."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator, PercentFormatter
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from torch.utils.data import DataLoader

from dataset import load_test_set, set_seed
from model import Cifar10CNN
from train import OUTPUT_DIR


@torch.inference_mode()
def predict(
    model: Cifar10CNN, test_loader: DataLoader, device: torch.device,
) -> tuple[list[int], list[int]]:
    model.eval()
    y_true = []
    y_pred = []
    for images, labels in test_loader:
        logits = model(images.to(device))
        y_true.extend(labels.tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().tolist())
    return y_true, y_pred


def plot_training_curves(history: dict, path: Path) -> None:
    rows = history["epochs"]
    epochs = [row["epoch"] for row in rows]
    best_epoch = history["best_epoch"]

    figure = Figure(figsize=(11, 4.2), layout="constrained")
    figure.suptitle("CIFAR-10 training and validation")
    loss_axis, accuracy_axis = figure.subplots(1, 2)
    for axis, metric, label in (
        (loss_axis, "loss", "Cross-entropy loss"),
        (accuracy_axis, "acc", "Accuracy"),
    ):
        axis.plot(epochs, [row[f"train_{metric}"] for row in rows],
                  color="#24578C", marker="o", markersize=3, label="Training")
        axis.plot(epochs, [row[f"val_{metric}"] for row in rows],
                  color="#B34C1A", marker="s", markersize=3, label="Validation")
        axis.axvline(best_epoch, color="#555555", linestyle="--", linewidth=1,
                     label=f"Best epoch ({best_epoch})")
        axis.set(xlabel="Epoch", ylabel=label, xlim=(epochs[0] - 0.5, epochs[-1] + 0.5))
        axis.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
        axis.grid(alpha=0.22)

    accuracy_axis.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    handles, labels = loss_axis.get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncols=3, frameon=False)
    figure.savefig(path, dpi=300, backend="agg")


def save_confusion_matrix(matrix: np.ndarray, class_names: list[str], output_dir: Path) -> None:
    with (output_dir / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["true_label", *class_names])
        for class_name, counts in zip(class_names, matrix.tolist(), strict=True):
            writer.writerow([class_name, *counts])

    figure = Figure(figsize=(8.4, 7.2), layout="constrained")
    axis = figure.subplots()
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=class_names)
    display.plot(ax=axis, cmap="Blues", values_format="d", xticks_rotation=35)
    axis.set(title="CIFAR-10 test confusion matrix", xlabel="Predicted class", ylabel="True class")
    figure.savefig(output_dir / "confusion_matrix.png", dpi=300, backend="agg")


def main() -> None:
    parser = argparse.ArgumentParser(description="评估已训练的 CIFAR-10 CNN 并生成结果图")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="训练产物目录，默认 outputs")
    args = parser.parse_args()
    checkpoint_path = args.output_dir / "best_cnn_cifar10.pth"
    history_path = args.output_dir / "history.json"
    for path in (checkpoint_path, history_path):
        if not path.is_file():
            parser.error(f"找不到训练产物 {path}。请先运行 train.py。")

    history = json.loads(history_path.read_text(encoding="utf-8"))
    set_seed(history["config"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Cifar10CNN()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    model.to(device)
    test_set = load_test_set()
    test_loader = DataLoader(test_set, batch_size=history["config"]["batch_size"], shuffle=False)
    y_true, y_pred = predict(model, test_loader, device)

    class_names = test_set.classes
    class_ids = list(range(len(class_names)))
    report = classification_report(y_true, y_pred, labels=class_ids, target_names=class_names,
                                   digits=4, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=class_ids)

    (args.output_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    plot_training_curves(history, args.output_dir / "training_curves.png")
    save_confusion_matrix(matrix, class_names, args.output_dir)

    print(f"设备 {device} | 官方测试集 {len(test_set):,} 张 | 权重来自第 {history['best_epoch']} 轮")
    print(report)
    print("混淆矩阵各行是真实类别，各列是预测类别，类别顺序与分类报告相同。")
    print(matrix)
    print(f"评估结果已保存到 {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()

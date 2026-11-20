"""Train the CIFAR-10 CNN and save the best validation checkpoint."""

import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset import DEFAULT_BATCH_SIZE, DEFAULT_SEED, load_train_validation, make_loaders, set_seed
from model import Cifar10CNN

DEFAULT_EPOCHS = 30
DEFAULT_LR = 1e-3
WEIGHT_DECAY = 1e-4
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Adam | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total_samples = 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            if optimizer is not None:
                optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size

    return total_loss / total_samples, correct / total_samples


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 CIFAR-10 CNN 并保存验证集最优权重")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="训练轮数，默认 30")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="批大小，默认 128")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="学习率，默认 0.001")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="随机种子，默认 42")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="产物目录，默认 outputs")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs 和 --batch-size 必须是正整数")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr 必须是有限正数")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed 必须在 0 到 2**32 - 1 之间")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set, val_set = load_train_validation(args.seed)
    train_loader, val_loader = make_loaders(train_set, val_set, args.batch_size, args.seed)
    model = Cifar10CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "best_cnn_cifar10.pth"
    history_path = args.output_dir / "history.json"
    checkpoint_temp = checkpoint_path.with_suffix(".pth.tmp")
    history_temp = history_path.with_suffix(".json.tmp")
    rows = []
    config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": WEIGHT_DECAY,
        "seed": args.seed,
        "train_samples": len(train_set),
        "val_samples": len(val_set),
    }

    best_accuracy = -1.0
    best_epoch = 0
    print(f"设备 {device} | 训练 {len(train_set):,} 张 | 验证 {len(val_set):,} 张", flush=True)
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        if val_acc > best_accuracy:
            with checkpoint_temp.open("wb") as file:
                torch.save(model.state_dict(), file)
            checkpoint_temp.replace(checkpoint_path)
            best_accuracy = val_acc
            best_epoch = epoch

        history = {"config": config, "best_epoch": best_epoch, "best_val_acc": best_accuracy, "epochs": rows}
        history_temp.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        history_temp.replace(history_path)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | train loss {train_loss:.4f}, acc {train_acc:.2%}"
            f" | val loss {val_loss:.4f}, acc {val_acc:.2%}",
            flush=True,
        )

    print(f"最优权重来自第 {best_epoch} 轮，验证准确率 {best_accuracy:.2%}。")
    print(f"权重 {checkpoint_path.resolve()}\n日志 {history_path.resolve()}")


if __name__ == "__main__":
    main()

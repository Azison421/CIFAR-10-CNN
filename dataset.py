"""Load, preprocess, and inspect the CIFAR-10 dataset."""

import argparse
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

DATA_DIR = Path(__file__).resolve().parent / "data"
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
DEFAULT_BATCH_SIZE = 128
DEFAULT_SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_cifar10() -> tuple[CIFAR10, CIFAR10]:
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    train_set = CIFAR10(root=DATA_DIR, train=True, transform=train_transform, download=True)
    test_set = CIFAR10(root=DATA_DIR, train=False, transform=test_transform, download=True)
    return train_set, test_set


def make_loaders(
    train_set: CIFAR10,
    test_set: CIFAR10,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = DEFAULT_SEED,
) -> tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def print_distribution(train_set: CIFAR10, test_set: CIFAR10) -> None:
    train_counts = Counter(train_set.targets)
    test_counts = Counter(test_set.targets)
    print(f"{'ID':>2}  {'Class':<12} {'Train':>7} {'Share':>8} {'Test':>7} {'Share':>8}")
    for label, name in enumerate(train_set.classes):
        train_count = train_counts[label]
        test_count = test_counts[label]
        print(
            f"{label:>2}  {name:<12} {train_count:>7} {train_count / len(train_set):>8.2%}"
            f" {test_count:>7} {test_count / len(test_set):>8.2%}"
        )

    print(f"\n训练集共 {len(train_set):,} 张，测试集共 {len(test_set):,} 张。")
    if train_counts == dict.fromkeys(range(10), 5000) and test_counts == dict.fromkeys(range(10), 1000):
        print("均衡性验证通过。训练集和测试集的每个类别均占 10%。")
    else:
        print("均衡性验证未通过，请核对各类样本数。")


def main() -> None:
    parser = argparse.ArgumentParser(description="加载 CIFAR-10 并检查类别分布与预处理结果")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="批大小，默认 128")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="随机种子，默认 42")
    args = parser.parse_args()

    set_seed(args.seed)
    train_set, test_set = load_cifar10()
    train_loader, test_loader = make_loaders(train_set, test_set, args.batch_size, args.seed)
    print_distribution(train_set, test_set)

    for name, loader in (("训练集", train_loader), ("测试集", test_loader)):
        images, labels = next(iter(loader))
        print(f"{name}首批图像 {tuple(images.shape)} {images.dtype}，标签 {tuple(labels.shape)} {labels.dtype}")


if __name__ == "__main__":
    main()

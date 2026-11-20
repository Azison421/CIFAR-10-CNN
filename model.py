"""Baseline and deeper convolutional networks for CIFAR-10."""

import argparse

import torch
from torch import nn

ARCHITECTURES = ("baseline", "deeper")
DEFAULT_DROPOUT = {"baseline": 0.5, "deeper": 0.3}


def _conv_layers(in_channels: int, out_channels: int) -> tuple[nn.Module, ...]:
    return (
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )


def _conv_block(in_channels: int, out_channels: int, convolutions: int = 1) -> nn.Sequential:
    layers = list(_conv_layers(in_channels, out_channels))
    if convolutions == 2:
        layers.extend(_conv_layers(out_channels, out_channels))
    return nn.Sequential(*layers, nn.MaxPool2d(kernel_size=2, stride=2))


class Cifar10CNN(nn.Module):
    """Map a batch of 32 × 32 RGB images to ten logits per image."""

    def __init__(self, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(3, 32),
            _conv_block(32, 64),
            _conv_block(64, 128),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        return self.classifier(features)


class DeeperCifar10CNN(nn.Module):
    """Extract features with two convolutions per block and average pool them."""

    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(3, 64, convolutions=2),
            _conv_block(64, 128, convolutions=2),
            _conv_block(128, 256, convolutions=2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))


def build_model(arch: str, dropout: float) -> nn.Module:
    if arch == "baseline":
        return Cifar10CNN(dropout)
    if arch == "deeper":
        return DeeperCifar10CNN(dropout)
    raise ValueError(f"Unknown architecture: {arch}")


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="检查 CIFAR-10 CNN 结构与输出")
    parser.add_argument("--arch", choices=ARCHITECTURES, default="baseline", help="模型架构")
    parser.add_argument("--dropout", type=float, help="分类头的 Dropout 概率")
    args = parser.parse_args()
    dropout = DEFAULT_DROPOUT[args.arch] if args.dropout is None else args.dropout
    if not 0 <= dropout <= 1:
        parser.error("--dropout 必须在 0 到 1 之间")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.arch, dropout).to(device).eval()
    images = torch.zeros(1, 3, 32, 32, device=device)
    logits = model(images)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    print(model)
    print(f"设备 {device}")
    print(f"参数量 {parameter_count:,}")
    print(f"输入形状 {tuple(images.shape)}")
    print(f"输出形状 {tuple(logits.shape)}")


if __name__ == "__main__":
    main()

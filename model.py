"""A three-block convolutional network for CIFAR-10."""

import torch
from torch import nn


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )


class Cifar10CNN(nn.Module):
    """Map a batch of 32 × 32 RGB images to ten logits per image."""

    def __init__(self) -> None:
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
            nn.Dropout(p=0.5),
            nn.Linear(256, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        return self.classifier(features)


@torch.inference_mode()
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Cifar10CNN().to(device).eval()
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

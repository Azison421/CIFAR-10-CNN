# 基于 CNN 的 CIFAR-10 图像分类

《数字图像处理》实验一。当前阶段完成 CIFAR-10 数据加载、类别分布统计与预处理，使用 Python 3.12、PyTorch 和 TorchVision。

## 运行数据检查

在本项目目录打开 PowerShell。已有 `.venv` 时，直接用项目环境运行数据检查。

```powershell
.\.venv\Scripts\python.exe dataset.py
```

在新环境中，使用标准 Python 3.12 安装依赖。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
```

使用与本机相同的 CUDA 13.2 构建时，将安装命令中的 `/whl/cpu` 改为 `/whl/cu132`。[PyTorch 安装说明](https://pytorch.org/get-started/previous-versions/)

首次运行会将 [CIFAR-10 官方数据](https://www.cs.toronto.edu/~kriz/cifar.html) 下载到 `data/`，后续运行复用本地数据。可以用 `--batch-size` 和 `--seed` 修改批大小和随机种子，默认值为 128 和 42。

官方站点下载较慢时，可将镜像的 `cifar-10-python.tar.gz` 放入 `data/` 后运行脚本。压缩包的官方 MD5 为 `c58f30108f718f92721af3b95e74349a`。[PaddlePaddle 数据镜像](https://dataset.bj.bcebos.com/cifar/cifar-10-python.tar.gz)

## 检查项

- 统计训练集和测试集 10 个类别的样本数与占比。预期训练集 50,000 张、每类 5,000 张，测试集 10,000 张、每类 1,000 张。两个集合中各类均占 10%。
- 训练集依次进行随机裁剪、随机水平翻转、颜色抖动、张量转换和 RGB 归一化。测试集仅进行张量转换并使用相同参数归一化。
- 训练集按批随机读取，测试集按原顺序读取。默认批大小下，两者的首批图像形状均为 `(128, 3, 32, 32)`，类别标签形状为 `(128,)`。

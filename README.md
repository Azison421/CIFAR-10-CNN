# 基于 CNN 的 CIFAR-10 图像分类

《数字图像处理》实验一。使用 PyTorch 训练三块卷积的 CNN，在 CIFAR-10 官方测试集上正确分类 **8,218 / 10,000** 张。

## 实验结果

第 30 轮的验证准确率最高，为 **82.98%**。用该轮权重评估，CUDA 测试准确率为 **82.18%**。宏平均与加权平均相同，Precision 为 **82.56%**，Recall 为 **82.18%**，F1 为 **82.29%**。CPU 复算准确率为 **82.17%**。两次评估只有一张图像的预测不同，CUDA 判为猫，CPU 判为鹿。

## 复现实验

在项目目录使用标准 Python 3.12 创建环境并安装依赖。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

安装 CUDA 13.2 构建时，将 PyTorch 安装命令中的 `/whl/cpu` 改为 `/whl/cu132`。[PyTorch 安装说明](https://pytorch.org/get-started/previous-versions/)

依次运行数据检查、模型检查、训练和评估。模型检查、训练和评估优先使用 CUDA，CPU 环境可直接运行。

```powershell
.\.venv\Scripts\python.exe dataset.py
.\.venv\Scripts\python.exe model.py
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe evaluate.py
```

首次运行会下载数据到 `data/`。也可将镜像的 `cifar-10-python.tar.gz` 放入该目录。官方 MD5 为 `c58f30108f718f92721af3b95e74349a`。[CIFAR-10 官方页面](https://www.cs.toronto.edu/~kriz/cifar.html) · [PaddlePaddle 数据镜像](https://dataset.bj.bcebos.com/cifar/cifar-10-python.tar.gz)

默认设置为 30 轮、批大小 128、学习率 `0.001`、权重衰减 `0.0001` 和随机种子 42，使用交叉熵和 Adam。官方训练集分为 45,000 张训练图像和 5,000 张验证图像。仅训练子集使用随机裁剪、水平翻转和颜色抖动，官方测试集用于最终评估。

## 输出文件

脚本在本地生成 `outputs/`。用 `--output-dir` 指定其他目录时，训练和评估使用同一个目录。

- `best_cnn_cifar10.pth` 和 `history.json` 保存最佳验证权重、训练配置和逐轮指标。
- `classification_report.txt` 记录总体和逐类测试指标。
- `training_curves.png` 标出最佳轮次，`confusion_matrix.csv` 和 `confusion_matrix.png` 按真实类别为行、预测类别为列记录计数。

## 优化对比实验

四组实验使用相同的 45,000／5,000 训练验证划分、种子 42 和批大小 128。运行以下脚本可依次训练、评估并汇总四组结果。

```powershell
.\run_experiments.ps1
```

| 组别 | 轮数 | 架构 | 优化器与调度 | 标签平滑 | Dropout | 数据增强 |
| --- | ---: | --- | --- | ---: | ---: | --- |
| Exp0 | 30 | baseline | Adam，固定学习率 | 0 | 0.5 | standard |
| Exp1 | 50 | baseline | AdamW，预热 3 轮后余弦退火 | 0.05 | 0.5 | standard |
| Exp2 | 50 | deeper | 同 Exp1 | 0.05 | 0.3 | standard |
| Exp3 | 50 | deeper | 同 Exp1 | 0.05 | 0.7 | heavy |

四组的目标学习率均为 `0.001`。Exp0 的权重衰减为 `0.0001`，其余为 `0.0005`。Exp3 比较强数据增强与高 Dropout 的组合效果，实际影响以结果为准。验证 Loss 统一使用普通交叉熵，标签平滑只用于训练。

`train.py` 可单独指定 `--arch`、`--optimizer`、`--scheduler`、`--label-smoothing`、`--dropout` 和 `--aug-level`。SGD 的动量为 0.9。评估从对应训练日志读取模型配置。日志记录参数量、每轮学习率和耗时。分类报告记录首个测试批次预热后的 10 次前向传播平均耗时，不计数据搬运。

`compare.py` 汇总四组测试指标与猫、狗、鸟的 F1 变化，按验证准确率从 Exp1–Exp3 选择对比对象。测试集用于报告，不参与选择。汇总表和三张图分别保存为 `outputs/comparison_summary.csv`、`outputs/comparison_curves.png`、`outputs/comparison_class_f1.png` 和 `outputs/comparison_cm_diff.png`。

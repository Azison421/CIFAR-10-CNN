# 《数字图像处理》实验一：CNN 图像分类优化策略与对比实验方案

> **设计背景**：当前第一版基线模型（`Baseline`：3 层单卷积块 + 固定学习率 `Adam(lr=1e-3)` + `30 epochs`）在 CIFAR-10 官方测试集上已取得 **82.18%** 准确率（宏平均 F1 为 **0.8229**），但在混淆矩阵中暴露出 `cat`（F1=0.6785）与 `dog`（F1=0.7562）、`bird`（F1=0.7438）互判较多，且后期固定学习率收敛进入平台期等问题。
>
> **方案目标**：为满足实验报告中“**多种合理改进手段 + 优化前后科学对比 + 负优化/失效策略原因剖析**”的深度要求，本方案基于现有仓库（`dataset.py`、`model.py`、`train.py`、`evaluate.py`）设计 **4 组递进对照实验**，通过单一变量控制与综合优化，自动生成完整的对比数据表与可视化图表。

---

## 一、 实验组别设计（控制变量与对照矩阵）

固定随机种子 `seed = 42`、训练集/验证集划分（`45,000 / 5,000`）与 `batch_size = 128` 不变，依次运行以下 **4 组实验**，并将每组产物分别保存至 `./outputs/exp0_baseline/`、`./outputs/exp1_scheduler/`、`./outputs/exp2_deeper_cnn/`、`./outputs/exp3_over_reg/`（用于触发并分析负优化/性能下降现象）：

| 实验编号与名称 | 网络结构 | 训练轮数 | 优化器与学习率策略 | 数据增强与正则化 | 预期验证目的 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Exp0: Baseline（基线组）** | 3 块单层卷积 (`32-64-128`) | `30` | `Adam(lr=1e-3)`，**无学习率调度** | `RandomCrop(32,4)` + `HFlip` + `ColorJitter(0.2)` + `Dropout(0.5)` | 作为客观对比基准（当前测试集 Acc = **82.18%**）。 |
| **Exp1: Scheduler（训练调度调参组）** | 同 Exp0 | `50` | `AdamW(lr=1e-3, weight_decay=5e-4)` + **余弦退火调度 `CosineAnnealingLR`**（配合前 3 轮线性 Warmup） | 同 Exp0 + **标签平滑 `LabelSmoothing=0.05`** | 验证在**不改变网络结构**的前提下，延长训练周期配合余弦退火学习率衰减与权重衰减对后期震荡收敛的改善效果。 |
| **Exp2: DeeperVGG-CNN（结构与特征提取改进组）** | **升级为 3 阶段双卷积堆叠网络**（每阶段 `Conv-BN-ReLU` $\times 2$ + `MaxPool`，通道 `64-128-256`）+ **全局平均池化 `AdaptiveAvgPool2d(1)`** 替代大尺寸 `Flatten` 全连接 | `50` | 同 Exp1（`AdamW` + Warmup + `CosineAnnealingLR`） | 同 Exp1（全连接前 `Dropout(0.3)`） | 1. 双层 $3\times 3$ 卷积串联等价于 $5\times 5$ 感受野且非线性更强，增强对 `cat/dog/bird` 细粒度局部纹理的提取能力；<br>2. 用全局平均池化替代 `2048 -> 256` 全连接，大幅削减全连接层参数并抑制过拟合。 |
| **Exp3: Over-Regularized（过度正则化对照组 / 失效分析组）** | 同 Exp2 | `50` | `SGD(lr=0.1, momentum=0.9)` **不加学习率衰减**，或叠加**过强数据增强与高丢弃率**（`RandomErasing(p=0.5)` + `ColorJitter(0.5)` + `Dropout(0.7)`） | 过强随机遮挡/色彩畸变 + 过高 `Dropout(0.7)` | **专门用于课程报告中的“失败/负优化案例分析”**：观察过度正则化与过强噪声增强导致模型欠拟合（训练准确率与测试准确率同步下滑）的现象，定量分析“正则化强度与模型容量匹配”的机理。 |

---

## 二、 针对现有代码的具体改造要点

### 1. `model.py`：支持多模型架构切换
* **干什么**：保留原有的 `Cifar10CNN`（作为 `baseline` 模型），新增改进型 `DeeperCifar10CNN`（双卷积块 + 通道扩充 `64->128->256` + `AdaptiveAvgPool2d`），并通过工厂函数 `build_model(arch: str, dropout: float)` 统一调用。
* **关键结构设计（`DeeperCifar10CNN`）**：
  * **Stage 1**：`[Conv(3->64, 3x3, pad=1) + BN + ReLU] × 2` $\rightarrow$ `MaxPool(2)`（输出 `64 × 16 × 16`）
  * **Stage 2**：`[Conv(64->128, 3x3, pad=1) + BN + ReLU] × 2` $\rightarrow$ `MaxPool(2)`（输出 `128 × 8 × 8`）
  * **Stage 3**：`[Conv(128->256, 3x3, pad=1) + BN + ReLU] × 2` $\rightarrow$ `MaxPool(2)`（输出 `256 × 4 × 4`）
  * **分类头**：`AdaptiveAvgPool2d((1, 1))` $\rightarrow$ `Flatten()` $\rightarrow$ `Dropout(p=dropout)` $\rightarrow$ `Linear(256, 10)`
* **同时统计并保存**：每组模型的**总参数量（Parameters）**与单批推理耗时，用于报告中“精度-复杂度权衡”分析。

### 2. `dataset.py`：支持增强强度配置与 CUDA 种子补全
* **干什么**：
  1. 在 `set_seed(seed)` 中补充 `torch.cuda.manual_seed_all(seed)`，消除 GPU/CPU 评估结果存在单张样本差异的问题。
  2. 为 `load_train_validation` 增加 `aug_level` 参数（`"standard"` 为标准增强；`"heavy"` 为 Exp3 专用强扰动增强，加入强 `ColorJitter(0.5, 0.5, 0.5)` 与 `RandomErasing(p=0.5)`），确保测试集始终保持确定性归一化。

### 3. `train.py`：支持优化器、学习率调度器与标签平滑参数化
* **干什么**：扩展命令行参数，使单条命令即可复现任意一组实验：
  * `--arch`：`baseline` 或 `deeper`
  * `--optimizer`：`adam`、`adamw`、`sgd`
  * `--scheduler`：`none` 或 `cosine`（带前 3 轮线性 Warmup）
  * `--label-smoothing`：默认 `0.0`，改进组设为 `0.05`
  * `--dropout`：基线组 `0.5`，改进组 `0.3`，过正则组 `0.7`
  * `--aug-level`：`standard` 或 `heavy`
* **日志增强**：在 `history.json` 中额外记录每个 epoch 的当前学习率 `lr`、总参数量 `param_count` 以及总训练耗时 `elapsed_seconds`。

### 4. `compare.py`（新增汇总对比脚本）：一键生成报告所需的全部对比图表
* **干什么**：读取 4 组实验目录下保存的 `history.json`、`classification_report.txt` 和 `confusion_matrix.csv`，自动汇总输出：
  1. **`comparison_summary.csv` & 终端 Markdown 表格**：横向对比 4 组实验的**参数量、最佳验证轮次、最佳验证准确率、测试集 Accuracy、Macro Precision、Macro Recall、Macro F1**，以及难分类别（`cat`、`dog`、`bird`）的 F1 提升幅度。
  2. **`comparison_curves.png`（多组训练动态对比图）**：在同一张双子图中对比 4 组实验的验证集 Loss 曲线与验证集 Accuracy 曲线（附带学习率调度衰减曲线子图），直观展示余弦退火与加深网络对收敛速度及上限的提升，以及 Exp3 欠拟合的曲线特征。
  3. **`comparison_class_f1.png`（各类别 F1 对比柱状图）**：横向对比优化前后（如 `Exp0` vs `Exp2`）在 10 个类别上的 F1 分数变化，直观展示易混淆类别（尤其是 `cat`、`dog`、`bird`）的改善幅度。
  4. **`comparison_cm_diff.png`（优化前后混淆矩阵对比图）**：并排展示 `Exp0` 与 `Exp2` 的混淆矩阵热力图（或差值矩阵），直观呈现 `cat` 与 `dog` 之间误判次数的下降。

---

## 三、 发给 GPT 的直接执行提示词

你可以把这份 Markdown 文档连同现有仓库代码（或仓库结构说明）发给 GPT，并附上以下指令：

```text
请基于我现有的 CIFAR-10-CNN 仓库代码（dataset.py、model.py、train.py、evaluate.py），严格按照上述《实验一：优化策略与消融对比实验方案》进行升级改造，并新增 compare.py 对比分析脚本。

具体要求：
1. 保持原有代码风格与命令行参数规范，补全 set_seed 中的 torch.cuda.manual_seed_all(seed)。
2. 改造 model.py、dataset.py、train.py、evaluate.py，使其支持通过命令行参数切换架构（baseline / deeper）、优化器（adam / adamw / sgd）、学习率调度（none / cosine）、标签平滑、Dropout 比例与数据增强强度（standard / heavy），并在 history.json 中记录模型参数量、每轮学习率及训练耗时。
3. 新增 compare.py 脚本，自动读取 outputs/exp0_baseline、outputs/exp1_scheduler、outputs/exp2_deeper_cnn、outputs/exp3_over_reg 四个目录的评估结果，生成：
   - 汇总对比表（打印 Markdown 表格并保存为 outputs/comparison_summary.csv）；
   - 多组验证 Loss / Accuracy / 学习率调度对比曲线图（outputs/comparison_curves.png）；
   - 优化前后 10 类别 F1 对比柱状图（outputs/comparison_class_f1.png）；
   - 基线组与最优组混淆矩阵并排对比图（outputs/comparison_cm_diff.png）。
4. 请给出一次性依次运行 4 组实验及生成对比结果的完整 PowerShell 命令脚本。
```

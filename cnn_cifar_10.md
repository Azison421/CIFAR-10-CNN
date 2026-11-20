# 《数字图像处理》实验一：基于 CNN 的图像分类实施方案

> **使用说明**：本方案已按标准工程规范拆解为“干什么、怎么做、目的是什么、实现途径”四要素。你可以直接将整份 Markdown 文档连同文末的提示词一起发给 GPT 生成完整可运行代码。

---

## 0. 项目工程规范与总目标

* **干什么**：搭建一套端到端的 CIFAR-10（10 分类，$32 \times 32$ RGB 彩色图像，训练集 $50,000$ 张、测试集 $10,000$ 张）图像分类实验流水线，涵盖数据加载与统计、预处理与数据增强、简易 CNN 网络搭建、模型训练与最优权重保存、测试集多维指标评估及可视化图表输出。
* **怎么做**：
  1. 采用模块化脚本结构（如 `dataset.py`、`model.py`、`train.py`、`evaluate.py` 或清晰分段的完整单脚本）。
  2. 固定随机种子（Python / NumPy / PyTorch）以保证实验结果可复现。
  3. 自动检测计算设备（优先使用 CUDA GPU，否则自动回退至 CPU）。
  4. 建立统一输出目录（如 `./outputs/`），自动保存最优模型权重、训练指标日志、Loss/Accuracy 曲线图及混淆矩阵图。
* **目的是**：建立清晰、可追溯、可复现的实验基线，确保课程要求的每一项任务都有明确的程序输出与量化结果，方便后续参数调优及撰写实验报告。
* **实现途径**：Python + PyTorch + `torchvision` + `scikit-learn` + `matplotlib`（可选 `seaborn`）。

---

## 第一阶段：数据加载与预处理

### 1.1 加载数据集与统计样本分布（验证均衡性）

* **干什么**：读取 CIFAR-10 训练集与测试集，统计 $10$ 个类别（飞机 `airplane`、汽车 `automobile`、鸟 `bird`、猫 `cat`、鹿 `deer`、狗 `dog`、青蛙 `frog`、马 `horse`、船 `ship`、卡车 `truck`）各自的样本数量，并验证数据集类别分布是否均衡。
* **怎么做**：
  1. 调用 `torchvision.datasets.CIFAR10` 分别加载 `train=True` 和 `train=False` 数据集。
  2. 遍历或读取标签列表，按类别索引 `0~9` 统计训练集与测试集各类样本数及占比。
  3. 在终端打印格式化的样本分布统计表，并输出均衡性验证结论（标准 CIFAR-10 训练集每类 $5,000$ 张、测试集每类 $1,000$ 张，各占 $10\%$）。
* **目的是**：确认数据下载与读取完整、类别标签映射准确无误，同时校验是否存在类别不平衡问题，避免后续评估指标因样本倾斜而失真。
* **实现途径**：`torchvision.datasets.CIFAR10`、`collections.Counter` 或 `numpy.bincount`。

### 1.2 像素归一化与训练集数据增强

* **干什么**：将原始图像像素转换为模型可计算的标准化张量；仅对**训练集**实施随机水平翻转、随机裁剪、颜色抖动等数据增强操作，**测试集**仅做张量转换与同参数归一化。
* **怎么做**：
  1. **训练集变换流水线（`train_transform`）**：
     * 随机裁剪：`RandomCrop(32, padding=4)`（先四周填充 $4$ 像素再随机裁回 $32 \times 32$）。
     * 随机水平翻转：`RandomHorizontalFlip(p=0.5)`。
     * 颜色抖动：`ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)`（适度扰动亮度、对比度与饱和度）。
     * 张量转换：`ToTensor()`（将 `[0, 255]` 像素映射至 `[0.0, 1.0]` 且维度转为 `(C, H, W)`）。
     * 像素归一化：`Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))`（CIFAR-10 RGB 三通道标准均值与标准差）。
  2. **测试集变换流水线（`test_transform`）**：
     * 仅包含 `ToTensor()` 与与训练集完全一致的 `Normalize(...)`，**严禁加入任何随机增强**。
  3. 使用 `DataLoader` 封装训练集（`shuffle=True`）与测试集（`shuffle=False`）。
* **目的是**：
  * **归一化**：消除各通道像素量纲差异，使输入分布接近均值为 $0$、方差为 $1$，加快梯度下降收敛速度并防止数值不稳定。
  * **训练集数据增强**：在不增加额外标注成本的前提下扩充图像视角、位置与光照多样性，有效抑制过拟合。
  * **测试集确定性处理**：保证评估基准统一、客观、可复现。
* **实现途径**：`torchvision.transforms`（`Compose`、`RandomCrop`、`RandomHorizontalFlip`、`ColorJitter`、`ToTensor`、`Normalize`）与 `torch.utils.data.DataLoader`。

---

## 第二阶段：搭建简易 CNN 网络

### 2.1 设计卷积、池化与全连接网络结构

* **干什么**：自行设计一个适配 $32 \times 32$ 小尺寸 RGB 图像的简易卷积神经网络（CNN），由若干特征提取卷积块与全连接分类头组成。
* **怎么做**：
  1. 明确输入张量形状为 `(B, 3, 32, 32)`（`B` 为 `batch_size`，`3` 为 RGB 通道数，高宽为 $32 \times 32$）。
  2. 设计 $3$ 个递进式特征提取模块（卷积块）：
     * **卷积块 1**：`Conv2d(3, 32, kernel_size=3, padding=1)` $\rightarrow$ `BatchNorm2d(32)` $\rightarrow$ `ReLU()` $\rightarrow$ `MaxPool2d(2, 2)`，输出形状变为 `(B, 32, 16, 16)`。
     * **卷积块 2**：`Conv2d(32, 64, kernel_size=3, padding=1)` $\rightarrow$ `BatchNorm2d(64)` $\rightarrow$ `ReLU()` $\rightarrow$ `MaxPool2d(2, 2)`，输出形状变为 `(B, 64, 8, 8)`。
     * **卷积块 3**：`Conv2d(64, 128, kernel_size=3, padding=1)` $\rightarrow$ `BatchNorm2d(128)` $\rightarrow$ `ReLU()` $\rightarrow$ `MaxPool2d(2, 2)`，输出形状变为 `(B, 128, 4, 4)`。
  3. 设计全连接分类头：
     * 将特征图展平：`Flatten()`，得到维度为 $128 \times 4 \times 4 = 2048$ 的一维特征向量。
     * 隐藏全连接层：`Linear(2048, 256)` $\rightarrow$ `ReLU()` $\rightarrow$ `Dropout(0.5)`。
     * 输出层：`Linear(256, 10)`，输出 $10$ 个类别的原始得分（logits）。
  4. 提供前向传播维度检查函数或打印网络结构摘要，确认各层输入输出维度匹配无误。
* **目的是**：利用卷积层的局部感受野与权值共享逐层提取“边缘/颜色 $\rightarrow$ 纹理/部件 $\rightarrow$ 整体轮廓”的层次化语义特征，利用池化层压缩空间维度并增强平移不变性，同时控制参数规模以兼顾训练效率与分类精度。
* **实现途径**：继承 `torch.nn.Module`，使用 `nn.Conv2d`、`nn.BatchNorm2d`、`nn.ReLU`、`nn.MaxPool2d`、`nn.Flatten`、`nn.Linear`、`nn.Dropout`。

---

## 第三阶段：模型训练

### 3.1 配置超参数、损失函数、优化器与训练循环

* **干什么**：设定关键训练超参数（`batch_size`、学习率 `lr`、`epochs`），配置多分类损失函数与参数优化器，编写完整的迭代训练循环。
* **怎么做**：
  1. **超参数设置**：`batch_size = 128`，初始学习率 `lr = 1e-3`（如使用 `Adam`），训练轮数 `epochs = 30`（可根据收敛情况调整至 `40~50`），可搭配学习率调度器（如 `CosineAnnealingLR` 或 `StepLR`）。
  2. **损失函数与优化器**：
     * 损失函数：`nn.CrossEntropyLoss()`（内部自动包含 `LogSoftmax` 与负对数似然损失，直接接收网络输出的 logits 与真实类别索引）。
     * 优化器：`torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)`。
  3. **单轮训练流程（每个 epoch）**：
     * 切换至训练模式 `model.train()`，遍历训练集 `DataLoader`。
     * 执行标准五步：数据移至设备 $\rightarrow$ `optimizer.zero_grad()` 清空历史梯度 $\rightarrow$ `outputs = model(images)` 前向传播 $\rightarrow$ `loss = criterion(outputs, labels)` 计算损失 $\rightarrow$ `loss.backward()` 反向传播与 `optimizer.step()` 参数更新。
     * 累计并计算本轮训练平均 Loss 与训练 Accuracy。
     * 同步在评估数据集上计算本轮验证/测试 Loss 与 Accuracy（在 `model.eval()` 与 `torch.no_grad()` 下进行），并打印本轮训练进度日志。
* **目的是**：通过小批量随机梯度下降不断最小化交叉熵损失，使网络权重逐步学习到区分 $10$ 类物体的最优特征映射，并通过逐轮监控掌握模型收敛状态。
* **实现途径**：`torch.nn.CrossEntropyLoss`、`torch.optim.Adam`、`torch.optim.lr_scheduler` 及标准 PyTorch 训练循环。

### 3.2 记录训练指标与保存最优权重

* **干什么**：在训练全程记录每轮的 Loss 与 Accuracy 变化历史，自动筛选并保存泛化表现最好的模型权重。
* **怎么做**：
  1. 初始化记录容器（如字典或列表：`history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}`），每完成一个 epoch 即追加写入当前数值（同时保存为 JSON 或 CSV 文件备用）。
  2. 维护变量 `best_acc = 0.0` 与 `best_epoch = 0`；每轮评估结束后，若当前准确率高于 `best_acc`，则更新 `best_acc` 并立即将当前模型参数保存至 `./outputs/best_cnn_cifar10.pth`。
* **目的是**：防止训练后期可能出现的过拟合或学习率波动导致最终权重性能退化，确保最终评估阶段调用的是整个训练周期内表现最佳的模型参数。
* **实现途径**：Python 列表/字典、`json` 模块、`torch.save(model.state_dict(), save_path)`。

---

## 第四阶段：模型评估

### 4.1 计算测试集四大评估指标

* **干什么**：加载保存的最优模型权重，在 $10,000$ 张测试集上完成推理，计算并输出**准确率（Accuracy）、精确率（Precision）、召回率（Recall）、F1 值**。
* **怎么做**：
  1. 重新实例化 CNN 模型并通过 `model.load_state_dict(torch.load('./outputs/best_cnn_cifar10.pth'))` 加载最优权重。
  2. 设置 `model.eval()` 并包裹在 `with torch.no_grad():` 中遍历测试集 `DataLoader`，收集全部样本的真实标签 `y_true` 和预测类别 `y_pred = outputs.argmax(dim=1)`。
  3. 计算并打印以下指标：
     * 整体准确率（`Accuracy`）。
     * 宏平均（`macro`）与加权平均（`weighted`）的精确率（`Precision`）、召回率（`Recall`）与 `F1` 值。
     * 各类别（$10$ 类）各自的 `Precision`、`Recall`、`F1` 详细分类报告（`classification_report`）。
* **目的是**：突破单一准确率指标的局限，从整体正确率、单类查准能力、查全能力以及综合平衡度四个维度全面评估模型的泛化性能。
* **实现途径**：`torch.no_grad()`、`sklearn.metrics.accuracy_score`、`precision_score`、`recall_score`、`f1_score`、`classification_report`。

### 4.2 绘制 Loss/Accuracy 曲线与输出混淆矩阵

* **干什么**：绘制训练过程中的 Loss 变化曲线和 Accuracy 变化曲线，并计算、绘制测试集 $10 \times 10$ 混淆矩阵。
* **怎么做**：
  1. **Loss 曲线与 Accuracy 曲线**：
     * 以 `epoch`（$1 \sim N$）为横轴，绘制包含训练曲线与验证/测试曲线的双折线图（左图为 Loss 曲线，右图为 Accuracy 曲线），用竖线或星号标出 `best_epoch` 位置。
     * 添加标题、坐标轴标签、图例与网格线，保存为高分辨率图片 `./outputs/training_curves.png`。
  2. **混淆矩阵（Confusion Matrix）**：
     * 基于 `y_true` 与 `y_pred` 计算 $10 \times 10$ 整数计数混淆矩阵，在终端打印矩阵数值并保存为文本/CSV。
     * 绘制带 $10$ 个英文/中文类别标签刻度与单元格具体数值标注的混淆矩阵热力图，保存为 `./outputs/confusion_matrix.png`。
* **目的是**：
  * 通过 Loss 与 Accuracy 曲线直观判断模型是否充分收敛、是否存在欠拟合或过拟合。
  * 通过混淆矩阵定位具体易混淆类别对（例如“猫 `cat`”与“狗 `dog`”、“汽车 `automobile`”与“卡车 `truck`”），为后续网络结构或数据增强改进提供直接依据。
* **实现途径**：`sklearn.metrics.confusion_matrix`、`ConfusionMatrixDisplay`、`matplotlib.pyplot`（配合 `plt.savefig(..., dpi=300, bbox_inches='tight')`）。

---

## 附：发给 GPT 的直接执行提示词

你可以把本 Markdown 文件完整发送给 GPT，并附上下面这段指令：

```text
请严格按照上述《数字图像处理》实验一实施方案，使用 Python + PyTorch 编写完整、可直接运行、注释清晰的代码。

具体要求：
1. 代码结构清晰，可将全部逻辑整合为一个运行无报错的主脚本（或清晰的模块化文件并说明运行命令）。
2. 完整实现方案中的所有步骤：
   - 加载 CIFAR-10 并打印训练集/测试集各类样本数及均衡性验证结果；
   - 实现规定的像素归一化与训练集数据增强（随机裁剪、随机水平翻转、颜色抖动），测试集不做随机增强；
   - 搭建方案中的简易 CNN 网络并打印网络结构与参数量；
   - 完成训练循环，记录每轮 loss 和 accuracy，自动保存最优权重到 ./outputs/best_cnn_cifar10.pth；
   - 加载最优权重在测试集上计算 Accuracy、Precision、Recall、F1（含整体及各类别详细报告）；
   - 绘制并保存 Loss/Accuracy 曲线图（./outputs/training_curves.png）和带数值标注的 10×10 混淆矩阵图（./outputs/confusion_matrix.png）。
3. 请给出依赖库清单、运行步骤说明以及预期输出示例。
```
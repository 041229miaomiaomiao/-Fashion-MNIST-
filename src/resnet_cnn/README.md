# Fashion-MNIST CNN 分类项目

这是一个基于 TensorFlow/Keras 的 CNN 图像分类项目，使用 Fashion-MNIST 数据集进行 10 类服装图像分类。

## 项目简介

- 使用 `Fashion-MNIST` 数据集的 IDX 原始文件：
  - `train-images-idx3-ubyte`
  - `train-labels-idx1-ubyte`
  - `t10k-images-idx3-ubyte`
  - `t10k-labels-idx1-ubyte`
- 代码直接读取 `data/` 目录中的原始 IDX 文件。
- 训练使用 ResNet 风格的卷积神经网络模型，适用于灰度图像输入。

## 主要功能

- 合并原始训练集与测试集后重新划分为：
  - 70% 训练集
  - 15% 验证集
  - 15% 测试集
- 数据预处理包括：
  - 重塑为 `(28, 28, 1)` 的张量格式
  - 归一化到 `[0, 1]`
- 模型训练使用 `EarlyStopping`：
  - 监控 `val_loss`
  - `patience=5`
  - 验证集性能连续 5 轮没有提升时自动停止训练
  - 使用 `restore_best_weights=True` 恢复验证集上表现最好的模型权重
- 训练完成后保存模型和可视化结果。

## 模型结构

模型采用 ResNet 风格结构，包含以下关键组件：

1. `Conv2D(32, (3, 3), padding='same')`
2. `BatchNormalization()`
3. `Activation('relu')`
4. 残差块（Residual Block），包含两个 `Conv2D(3, 3)` + `BatchNormalization` + `Activation`，并通过 `Add()` 进行捷径连接
5. `GlobalAveragePooling2D()`
6. `Dense(128, relu)`
7. `Dropout(0.5)`
8. `Dense(10, softmax)`

- 数据预处理通过将输入图像重塑为 `(28, 28, 1)`，并将像素值归一化到 `[0, 1]`，这样可以加速神经网络训练并提高数值稳定性。
- 模型采用 ResNet 风格结构，通过残差连接避免梯度消失，并帮助网络更稳定地学习深层特征。
- 卷积层（`Conv2D`）负责从图像中提取局部特征，例如边缘、纹理和简单形状。
- 批量归一化（`BatchNormalization`）用于加速训练并提高数值稳定性。
- 残差连接（`Add`）将主路径与捷径路径相加，使网络更易于训练。
- 全局平均池化（`GlobalAveragePooling2D`）替代传统展平层，减少参数量并保留空间信息。
- 全连接层（`Dense`）负责将提取到的特征映射到 10 个类别上。
- `Dropout(0.5)` 用于随机丢弃一半神经元，防止过拟合，使模型更具泛化能力。
- 输出层使用 `softmax` 激活，将网络输出转换为 10 类概率分布。
- 训练时使用 `sparse_categorical_crossentropy` 损失函数，适用于整数标签的多分类任务。
- 早停回调 `EarlyStopping` 监控验证集损失 `val_loss`，如果验证性能连续 5 轮不再提升，则自动停止训练，并恢复验证集上表现最好的权重。

## 输出文件

训练完成后，结果将保存到 `output/` 目录：

- `sample_images.png`：训练样本图像
- `model_summary.txt`：模型结构信息
- `fashion_mnist_cnn_model.h5`：训练后的模型文件
- `training_history.xlsx`：训练历史 Excel 文件，保存每个 epoch 的训练和验证指标
- `training_history.png`：训练/验证准确率与损失曲线
- `confusion_matrix.png`：测试集混淆矩阵

## 训练结果说明

- `training_history.png` 包含训练准确率、验证准确率、训练损失和验证损失随 epoch 的变化曲线。通过这张图可以判断模型是否过拟合、欠拟合或训练是否稳定。
- `confusion_matrix.png` 显示了模型在测试集上的分类效果，每个格子的数值代表真实类别与预测类别的样本数量。
- `sample_images.png` 展示了模型训练前的随机样本，方便直观检查数据加载和标签是否对应正确。
- `fashion_mnist_cnn_model.h5` 是最终保存的模型，可以在后续加载并用于推理或部署。

以下为在测试集上得到的分类准确率：

T-shirt/top    : 0.8771 (87.71%)
Trouser        : 0.9781 (97.81%)
Pullover       : 0.8695 (86.95%)
Dress          : 0.9000 (90.00%)
Coat           : 0.8610 (86.10%)
Sandal         : 0.9762 (97.62%)
Shirt          : 0.6543 (65.43%)
Sneaker        : 0.9657 (96.57%)
Bag            : 0.9752 (97.52%)
Ankle boot     : 0.9400 (94.00%)

## 运行方法

1. 安装依赖：

```powershell
cd /d e:\机器学习\fashion_mnist_cnn
python -m pip install -r requirements.txt
```

2. 运行训练脚本：

```powershell
python train.py
```

## 依赖

- tensorflow
- numpy
- matplotlib
- seaborn
- scikit-learn

依赖在 `requirements.txt` 中已列出。

## 代码说明

- `train.py`：主训练脚本，负责数据加载、划分、模型训练、评估和结果保存。
- `utils.py`：辅助函数，包括数据加载、图像绘制、混淆矩阵和分类报告打印。
- `config.py`：配置参数，例如批次大小、训练轮数、验证比例和类别名。
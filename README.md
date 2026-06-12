# Fashion MNIST 服饰图像多分类

## 项目简介
这是我们机器学习第九组的大作业，基于 Fashion MNIST 数据集的服饰图像多分类任务，实现从传统机器学习到深度学习的 8 种分类算法，并进行系统性对比分析。

## 算法清单
| 序号 | 算法 | 测试准确率 |
|:----:|------|:----------:|
| 1 | 逻辑回归 (Logistic Regression) | 84.26% |
| 2 | K近邻 (KNN, k=5) | 85.54% |
| 3 | 随机森林 (Random Forest) | 87.74% |
| 4 | 线性支持向量机 (Linear SVM) | 83.95% |
| 5 | SGD逻辑回归 (SGD Logistic) | 82.95% |
| 6 | 多层感知机 (MLP) | ~88.0% |
| 7 | 卷积神经网络 (CNN) | **91.88%** |
| 8 | 残差卷积网络 (ResNet+CNN) | **92.00%** |

## 目录结构
```
├── README.md
├── requirements.txt          # Python 依赖
├── data/                     # Fashion MNIST 原始数据集 (.gz)
├── notebooks/                # 统一入口 Notebook
│   ├── 2.项目代码（Group_9）.ipynb   # 综合项目代码
│   └── 2.项目代码（Group_9）.html    # HTML 导出
└── src/                      # 各算法独立代码
    ├── logistic_regression/  # 逻辑回归
    ├── knn/                  # K近邻
    ├── random_forest/        # 随机森林
    ├── svm_sgd_comparison/   # SVM + SGD 对比
    ├── cnn/                  # 卷积神经网络
    ├── mlp/                  # 多层感知机
    └── resnet_cnn/           # 残差卷积网络
```

## 快速开始
```bash
pip install -r requirements.txt
jupyter notebook notebooks/2.项目代码（Group_9）.ipynb
```
运行 CNN / MLP / ResNet 需要额外安装 TensorFlow。

## 环境要求
- Python >= 3.8
- 基础依赖：numpy, matplotlib, seaborn, scikit-learn, jupyter
- 深度学习：tensorflow >= 2.10（可选）

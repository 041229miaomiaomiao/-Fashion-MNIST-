# utils.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import config
import os

# 解决中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_fashion_mnist(data_dir):
    """加载 Fashion-MNIST 的 idx 格式文件"""
    def load_images(filename):
        with open(filename, 'rb') as f:
            magic = int.from_bytes(f.read(4), 'big')
            num_images = int.from_bytes(f.read(4), 'big')
            rows = int.from_bytes(f.read(4), 'big')
            cols = int.from_bytes(f.read(4), 'big')
            images = np.frombuffer(f.read(), dtype=np.uint8)
            images = images.reshape(num_images, rows, cols)
        return images

    def load_labels(filename):
        with open(filename, 'rb') as f:
            magic = int.from_bytes(f.read(4), 'big')
            num_labels = int.from_bytes(f.read(4), 'big')
            labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

    train_images = load_images(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    train_labels = load_labels(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    test_images = load_images(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    test_labels = load_labels(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))

    return (train_images, train_labels), (test_images, test_labels)


def load_local_mnist(data_dir, normalize=True, flatten=False):
    """兼容 .gz 和未压缩 idx 文件的加载器。

    返回格式与 load_fashion_mnist 相同： (train_images, train_labels), (test_images, test_labels)
    参数：
      - normalize: 是否将像素值缩放到 [0,1]
      - flatten: 是否将图片展平为 (N, 784)，否则返回 (N, 28, 28)
    """
    import gzip

    def _read(kind):
        # 尝试优先读取 .gz 文件，如果不存在则读取未压缩文件
        labels_gz = os.path.join(data_dir, f"{kind}-labels-idx1-ubyte.gz")
        images_gz = os.path.join(data_dir, f"{kind}-images-idx3-ubyte.gz")
        labels_raw = None
        images_raw = None

        if os.path.exists(labels_gz) and os.path.exists(images_gz):
            with gzip.open(labels_gz, 'rb') as lbpath:
                labels_raw = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
            with gzip.open(images_gz, 'rb') as imgpath:
                images_raw = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16)
        else:
            labels_path = os.path.join(data_dir, f"{kind}-labels-idx1-ubyte")
            images_path = os.path.join(data_dir, f"{kind}-images-idx3-ubyte")
            with open(labels_path, 'rb') as lbpath:
                labels_raw = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
            with open(images_path, 'rb') as imgpath:
                images_raw = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16)

        if flatten:
            images = images_raw.reshape(len(labels_raw), 784)
        else:
            images = images_raw.reshape(len(labels_raw), 28, 28)

        return images, labels_raw

    train_images, train_labels = _read('train')
    test_images, test_labels = _read('t10k')

    if normalize:
        train_images = train_images.astype('float32') / 255.0
        test_images = test_images.astype('float32') / 255.0

    return (train_images, train_labels), (test_images, test_labels)

def plot_sample_images(images, labels, save_path=None, num_samples=10):
    """显示样本图片"""
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    for i in range(num_samples):
        idx = np.random.randint(0, len(images))
        axes[i].imshow(images[idx], cmap='gray')
        axes[i].set_title(config.CLASS_NAMES[labels[idx]], fontsize=10)
        axes[i].axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 样本图片已保存: {save_path}")
    plt.show()

def plot_training_history(history, save_path=None):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history['accuracy'], label='训练准确率', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='验证准确率', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('模型准确率曲线')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history['loss'], label='训练损失', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='验证损失', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('模型损失曲线')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 训练曲线已保存: {save_path}")
    plt.show()

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=config.CLASS_NAMES,
                yticklabels=config.CLASS_NAMES)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.xticks(rotation=45, ha='right')
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 混淆矩阵已保存: {save_path}")
    plt.show()

def print_classification_report(y_true, y_pred):
    """打印分类报告"""
    print("\n" + "="*70)
    print("分类报告")
    print("="*70)
    print(classification_report(y_true, y_pred, target_names=config.CLASS_NAMES))

    # 各类别准确率
    print("\n各类别准确率:")
    print("-"*40)
    cm = confusion_matrix(y_true, y_pred)
    for i, name in enumerate(config.CLASS_NAMES):
        acc = cm[i, i] / cm[i].sum()
        print(f"{name:15s}: {acc:.4f} ({acc*100:.2f}%)")
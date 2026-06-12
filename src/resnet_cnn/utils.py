# utils.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import config
import os

# 解决 matplotlib 中文乱码问题
def _set_chinese_font():
    zh_fonts = ['SimHei', 'Microsoft YaHei', 'Microsoft YaHei UI', 'PingFang SC', 'STSong', 'WenQuanYi Zen Hei']
    available_fonts = {f.name for f in font_manager.fontManager.ttflist}
    for font in zh_fonts:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font]
            break
    plt.rcParams['axes.unicode_minus'] = False

_set_chinese_font()


def load_fashion_mnist(data_dir):
    """加载 Fashion-MNIST 的 idx 格式文件"""
    def ensure_file_exists(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"数据文件不存在: {path}\n请检查 data 目录下是否包含 Fashion-MNIST 的 idx 文件。")

    def load_images(filename):
        ensure_file_exists(filename)
        with open(filename, 'rb') as f:
            magic = int.from_bytes(f.read(4), 'big')
            num_images = int.from_bytes(f.read(4), 'big')
            rows = int.from_bytes(f.read(4), 'big')
            cols = int.from_bytes(f.read(4), 'big')
            images = np.frombuffer(f.read(), dtype=np.uint8)
            images = images.reshape(num_images, rows, cols)
        return images
    
    def load_labels(filename):
        ensure_file_exists(filename)
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

def plot_sample_images(images, labels, save_path=None):
    """显示样本图片"""
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(10):
        idx = np.random.randint(0, len(images))
        axes[i].imshow(images[idx], cmap='gray')
        axes[i].set_title(config.CLASS_NAMES[labels[idx]], fontsize=10)
        axes[i].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 样本图片已保存: {save_path}")
    plt.show()
    plt.close(fig)

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
    plt.close(fig)

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
    plt.close()

def save_training_history_to_excel(history, save_path=None):
    """将训练历史保存为 Excel 文件"""
    history_dict = history.history
    df = pd.DataFrame({
        'epoch': list(range(1, len(history_dict.get('loss', [])) + 1)),
        'train_loss': history_dict.get('loss', []),
        'val_loss': history_dict.get('val_loss', []),
        'train_accuracy': history_dict.get('accuracy', []),
        'val_accuracy': history_dict.get('val_accuracy', [])
    })
    if save_path:
        df.to_excel(save_path, index=False, engine='openpyxl')
        print(f"✅ 训练历史已保存为 Excel: {save_path}")
    return df

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
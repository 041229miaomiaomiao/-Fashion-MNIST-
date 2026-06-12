# =========================================================================
# 队长个人报告专属：多分类算法对比研究 (Logistic Regression vs. KNN vs. SVM)
# 说明：本脚本用于队长个人大作业报告的研究与撰写。
#       同时实现了逻辑回归、K近邻、支持向量机，并对比三者的准确率、运行耗时。
#       生成两张专业图表：各模型性能对比图、三模型混淆矩阵对比图。
# 数据路径：默认从 ./data/ 读取离线二进制文件。
# =========================================================================

import os
import gzip
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.exceptions import ConvergenceWarning

# 忽略不影响结果的收敛警告与未来兼容警告
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

print("========================================")
print("第一步：读取本地离线数据 (./data/)")
print("========================================")

def load_mnist(path, kind='train'):
    """读取 Fashion MNIST 离线 gzip 文件的函数"""
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte.gz')
    
    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)
    return images, labels

try:
    data_path = './data'
    X_train_raw, y_train = load_mnist(data_path, kind='train')
    X_test_raw, y_test = load_mnist(data_path, kind='t10k')
    print("✅ 数据读取成功！")
except Exception as e:
    print(f"❌ 读取数据失败，请确保 ./data 目录下有四个 .gz 文件！\n详细报错: {e}")
    exit()

# 归一化 (对距离度量算法 KNN 和优化求解算法 LR/SVM 均至关重要)
X_train = X_train_raw / 255.0
X_test = X_test_raw / 255.0
print(f"训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}\n")

# =========================================================================
# 核心设置：如果您的电脑运行速度慢，可以将下方的 SAMPLE_SIZE 设为较小的值进行快速测试（如 10000）
# 使用全量数据时，将下方的使用样本设为 None。
# 为确保运行顺畅，我们默认采用全部训练数据。
# =========================================================================
SAMPLE_SIZE = None 

if SAMPLE_SIZE is not None:
    print(f"⚠️ 注意：已启用子样本模式，正在抽取 {SAMPLE_SIZE} 个样本进行训练以加速计算...")
    np.random.seed(42)
    indices = np.random.choice(len(X_train), SAMPLE_SIZE, replace=False)
    X_train_run, y_train_run = X_train[indices], y_train[indices]
else:
    X_train_run, y_train_run = X_train, y_train

results = {}

# =========================================================================
# 模型 1：逻辑回归 (Logistic Regression)
# =========================================================================
print("========================================")
print("模型 1/3：正在训练 逻辑回归 (Logistic Regression) ...")
print("========================================")
start_time = time.time()
lr_model = LogisticRegression(max_iter=500, random_state=42)
lr_model.fit(X_train_run, y_train_run)
lr_train_time = time.time() - start_time

print("正在进行预测...")
pred_start = time.time()
y_pred_lr = lr_model.predict(X_test)
lr_pred_time = time.time() - pred_start

acc_lr = accuracy_score(y_test, y_pred_lr)
results['Logistic Regression'] = {
    'acc': acc_lr,
    'train_time': lr_train_time,
    'pred_time': lr_pred_time,
    'y_pred': y_pred_lr
}
print(f"✅ 逻辑回归训练完成！准确率: {acc_lr*100:.2f}%, 训练耗时: {lr_train_time:.2f} 秒\n")


# =========================================================================
# 模型 2：K近邻 (K-Nearest Neighbors, KNN)
# =========================================================================
print("========================================")
print("模型 2/3：正在训练 K近邻 (KNN) ...")
print("========================================")
# 针对图像分类，我们选择 k=5，使用默认欧氏距离
start_time = time.time()
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_run, y_train_run)
knn_train_time = time.time() - start_time

print("正在进行预测 (KNN 预测较慢，请耐心等候大约 1 分钟)...")
pred_start = time.time()
y_pred_knn = knn_model.predict(X_test)
knn_pred_time = time.time() - pred_start

acc_knn = accuracy_score(y_test, y_pred_knn)
results['KNN (k=5)'] = {
    'acc': acc_knn,
    'train_time': knn_train_time,
    'pred_time': knn_pred_time,
    'y_pred': y_pred_knn
}
print(f"✅ KNN 训练与预测完成！准确率: {acc_knn*100:.2f}%, 预测耗时: {knn_pred_time:.2f} 秒\n")


# =========================================================================
# 模型 3：支持向量机 (Linear SVM)
# =========================================================================
print("========================================")
print("模型 3/3：正在训练 支持向量机 (Linear SVM) ...")
print("========================================")
# 使用 LinearSVC 并开启双重公式(dual=False)以在样本数多时加速收敛
start_time = time.time()
svm_model = LinearSVC(dual=False, random_state=42, tol=1e-3, max_iter=1000)
svm_model.fit(X_train_run, y_train_run)
svm_train_time = time.time() - start_time

print("正在进行预测...")
pred_start = time.time()
y_pred_svm = svm_model.predict(X_test)
svm_pred_time = time.time() - pred_start

acc_svm = accuracy_score(y_test, y_pred_svm)
results['Linear SVM'] = {
    'acc': acc_svm,
    'train_time': svm_train_time,
    'pred_time': svm_pred_time,
    'y_pred': y_pred_svm
}
print(f"✅ SVM 训练完成！准确率: {acc_svm*100:.2f}%, 训练耗时: {svm_train_time:.2f} 秒\n")


# =========================================================================
# 评估报告输出与图表生成
# =========================================================================
print("========================================")
print("第四步：生成对比分析报告与图表")
print("========================================")

# 打印详细分类报告
for name, data in results.items():
    print(f"【{name} 详细分类报告】:")
    print(classification_report(y_test, data['y_pred']))
    print("-" * 50)

# 动态检测 CJK 中文字体，若无则自动切换为英文以避免乱码及大量警告
import matplotlib.font_manager as fm
cjk_fonts = ['SimHei', 'Microsoft YaHei', 'STXihei', 'SimSun', 'WenQuanYi Micro Hei', 'Droid Sans Fallback']
available_fonts = [f.name for f in fm.fontManager.ttflist]
cjk_font_to_use = None
for f in cjk_fonts:
    if f in available_fonts:
        cjk_font_to_use = f
        break

if cjk_font_to_use:
    plt.rcParams['font.sans-serif'] = [cjk_font_to_use]
    plt.rcParams['axes.unicode_minus'] = False
    
    label_acc_title = '三算法准确率 (Accuracy) 对比'
    label_time_title = '三算法耗时对比 (训练 vs 预测)'
    label_acc_y = '准确率 (%)'
    label_time_y = '耗时 (秒)'
    label_train = '训练耗时'
    label_pred = '预测/推理耗时'
    
    class_names = ['T恤/上衣', '裤子', '套头衫', '连衣裙', '外套', 
                   '凉鞋', '衬衫', '运动鞋', '包', '短靴']
    cf_title = '{} - 混淆矩阵'
    xlabel_txt = '模型预测类别'
    ylabel_txt = '真实类别'
else:
    label_acc_title = 'Accuracy Comparison of 3 Algorithms'
    label_time_title = 'Time Cost Comparison (Train vs. Predict)'
    label_acc_y = 'Accuracy (%)'
    label_time_y = 'Time (Seconds)'
    label_train = 'Train Time'
    label_pred = 'Inference/Predict Time'
    
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    cf_title = '{} - Confusion Matrix'
    xlabel_txt = 'Predicted Class'
    ylabel_txt = 'True Class'

# 图表 1: 准确率与耗时对比图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 准确率柱状图
model_names = list(results.keys())
accuracies = [results[m]['acc'] * 100 for m in model_names]
bars = ax1.bar(model_names, accuracies, color=['#4C72B0', '#55A868', '#C44E52'], width=0.5)
ax1.set_title(label_acc_title, fontsize=14, fontweight='bold')
ax1.set_ylabel(label_acc_y, fontsize=12)
ax1.set_ylim(70, 100)
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{height:.2f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# 耗时对比条形图 (并排柱状图)
x = np.arange(len(model_names))
width = 0.35
train_times = [results[m]['train_time'] for m in model_names]
pred_times = [results[m]['pred_time'] for m in model_names]

rects1 = ax2.bar(x - width/2, train_times, width, label=label_train, color='#DD8452')
rects2 = ax2.bar(x + width/2, pred_times, width, label=label_pred, color='#937860')
ax2.set_title(label_time_title, fontsize=14, fontweight='bold')
ax2.set_ylabel(label_time_y, fontsize=12)
ax2.set_xticks(x)
ax2.set_xticklabels(model_names)
ax2.legend(fontsize=11)

# 为耗时图表添加数值标签
def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., height + 0.1, f'{height:.2f}s', ha='center', va='bottom', fontsize=9)

autolabel(rects1, ax2)
autolabel(rects2, ax2)

plt.tight_layout()
plt.savefig("01_三算法性能对比图.png", dpi=300)
plt.close()

# 图表 2: 三模型混淆矩阵并排对比图 (极具学术报告质感)
fig, axes = plt.subplots(1, 3, figsize=(22, 6))

for idx, (name, data) in enumerate(results.items()):
    cm = confusion_matrix(y_test, data['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=class_names, yticklabels=class_names, ax=axes[idx])
    axes[idx].set_title(cf_title.format(name), fontsize=13, fontweight='bold')
    axes[idx].set_xlabel(xlabel_txt, fontsize=10)
    axes[idx].set_ylabel(ylabel_txt, fontsize=10)
    axes[idx].set_xticklabels(class_names, rotation=45, ha='right')

plt.tight_layout()
plt.savefig("02_三算法混淆矩阵对比图.png", dpi=300)
plt.close()

print("\n========================================")
print("🎉 所有分析已完成！已为您生成两张精美图表：")
print("1. [01_三算法性能对比图.png] - 展示模型准确率与运行速度")
print("2. [02_三算法混淆矩阵对比图.png] - 横向对比各算法对不同品类服装的识别误差")
print("您可将终端打印出的分类报告与生成的图片直接插入您的个人大作业报告中！")
print("========================================")

# =========================================================================
# 大作业算法 1：逻辑回归 (Logistic Regression) - 队长详细版
# 说明：此版本专为队长撰写个人报告定制。
#       包含了详细的数据读取、模型训练、耗时统计，以及额外的混淆矩阵可视化图表生成。
# 数据路径：默认从 ../output2/data/ 读取离线二进制文件。
# =========================================================================

import os
import gzip
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import warnings
from sklearn.exceptions import ConvergenceWarning

# 忽略不影响结果的收敛警告与未来兼容警告
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


print("========================================")
print("第一步：读取本地离线数据 (../output2/data/)")
print("========================================")

def load_mnist(path, kind='train'):
    """读取 Fashion MNIST 离线 gzip 文件的函数"""
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte.gz')
    
    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)
    with gzip.open(images_path, 'rb') as imgpath:
        # 直接展平为 784 维
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)
    return images, labels

try:
    # 数据已经帮您搬到了当前项目的 data 文件夹下，方便 PyCharm 直接读取
    data_path = './data'
    X_train_raw, y_train = load_mnist(data_path, kind='train')
    X_test_raw, y_test = load_mnist(data_path, kind='t10k')
    print("✅ 数据读取成功！")
except Exception as e:
    print(f"❌ 读取数据失败，请确保 ../output2/data 目录下有那四个 .gz 文件！\n详细报错: {e}")
    exit()

# 归一化 (将像素值缩小到 0-1 之间，对逻辑回归收敛极度重要)
X_train = X_train_raw / 255.0
X_test = X_test_raw / 255.0
print(f"训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}\n")


print("========================================")
print("第二步：训练逻辑回归模型")
print("========================================")
start_time = time.time()

# 初始化逻辑回归模型
# 参数说明 (方便您写进报告):
# - max_iter=500: 最大迭代次数，防止不收敛
lr_model = LogisticRegression(max_iter=500, random_state=42)

# 开始训练
print("模型学习中，请稍候...")
lr_model.fit(X_train, y_train)
train_time = time.time() - start_time
print(f"✅ 训练完成！耗时: {train_time:.2f} 秒\n")


print("========================================")
print("第三步：模型预测与结果评估")
print("========================================")
y_pred = lr_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"🎯 测试集最终准确率 (Accuracy): {accuracy * 100:.2f}%\n")

# 打印详细的分类报告（每个类别的精确率、召回率）
print("【详细分类报告】:")
print(classification_report(y_test, y_pred))


print("========================================")
print("第四步：生成混淆矩阵可视化图表 (报告加分项)")
print("========================================")
# 画一张混淆矩阵热力图，放在报告里极其好看
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)

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
    class_names = ['T恤/上衣', '裤子', '套头衫', '连衣裙', '外套', 
                   '凉鞋', '衬衫', '运动鞋', '包', '短靴']
    title_text = '逻辑回归 (Logistic Regression) - 混淆矩阵'
    xlabel_text = '模型预测类别'
    ylabel_text = '真实类别'
else:
    # 英文标签，100% 跨平台兼容，无任何警告，学术报告更常用
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    title_text = 'Logistic Regression - Confusion Matrix'
    xlabel_text = 'Predicted Class'
    ylabel_text = 'True Class'

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title(title_text, fontsize=15)
plt.xlabel(xlabel_text, fontsize=12)
plt.ylabel(ylabel_text, fontsize=12)
plt.xticks(rotation=45)

# 保存图片到当前 output1 目录下
plt.tight_layout()
plt.savefig("逻辑回归_混淆矩阵结果图.png", dpi=300)
print("✅ 图表生成成功！已保存为 '逻辑回归_混淆矩阵结果图.png'，请直接插入您的个人大作业报告中！")

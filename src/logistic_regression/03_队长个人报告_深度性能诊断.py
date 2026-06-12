# =========================================================================
# 队长个人报告专属（加分研究项）：多算法深度性能诊断与错分分析
# 说明：本脚本为队长的个人报告提供更深入的数据诊断和可视化图表。
#       1. 绘制各类别（T恤、裤子等 10 类）在三算法下的 F1-Score 条形对比图，诊断难分类别。
#       2. 绘制三模型的宏观平均 ROC 曲线与 AUC 值，进行学术性模型评估。
#       3. 提取并可视化逻辑回归模型中的经典错分样本，展示 True vs. Pred 对比。
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
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.exceptions import ConvergenceWarning

# 忽略不影响结果的警告
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

print("========================================")
print("第一步：加载并处理离线 Fashion MNIST 数据")
print("========================================")

def load_mnist(path, kind='train'):
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
    print("✅ 数据加载成功！")
except Exception as e:
    print(f"❌ 数据加载失败，请确保 ./data 目录下有四个 .gz 文件！\n详细报错: {e}")
    exit()

# 归一化
X_train = X_train_raw / 255.0
X_test = X_test_raw / 255.0

# 二值化标签（用于 ROC 曲线计算）
y_test_bin = label_binarize(y_test, classes=list(range(10)))

results = {}

# =========================================================================
# 模型训练与预测分数获取
# =========================================================================
print("\n========================================")
# 1. 逻辑回归
print("正在训练 逻辑回归模型...")
lr = LogisticRegression(max_iter=500, random_state=42)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
y_score_lr = lr.predict_proba(X_test)
results['Logistic Regression'] = {'y_pred': y_pred_lr, 'y_score': y_score_lr}

# 2. KNN
print("正在训练 KNN 模型 (请稍候)...")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
y_score_knn = knn.predict_proba(X_test)
results['KNN (k=5)'] = {'y_pred': y_pred_knn, 'y_score': y_score_knn}

# 3. SVM
print("正在训练 Linear SVM 模型...")
svm = LinearSVC(dual=False, random_state=42, tol=1e-3, max_iter=1000)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
# LinearSVC 没有 predict_proba，使用 decision_function 作为 ROC 曲线的打分依据
y_score_svm = svm.decision_function(X_test)
results['Linear SVM'] = {'y_pred': y_pred_svm, 'y_score': y_score_svm}

print("✅ 所有模型训练与打分完成！")

# =========================================================================
# 可视化字体检测与语言切换
# =========================================================================
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
    
    f1_title = '三算法各类别 F1-Score 诊断对比'
    f1_xlabel = '服装类别'
    f1_ylabel = 'F1-Score'
    
    roc_title = '三算法宏观平均 ROC 曲线对比 (Macro ROC)'
    roc_xlabel = '假阳性率 (False Positive Rate)'
    roc_ylabel = '真阳性率 (True Positive Rate)'
    
    err_title = '逻辑回归典型错分样本诊断 (True: 真实 | LR: 逻辑预测 | KNN: KNN预测 | SVM: SVM预测)'
else:
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    f1_title = 'Class-wise F1-Score Comparison of 3 Algorithms'
    f1_xlabel = 'Clothing Class'
    f1_ylabel = 'F1-Score'
    
    roc_title = 'Macro-average ROC Curve Comparison'
    roc_xlabel = 'False Positive Rate'
    roc_ylabel = 'True Rate (Sensitivity)'
    
    err_title = 'Error Diagnosis (True: Actual | LR: LR Pred | KNN: KNN Pred | SVM: SVM Pred)'

# =========================================================================
# 图表 3：各类别 F1-Score 条形对比图 (柱状图)
# =========================================================================
print("\n正在生成：各类别 F1-Score 诊断对比图...")
plt.figure(figsize=(14, 7))

# 提取各模型在各类别上的 F1-Score
f1_data = []
for name, data in results.items():
    report = classification_report(y_test, data['y_pred'], output_dict=True)
    for class_idx in range(10):
        f1_data.append({
            'Model': name,
            'Class': class_names[class_idx],
            'F1-Score': report[str(class_idx)]['f1-score']
        })

import pandas as pd
df_f1 = pd.DataFrame(f1_data)

sns.barplot(x='Class', y='F1-Score', hue='Model', data=df_f1, palette='Set2')
plt.title(f1_title, fontsize=15, fontweight='bold', pad=15)
plt.xlabel(f1_xlabel, fontsize=12)
plt.ylabel(f1_ylabel, fontsize=12)
plt.ylim(0.4, 1.05)
plt.xticks(rotation=30)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(loc='lower left', fontsize=11)
plt.tight_layout()
plt.savefig("03_三算法各类别F1值横向对比图.png", dpi=300)
plt.close()

# =========================================================================
# 图表 4：宏观平均 ROC 曲线对比图 (ROC & AUC)
# =========================================================================
print("正在生成：宏观平均 ROC 曲线对比图...")
plt.figure(figsize=(8, 7))

def get_macro_roc(y_test_bin, y_score):
    n_classes = 10
    fpr = dict()
    tpr = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    
    # 汇总所有的 FPR
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    # 插值得到所有的 TPR
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    macro_auc = auc(all_fpr, mean_tpr)
    return all_fpr, mean_tpr, macro_auc

colors = ['#4C72B0', '#55A868', '#C44E52']
for idx, (name, data) in enumerate(results.items()):
    fpr_val, tpr_val, auc_val = get_macro_roc(y_test_bin, data['y_score'])
    plt.plot(fpr_val, tpr_val, color=colors[idx], lw=2.5,
             label=f'{name} (AUC = {auc_val:.4f})')

plt.plot([0, 1], [0, 1], 'k--', lw=1.5)
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.title(roc_title, fontsize=14, fontweight='bold', pad=15)
plt.xlabel(roc_xlabel, fontsize=11)
plt.ylabel(roc_ylabel, fontsize=11)
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig("04_三算法宏观ROC曲线对比图.png", dpi=300)
plt.close()

# =========================================================================
# 图表 5：错分样本诊断可视化 (错分样本格点图)
# =========================================================================
print("正在生成：逻辑回归经典错分样本诊断图...")

# 找出逻辑回归预测错误，但是可能其他算法预测正确或者也错误的样本
incorrect_indices = np.where(y_pred_lr != y_test)[0]

# 选取前 10 个错分样本
select_indices = incorrect_indices[:10]

fig, axes = plt.subplots(2, 5, figsize=(15, 7.5))
fig.suptitle(err_title, fontsize=14, fontweight='bold', y=0.96)

for idx, sample_idx in enumerate(select_indices):
    ax = axes[idx // 5, idx % 5]
    
    # 还原显示图像 (从 X_test 中取数据并 reshape 还原)
    img = X_test_raw[sample_idx].reshape(28, 28)
    ax.imshow(img, cmap='gray')
    ax.axis('off')
    
    true_label = class_names[y_test[sample_idx]]
    lr_pred = class_names[y_pred_lr[sample_idx]]
    knn_pred = class_names[y_pred_knn[sample_idx]]
    svm_pred = class_names[y_pred_svm[sample_idx]]
    
    # 在子图下方标注各算法的预测值
    ax.set_title(f"True: {true_label}\nLR: {lr_pred}\nKNN: {knn_pred}\nSVM: {svm_pred}", 
                 fontsize=10, pad=8, color='red' if lr_pred != true_label else 'black')

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig("05_典型错分样本诊断网格图.png", dpi=300)
plt.close()

print("\n========================================")
print("🎉 加分分析完成！新生成了三张高级分析图表：")
print("1. [03_三算法各类别F1值横向对比图.png] - 展示模型对不同衣服品类的识别能力差异，可写进报告分析“难点类别”。")
print("2. [04_三算法宏观ROC曲线对比图.png] - 宏观平均多分类 ROC/AUC，学术性评估的硬核图表。")
print("3. [05_典型错分样本诊断网格图.png] - 展示衣服真实图案与三个模型的预测结果对比，做错分原因定性分析。")
print("========================================")

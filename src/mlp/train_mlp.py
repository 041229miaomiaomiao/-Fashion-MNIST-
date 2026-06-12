# train_mlp.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import time
import os

import config
from utils import (load_fashion_mnist, plot_sample_images,
                   plot_training_history, plot_confusion_matrix,
                   print_classification_report)

# 设置随机种子
tf.random.set_seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)

def create_mlp_model():
    """创建 MLP 模型"""
    model = Sequential([
        Input(shape=(config.IMG_SIZE, config.IMG_SIZE, 1)),
        Flatten(),   # 将 28x28 展平为 784 维向量
        Dense(config.MLP_HIDDEN_UNITS[0], activation='relu'),
        Dropout(config.MLP_DROPOUT_RATE),
        Dense(config.MLP_HIDDEN_UNITS[1], activation='relu'),
        Dropout(config.MLP_DROPOUT_RATE),
        Dense(config.MLP_HIDDEN_UNITS[2], activation='relu'),
        Dropout(config.MLP_DROPOUT_RATE),
        Dense(config.NUM_CLASSES, activation='softmax')
    ])
    return model

def main():
    print("="*70)
    print("Fashion-MNIST 服装分类 - 多层感知机 (MLP)")
    print("="*70)

    # 1. 加载原始数据
    print("\n[1/5] 加载数据...")
    (train_images, train_labels), (test_images, test_labels) = load_fashion_mnist(config.DATA_DIR)
    print(f"原始训练集: {train_images.shape[0]} 张")
    print(f"原始测试集: {test_images.shape[0]} 张")

    # 2. 合并并重新划分数据集 (70% 训练 / 15% 验证 / 15% 测试)
    print("\n[2/5] 划分数据集...")
    X_all = np.concatenate([train_images, test_images], axis=0)
    y_all = np.concatenate([train_labels, test_labels], axis=0)
    print(f"总数据量: {len(X_all)} 张")

    # 第一次划分：70% 训练，30% 临时
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_all, y_all, test_size=(1 - config.TRAIN_RATIO),
        random_state=config.RANDOM_SEED, stratify=y_all
    )
    # 第二次划分：临时集中各 50%（即各占总数 15%）
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5,
        random_state=config.RANDOM_SEED, stratify=y_temp
    )

    # 归一化并 reshape 为 (样本, 28, 28, 1)
    X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    X_val   = X_val.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    X_test  = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

    print(f"训练集: {X_train.shape[0]} 张 ({X_train.shape[0]/len(X_all)*100:.1f}%)")
    print(f"验证集: {X_val.shape[0]} 张 ({X_val.shape[0]/len(X_all)*100:.1f}%)")
    print(f"测试集: {X_test.shape[0]} 张 ({X_test.shape[0]/len(X_all)*100:.1f}%)")

    # 3. 显示样本图片
    print("\n[3/5] 显示样本图片...")
    plot_sample_images(X_train.squeeze(), y_train,
                       save_path=os.path.join(config.OUTPUT_DIR, 'sample_images.png'))

    # 4. 创建并训练 MLP 模型
    print("\n[4/5] 创建并训练 MLP 模型...")
    model = create_mlp_model()
    model.summary()

    # 保存模型结构
    with open(os.path.join(config.OUTPUT_DIR, 'model_summary.txt'), 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=config.PATIENCE,
                               restore_best_weights=True, verbose=1)

    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1
    )
    train_time = time.time() - start_time

    # 5. 评估模型
    print("\n[5/5] 评估模型...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)

    print("\n" + "="*70)
    print("📊 MLP 最终评估结果")
    print("="*70)
    print(f"⏱️ 训练耗时: {train_time:.2f} 秒")
    print(f"📈 验证集准确率: {val_acc*100:.2f}%")
    print(f"📈 测试集准确率: {test_acc*100:.2f}%")
    print(f"📉 测试集损失: {test_loss:.4f}")

    # 预测并输出详细分类报告
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    print_classification_report(y_test, y_pred)

    # 保存模型
    model_save_path = os.path.join(config.OUTPUT_DIR, 'fashion_mnist_mlp_model.h5')
    model.save(model_save_path)
    print(f"✅ 模型已保存: {model_save_path}")

    # 绘制可视化图表
    print("\n生成可视化图表...")
    plot_training_history(history,
                          save_path=os.path.join(config.OUTPUT_DIR, 'training_history.png'))
    plot_confusion_matrix(y_test, y_pred,
                          save_path=os.path.join(config.OUTPUT_DIR, 'confusion_matrix.png'))

    print("\n" + "="*70)
    print(f"✅ MLP 训练完成！所有结果已保存到 {config.OUTPUT_DIR}/")
    print("="*70)

if __name__ == "__main__":
    main()
# train.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization, Activation,
                                     Add, GlobalAveragePooling2D, Dense, Dropout)
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import time
import os

import config
from utils import (load_fashion_mnist, plot_sample_images, plot_training_history, plot_confusion_matrix,
                   print_classification_report, save_training_history_to_excel)

# 设置随机种子
tf.random.set_seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)

def residual_block(inputs, filters, stride=1):
    """构建一个残差块。"""
    x = Conv2D(filters, (3, 3), strides=stride, padding='same', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters, (3, 3), strides=1, padding='same', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)

    shortcut = inputs
    if stride != 1 or inputs.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), strides=stride, padding='same', kernel_initializer='he_normal')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


def create_resnet_model():
    """创建一个 ResNet 风格的模型。"""
    inputs = Input(shape=(28, 28, 1))
    x = Conv2D(32, (3, 3), strides=1, padding='same', kernel_initializer='he_normal')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = residual_block(x, 32, stride=1)
    x = residual_block(x, 64, stride=2)
    x = residual_block(x, 64, stride=1)

    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(config.NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs, name='ResNet_Small')
    return model

def main():
    print("="*70)
    print("Fashion-MNIST 服装分类 - CNN 卷积神经网络")
    print("="*70)
    
    # 1. 加载数据
    print("\n[1/5] 加载数据...")
    (train_images, train_labels), (test_images, test_labels) = load_fashion_mnist(config.DATA_DIR)
    print(f"原始训练集: {train_images.shape[0]} 张图片")
    print(f"原始测试集: {test_images.shape[0]} 张图片")

    # 2. 数据预处理
    print("\n[2/5] 数据预处理...")
    X_all = np.concatenate([train_images, test_images], axis=0)
    y_all = np.concatenate([train_labels, test_labels], axis=0)

    holdout_ratio = 0.30

    X_train, X_temp, y_train, y_temp = train_test_split(
        X_all, y_all, test_size=holdout_ratio,
        random_state=config.RANDOM_SEED, stratify=y_all
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5,
        random_state=config.RANDOM_SEED, stratify=y_temp
    )

    X_train = X_train.reshape(-1, 28, 28, 1).astype('float32')
    X_val = X_val.reshape(-1, 28, 28, 1).astype('float32')
    X_test = X_test.reshape(-1, 28, 28, 1).astype('float32')

    print(f"训练集: {X_train.shape[0]} 张")
    print(f"验证集: {X_val.shape[0]} 张")
    print(f"测试集: {X_test.shape[0]} 张")
    
    # 3. 创建并训练模型
    print("\n[4/5] 创建并训练 ResNet 风格模型...")
    model = create_resnet_model()
    model.summary()
    
    # 保存模型结构到文本文件
    with open(os.path.join(config.OUTPUT_DIR, 'model_summary.txt'), 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # 早停回调
    early_stop = EarlyStopping(monitor='val_loss', patience=20, 
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
    
    # 4. 评估模型
    print("\n[5/5] 评估模型...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    print("\n" + "="*70)
    print("📊 最终评估结果")
    print("="*70)
    # 训练耗时
    print(f"⏱️ 训练耗时: {train_time:.2f} 秒")
    # 验证集准确率（取训练历史中的最佳验证准确率）
    val_acc_history = history.history.get('val_accuracy') or history.history.get('val_acc')
    if val_acc_history:
        best_val_acc = max(val_acc_history)
        print(f"🔎 验证集最佳准确率: {best_val_acc*100:.2f}%")
    else:
        print("🔎 验证集准确率: 未找到训练历史中的验证准确率")

    # 测试集指标
    print(f"📈 测试集准确率: {test_acc*100:.2f}%")
    print(f"📉 测试集损失: {test_loss:.4f}")
    
    # 预测并输出分类报告
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    print_classification_report(y_test, y_pred)

    # 保存混淆矩阵为 CSV，保存分类报告为文本，方便后续查看
    cm = confusion_matrix(y_test, y_pred)
    cm_save_path = os.path.join(config.OUTPUT_DIR, 'confusion_matrix.csv')
    np.savetxt(cm_save_path, cm, fmt='%d', delimiter=',')
    print(f"✅ 混淆矩阵 CSV 已保存: {cm_save_path}")

    cls_report = classification_report(y_test, y_pred, target_names=config.CLASS_NAMES)
    report_save_path = os.path.join(config.OUTPUT_DIR, 'classification_report.txt')
    with open(report_save_path, 'w', encoding='utf-8') as f:
        f.write(cls_report)
    print(f"✅ 分类报告已保存: {report_save_path}")
    
    # 保存模型
    model_save_path = os.path.join(config.OUTPUT_DIR, 'fashion_mnist_cnn_model.h5')
    model.save(model_save_path)
    print(f"✅ 模型已保存: {model_save_path}")

    # 保存训练历史到 Excel
    excel_save_path = os.path.join(config.OUTPUT_DIR, 'training_history.xlsx')
    save_training_history_to_excel(history, save_path=excel_save_path)
    
    # 绘制可视化图表
    print("\n生成可视化图表...")
    plot_training_history(history, 
                          save_path=os.path.join(config.OUTPUT_DIR, 'training_history.png'))
    plot_confusion_matrix(y_test, y_pred,
                      save_path=os.path.join(config.OUTPUT_DIR, 'confusion_matrix.png'))
    
    print("\n" + "="*70)
    print(f"✅ 完成！所有结果已保存到 {config.OUTPUT_DIR}/")
    print("="*70)

if __name__ == "__main__":
    main()
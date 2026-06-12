# config.py
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 数据集参数
IMG_SIZE = 28
NUM_CLASSES = 10

# 类别标签
CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# 训练参数
BATCH_SIZE = 64
EPOCHS = 50          # 足够大，早停会自动提前停止
PATIENCE = 15        # 早停耐心值

# 数据划分比例
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

# MLP 模型结构
MLP_HIDDEN_UNITS = [512, 256, 128]   # 三个隐藏层
MLP_DROPOUT_RATE = 0.5
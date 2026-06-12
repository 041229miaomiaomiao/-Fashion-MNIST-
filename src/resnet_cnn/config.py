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
EPOCHS = 100
VALIDATION_SPLIT = 0.1
RANDOM_SEED = 42
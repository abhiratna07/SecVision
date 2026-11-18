import os
import torch
if torch.cuda.is_available():
    DEVICE="cuda"
else:
    DEVICE="cpu"
ROOT_DIR=os.path.dirname(os.path.abspath(__file__))
DATA_DIR=os.path.join(ROOT_DIR, "data")
CHECKPOINT_DIR=os.path.join(ROOT_DIR, "checkpoints")
RESULTS_DIR=os.path.join(ROOT_DIR, "results")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

BASELINE_MODEL_PATH=os.path.join(CHECKPOINT_DIR,"baseline_resnet.pth")
ROBUST_MODEL_PATH=os.path.join(CHECKPOINT_DIR,"robust_resnet.pth")

DATASET_NAME="CIFAR_10"
NUM_CLASSES=10
BATCH_SIZE=128
NUM_WORKERS=2
EPOCHS=30
MAX_LR=0.1
MOMENTUM=0.9
WEIGHT_DECAY=5e-4

EPSILON = 8/255
ALPHA=2/255
PGD_TRAIN_STEPS=7
PGD_EVAL_STEPS=20

EPSILON_VALUE=[0.0,2/255,4/255,8/255,16/255]

CLIP_MODEL= "ViT-B-32-quickgelu"
CLIP_PRETRAINED="openai"
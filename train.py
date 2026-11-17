import os
import config
from utils.data import cifar10_loaders,CIFAR10_CLASSES
from models.resnet import CIFAR10ResNet18
from models.clip import CLIP
from training.trainer import train_model,evaluate 

def main():
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    train_loader,test_loader=cifar10_loaders()
    print("Results For Baseline ResNet")
    resnet=CIFAR10ResNet18()
    train_model(resnet,train_loader,test_loader,save_path=config.BASELINE_MODEL_PATH)
    print("Results for Baseline CLIP")
    clip_model=CLIP(CIFAR10_CLASSES).to(config.DEVICE)
    clip_acc=evaluate(clip_model,test_loader)
    print(f"CLIP Accuracy: {clip_acc:.2f}%")

if __name__=="__main__":
    main()
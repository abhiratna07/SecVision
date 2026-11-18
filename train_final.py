import config
from utils.data import cifar10_loaders
from attacks.pgd import PGD 
from models.resnet import CIFAR10ResNet18
from training.trainer import train_model

def main():
    train_loader,test_loader=cifar10_loaders()
    model=CIFAR10ResNet18().to(config.DEVICE)
    train_pgd=PGD(
        model,
        epsilon=config.EPSILON,
        alpha=config.ALPHA,
        steps=config.PGD_TRAIN_STEPS,
        random_start=True,
    )
    train_model(
        model,
        train_loader,
        test_loader,
        save_path=config.ROBUST_MODEL_PATH,
        attack=train_pgd,
    )

if __name__=="__main__":
    main()
    

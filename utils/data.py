import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import config

CIFAR10_CLASSES=(
    "plane","car","bird","cat","deer",
    "dog","frog","horse","ship","truck"
)
def cifar10_loaders():
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.toTensor(),
    ])
    test_transform=transforms.ToTensor()
    train_set=torchvision.datasets.CIFAR10(
        root=config.DATA_DIR, train=True,download=True, transform=train_transform
    )
    test_set=torchvision.datasets.CIFAR10(
        root=config.DATA_DIR, train=False,download=True, transform=test_transform
    )
    train_loader= DataLoader(train_set,batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    test_loader= DataLoader(test_set,batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    return train_loader,test_loader

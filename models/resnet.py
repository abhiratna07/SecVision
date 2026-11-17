import torch
import torch.nn as nn
from torchvision import models
import config

CIFAR_MEAN=(0.4914,0.4822,0.4465)
CIFAR_STD=(0.2023,0.1994,0.2010)
class CIFAR10ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("mean",torch.tensor(CIFAR_MEAN).view(1,3,1,1))
        self.register_buffer("std",torch.tensor(CIFAR_STD).view(1,3,1,1))
        self.net=models.resnet18(weights=None)
        self.net.conv1=nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
        self.net.maxpool=nn.Identity()
        self.net.fc=nn.Linear(self.net.fc.in_features, config.NUM_CLASSES)

    def forward(self, x):
        x=(x-self.mean)/self.std
        return self.net(x)
    
import os
import torch
import config
from utils.data import cifar10_loaders,CIFAR10_CLASSES
from models.resnet import CIFAR10ResNet18
from models.clip import CLIP
from training.trainer import evaluate
from attacks.fgsm import FGSM
from attacks.pgd import PGD

def run(model,name,test_loader):
    print(f"{name}")
    for eps in config.EPSILON_VALUE:
        fgsm=FGSM(model, epsilon=eps)
        pgd=PGD(model,epsilon=eps,alpha=config.ALPHA,steps=config.PGD_EVAL_STEPS)
        fgsm_acc=evaluate(model,test_loader,attack=fgsm)
        pgd_acc=evaluate(model,test_loader,attack=pgd)
        print(f"eps={eps*255:4.0f}/255 | FGSM: {fgsm_acc:6.2f}% | PGD: {pgd_acc:6.2f}%")

def main():
    _,test_loader=cifar10_loaders()
    if os.path.exists(config.BASELINE_MODEL_PATH):
        resnet=CIFAR10ResNet18().to(config.DEVICE)
        resnet.load_state_dict(torch.load(config.BASELINE_MODEL_PATH,map_location=config.DEVICE,weights_only=True))
        run(resnet,"Baseline ResNet-18",test_loader)
    if os.path.exists(config.ROBUST_MODEL_PATH):
        final1=CIFAR10ResNet18().to(config.DEVICE)
        final1.load_state_dict(torch.load(config.ROBUST_MODEL_PATH,map_location=config.DEVICE,weights_only=True))
        run(final1,"Robust ResNet-18",test_loader)
    clip_model=CLIP(CIFAR10_CLASSES).to(config.DEVICE)
    run(clip_model,"CLIP",test_loader)

if __name__=="__main__":
    main()


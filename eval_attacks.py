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
        fgsm=FGSM(model, eps=eps)
        pgd=PGD(model,eps=eps,alpha=config.ALPHA,steps=config.PGD_EVAL_STEPS)
        fgsm_acc=evaluate(model,test_loader,attack=fgsm)
        pgd_acc=evaluate(model,test_loader,attack=pgd)
        print(f"eps={eps*255:4.0f}/255 | FGSM: {fgsm_acc:6.2f}% | PGD: {pgd_acc:6.2f}%")

def main():
    _,test_loader=cifar10_loaders()
    resnet=CIFAR10ResNet18().to(config.DEVICE)
    resnet.load_state_dict(torch.load(config.BASELINE_MODEL_PATH,map_location=config.DEVICE,weights_only=True))
    clip_model=CLIP(CIFAR10_CLASSES).to(config.DEVICE)
    run(resnet,"Baseline ResNet-18",test_loader)
    run(clip_model,"CLIP",test_loader)

if __name__=="__main__":
    main()


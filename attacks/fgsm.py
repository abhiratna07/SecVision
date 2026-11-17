import torch
import torch.nn as nn

class FGSM:
    def __init__(self,model,epsilon,alpha=None,random_start=False):
        self.model=model
        self.epsilon=epsilon
        self.alpha=epsilon if alpha is None else alpha
        self.random_start=random_start
        self.criterion=nn.CrossEntropyLoss()

    def generate(self,images,labels):
        if self.epsilon==0.0:
            return images.detach().clone()
        was_training=self.model.training
        self.model.eval()
        original_images=images.detach()
        x0=original_images
        if self.random_start:
            x0=torch.clamp(
                original_images+torch.empty_like(original_images).uniform_(-self.epsilon,self.epsilon),
                0.0,
                1.0,
            ) 
        x_req=x0.detach().requires_grad_(True)
        loss=self.criterion(self.model(x_req),labels)
        (grad,)=torch.autograd.grad(loss,x_req)
        adversarial_images=x0+self.alpha*grad.sign()
        delta=torch.clamp(adversarial_images-original_images,-self.epsilon,self.epsilon)
        adversarial_images=torch.clamp(original_images+delta,0.0,1.0)

        if was_training:
            self.model.train()

        return adversarial_images.detach()
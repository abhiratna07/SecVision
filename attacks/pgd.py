import torch
import torch.nn as nn

class PGD:
    def __init__(self,model,epsilon,alpha,steps,random_start=True):
        self.model=model
        self.epsilon=epsilon
        self.alpha=alpha
        self.steps=steps
        self.random_start=random_start
        self.criterion=nn.CrossEntropyLoss()
    def generate(self,images,labels):
        if self.epsilon==0.0:
            return images.detach().clone()
        was_training=self.model_training
        self.model.eval()
        original_images=images.detach()
        if self.random_start:
            adversarial_images=torch.clamp(
                original_images+torch.empty_like(original_images).uniform_(-self.epsilon,self.epsilon),
                0.0,
                1.0,
            )
        else:
            adversarial_images=original_images.clone()
        for _ in range(self.steps):
            x_req=adversarial_images.detach().requires_grad_(True)
            loss=self.criterion(self.model(x_req),labels)
            (grad,)=torch.autograd.grad(loss,x_req)
            adversarial_images=adversarial_images+self.alpha*grad.sign()
            delta=torch.clamp(adversarial_images-original_images,-self.epsilon,self.epsilon)
            adversarial_images=torch.clamp(original_images+delta,0.0,1.0)
        if was_training:
            self.model.train()

        return adversarial_images.detach()
    
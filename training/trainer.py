import torch
import torch.nn as nn
from tqdm import tqdm
import config

def evaluate(model, test_loader,attack=None):
    model.eval()
    correct, total=0,0
    for images,labels in test_loader:
        images,labels=images.to(config.DEVICE), labels.to(config.DEVICE)
        inputs=attack.generate(images,labels) if attack is not None else images
        with torch.no_grad():
            preds=model(inputs).argmax(dim=1)
        correct+=(preds==labels).sum().item()
        total+=labels.size(0)
    return 100.0*correct/total

def train_model(model, train_loader,test_loader,save_path,attack=None):
    model.to(config.DEVICE)
    criterion=nn.CrossEntropyLoss()
    optimizer=torch.optim.SGD(
        model.parameters(),
        lr=config.MAX_LR,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler=torch.optim.lr_scheduler.OneCycleLR(
        optimizer,max_lr=config.MAX_LR,epochs=config.EPOCHS,steps_per_epoch=len(train_loader)
    )
    for epoch in range(config.EPOCHS):
        model.train()
        for images,labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}",leave=False):
            images,labels=images.to(config.DEVICE), labels.to(config.DEVICE)
            inputs= attack.generate(images,labels) if attack is not None else images
            optimizer.zero_grad()
            loss=criterion(model(inputs),labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
        if (epoch+1)%5==0 or epoch==config.EPOCHS-1:
            acc=evaluate(model,test_loader)
            print(f"Epoch {epoch+1}/{config.EPOCHS}-Test Accuracy: {acc:.2f}%")
    torch.save(model.state_dict(),save_path)
    print(f"Model saved to {save_path}")

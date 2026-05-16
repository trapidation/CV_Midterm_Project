import os
os.environ["PYTHONUTF8"] = "1"
os.environ["TORCH_HOME"] = "./model_cache"
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import swanlab
from dataloader import get_dataloaders 
def run_experiment(project_name, exp_name, test_epochs, test_lr, test_weight_decay):
    swanlab.init(
        project=project_name, 
        experiment_name=exp_name, 
        config={
            "epochs": test_epochs,           
            "batch_size": 32,      
            "base_lr": test_lr,       
            "fc_lr": test_lr * 10,       
            "weight_decay": test_weight_decay   
        }
    )
    config = swanlab.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_dataloaders(batch_size=config.batch_size)
    print(f"\n正在运行目录 [{project_name}] 的实验: {exp_name}")
    model = models.resnet18(weights='DEFAULT') 
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 37)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    base_params = [p for name, p in model.named_parameters() if 'fc' not in name]
    fc_params = model.fc.parameters()
    
    optimizer = optim.Adam([
        {'params': base_params, 'lr': config.base_lr},
        {'params': fc_params, 'lr': config.fc_lr}
    ], weight_decay=config.weight_decay)

    for epoch in range(config.epochs):
        model.train() 
        running_loss = 0.0
        
        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if step % 20 == 0:
                swanlab.log({"train_step_loss": loss.item()})
                
        model.eval() 
        correct = 0
        total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        avg_val_loss = val_loss / len(test_loader)
        val_accuracy = 100 * correct / total
        print(f"[{exp_name}] 第 {epoch+1} 轮准确率: {val_accuracy:.2f}%")
        
        swanlab.log({
            "epoch": epoch + 1,
            "val_loss": avg_val_loss,
            "val_accuracy": val_accuracy
        })

    swanlab.finish()
    print(f"实验 {exp_name} 结束！\n")


if __name__ == '__main__':
    lr_list = [1e-5, 1e-4, 1e-3, 1e-2]
    for lr in lr_list:
        run_experiment(
            project_name="Task1_Test_LR", 
            exp_name=f"LR_{lr}", 
            test_epochs=5, test_lr=lr, test_weight_decay=0.0
        )
    wd_list = [0.0, 1e-4, 1e-3, 1e-2] 
    for wd in wd_list:
        run_experiment(
            project_name="Task1_Test_WeightDecay", 
            exp_name=f"WD_{wd}", 
            test_epochs=5, test_lr=1e-4, test_weight_decay=wd
        )
    epoch_list = [5, 10, 15] 
    for ep in epoch_list:
        run_experiment(
            project_name="Task1_Test_Epochs", 
            exp_name=f"Epoch_{ep}", 
            test_epochs=ep, test_lr=1e-4, test_weight_decay=1e-4
        )

import os
# --- 各种破解环境配置的咒语放在最前面 ---
os.environ["PYTHONUTF8"] = "1"
os.environ["TORCH_HOME"] = "./model_cache"  # ✨ 新加的这一行！强制让它把模型下到当前目录！
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
# 1. 换成 swanlab
import swanlab
from dataloader import get_dataloaders 

def train_model():
    # 2. 初始化换成 swanlab.init
    swanlab.init(
        project="CV_Homework_Task1", 
        experiment_name="ResNet18_NoPretrain", 
        config={
            "epochs": 5,           
            "batch_size": 32,      
            "base_lr": 1e-4,       
            "fc_lr": 1e-3          
        }
    )
    # 取参数的方式稍微改一下
    config = swanlab.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_dataloaders(batch_size=config.batch_size)

    print("正在加载预训练模型...")
    model = models.resnet18(weights=None) 
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 37)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    base_params = [p for name, p in model.named_parameters() if 'fc' not in name]
    fc_params = model.fc.parameters()
    
    optimizer = optim.Adam([
        {'params': base_params, 'lr': config.base_lr},
        {'params': fc_params, 'lr': config.fc_lr}
    ])

    print("🚀 引擎点火，开始训练...")
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
                print(f"Epoch [{epoch+1}/{config.epochs}], Step [{step}/{len(train_loader)}], Loss: {loss.item():.4f}")
                # 3. 记录日志换成 swanlab.log
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
        print(f"✅ 第 {epoch+1} 轮结束！验证集准确率: {val_accuracy:.2f}%")
        
        # 4. 记录日志换成 swanlab.log
        swanlab.log({
            "epoch": epoch + 1,
            "val_loss": avg_val_loss,
            "val_accuracy": val_accuracy
        })

if __name__ == '__main__':
    train_model()
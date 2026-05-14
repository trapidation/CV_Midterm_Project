import os
os.environ["PYTHONUTF8"] = "1"
os.environ["TORCH_HOME"] = "./model_cache"
import torch
import torch.nn as nn
import torch.optim as optim
import swanlab
import timm  # ✨ 引入大名鼎鼎的 timm 模型库
from dataloader import get_dataloaders 

def train_vit_model():
    swanlab.init(
        project="CV_Homework_Task1", 
        experiment_name="ViT_Tiny_Pretrain", # 实验名改为 ViT
        config={
            "epochs": 5,           
            "batch_size": 32,      
            "base_lr": 1e-4,       
            "fc_lr": 1e-3,
            "model_name": "vit_tiny_patch16_224" # 记录一下使用的模型名
        }
    )
    config = swanlab.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_dataloaders(batch_size=config.batch_size)

    # ==========================================
    # 🌟 核心变化：一句话加载 ViT-Tiny 并直接改好输出层
    # ==========================================
    print("正在召唤前沿的 Vision Transformer (ViT-Tiny) ...")
    # pretrained=True 自动下载预训练权重，num_classes=37 直接把分类头改成宠物的37类！
    model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=37)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    
    # 区分参数：在 ViT 中，最后的分类层名字叫 'head' 而不是 'fc'
    base_params = [p for name, p in model.named_parameters() if 'head' not in name]
    fc_params = model.head.parameters()
    
    optimizer = optim.Adam([
        {'params': base_params, 'lr': config.base_lr},
        {'params': fc_params, 'lr': config.fc_lr}
    ])

    print("🚀 ViT 引擎点火，开始训练...")
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
                swanlab.log({"train_step_loss": loss.item()})
                
        # 验证集评估
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
        print(f"✅ 第 {epoch+1} 轮结束！ViT 模型准确率: {val_accuracy:.2f}%")
        
        swanlab.log({
            "epoch": epoch + 1,
            "val_loss": avg_val_loss,
            "val_accuracy": val_accuracy
        })
    torch.save(model.state_dict(), 'vit_tiny_best.pth')
    print("💾 ViT 模型权重已成功保存到本地：vit_tiny_best.pth")
    swanlab.finish()
    print("🎉 ViT 训练大功告成！")

if __name__ == '__main__':
    train_vit_model()
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import swanlab
import os
from tqdm import tqdm

from dataset import get_dataloaders
from unet import UNet
from loss import DiceLoss, CombinedLoss

# 计算 mIoU 的辅助函数
def compute_miou(preds, targets, num_classes=3):
    preds = torch.argmax(preds, dim=1)
    ious = []
    for cls in range(num_classes):
        pred_inds = preds == cls
        target_inds = targets == cls
        intersection = (pred_inds[target_inds]).long().sum().item()
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
        if union == 0:
            ious.append(float('nan'))  # 忽略该类别没有出现的情况
        else:
            ious.append(float(intersection) / float(max(union, 1)))
    # 过滤掉 nan 后求平均
    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    return sum(valid_ious) / len(valid_ious) if valid_ious else 0.0

def train(args):
    # 1. 初始化 Swanlab
    swanlab.init(
        project="CV-Task3-Segmentation",
        experiment_name=f"unet-{args.loss_type}-loss",
        config=vars(args)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. 准备数据、模型和优化器
    train_loader, val_loader = get_dataloaders(batch_size=args.batch_size)
    model = UNet(in_channels=3, out_classes=3).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 3. 选择损失函数配置
    if args.loss_type == "ce":
        criterion = nn.CrossEntropyLoss()
    elif args.loss_type == "dice":
        criterion = DiceLoss()
    elif args.loss_type == "combined":
        criterion = CombinedLoss()
    else:
        raise ValueError("Unknown loss type")

    # 4. 训练循环
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)

        # 5. 验证循环
        model.eval()
        val_loss = 0.0
        total_miou = 0.0
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val]"):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                # 计算 mIoU
                total_miou += compute_miou(outputs, masks)

        avg_val_loss = val_loss / len(val_loader)
        avg_miou = total_miou / len(val_loader)

        # 6. 使用 Swanlab 记录数据
        swanlab.log({
            "Train/Loss": avg_train_loss,
            "Val/Loss": avg_val_loss,
            "Val/mIoU": avg_miou
        }, step=epoch)

        print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val mIoU: {avg_miou:.4f}")

    # 7. 保存最终模型权重
    os.makedirs('weights', exist_ok=True)
    save_path = f"weights/unet_{args.loss_type}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Training completed. Model saved to {save_path}")

if __name__ == "__main__":
    import numpy as np # mIoU中用到了np.isnan
    parser = argparse.ArgumentParser(description="Task 3: Segmentation Training")
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--loss_type', type=str, required=True, choices=['ce', 'dice', 'combined'], 
                        help='Choose loss function: ce, dice, or combined')
    
    args = parser.parse_args()
    train(args)
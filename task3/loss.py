import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        logits: [B, C, H, W] 网络输出 (未经过 softmax)
        targets: [B, H, W] 真实标签 (0, 1, 2)
        """
        num_classes = logits.shape[1]
        
        # 1. 计算 softmax 概率
        probs = F.softmax(logits, dim=1)
        
        # 2. 真实标签转为 one-hot，并调整维度为 [B, C, H, W]
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        # 3. 展平为 [B, C, H*W] 以便计算交集和并集
        probs = probs.view(probs.size(0), probs.size(1), -1)
        targets_one_hot = targets_one_hot.view(targets_one_hot.size(0), targets_one_hot.size(1), -1)
        
        # 4. 计算交集与基数 (cardinality)
        intersection = (probs * targets_one_hot).sum(dim=2)
        cardinality = probs.sum(dim=2) + targets_one_hot.sum(dim=2)
        
        # 5. 计算 Dice
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        
        # 返回 1 - dice 的均值作为 loss
        return 1. - dice_score.mean()

class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        # 组合损失：CE + Dice
        return self.ce(logits, targets) + self.dice(logits, targets)
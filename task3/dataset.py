import torch
import numpy as np
from torchvision.datasets import OxfordIIITPet
import torchvision.transforms as T
from torch.utils.data import DataLoader

class MaskTransform:
    def __init__(self, size=(256, 256)):
        self.size = size

    def __call__(self, mask):
        mask = T.Resize(self.size, interpolation=T.InterpolationMode.NEAREST)(mask)
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)
        mask = mask - 1  
        return mask

def get_dataloaders(batch_size=16, data_dir='./data'):
    # 图像预处理：缩放、转张量、归一化
    image_transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    mask_transform = MaskTransform((256, 256))

    # 加载训练集
    train_dataset = OxfordIIITPet(
        root=data_dir, split='trainval', target_types='segmentation',
        transform=image_transform, target_transform=mask_transform, download=True
    )
    # 加载测试/验证集
    test_dataset = OxfordIIITPet(
        root=data_dir, split='test', target_types='segmentation',
        transform=image_transform, target_transform=mask_transform, download=True
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(batch_size=32):
    # 1. 定义数据预处理方式
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224), 
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),             
        transforms.Normalize(              
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),            
        transforms.CenterCrop(224),        
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    # 2. 下载并加载 Dataset
    train_dataset = datasets.OxfordIIITPet(
        root='./data', 
        split='trainval', 
        target_types='category', 
        download=True, 
        transform=train_transform
    )

    test_dataset = datasets.OxfordIIITPet(
        root='./data', 
        split='test',    
        target_types='category', 
        download=True, 
        transform=test_transform
    )

    # 3. 封装成 DataLoader
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"数据准备完毕！训练集包含 {len(train_dataset)} 张图片，测试集包含 {len(test_dataset)} 张图片。")
    print(f"宠物种类数: {len(train_dataset.classes)}")
    return train_loader, test_loader

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的计算设备: {device}")
    train_dl, test_dl = get_dataloaders(batch_size=32)
    images, labels = next(iter(train_dl))
    print(f"一个 Batch 的图片形状: {images.shape}") 
    print(f"一个 Batch 的标签形状: {labels.shape}")
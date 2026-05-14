import torch
import torch.nn as nn

# U-Net 的基础组件：连续两次卷积
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

# U-Net 网络定义
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_classes=3):
        super(UNet, self).__init__()
        
        # 编码器 (Encoder) - 下采样
        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)
        
        # 瓶颈层 (Bottleneck)
        self.bottleneck = DoubleConv(512, 1024)
        
        # 解码器 (Decoder) - 上采样 + 特征拼接 (Skip Connection)
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv1 = DoubleConv(1024, 512) # 512(up) + 512(skip) = 1024
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConv(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv3 = DoubleConv(256, 128)
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv4 = DoubleConv(128, 64)
        
        # 输出层，通道数等于分类数 (3)
        self.out_conv = nn.Conv2d(64, out_classes, kernel_size=1)

    def forward(self, x):
        # --- 下采样阶段 ---
        x1 = self.down1(x)
        x2 = self.down2(self.pool(x1))
        x3 = self.down3(self.pool(x2))
        x4 = self.down4(self.pool(x3))
        
        # --- 瓶颈层 ---
        bn = self.bottleneck(self.pool(x4))
        
        # --- 上采样与拼接阶段 ---
        u1 = self.up1(bn)
        u1 = torch.cat([x4, u1], dim=1) # Skip Connection
        u1 = self.up_conv1(u1)
        
        u2 = self.up2(u1)
        u2 = torch.cat([x3, u2], dim=1) # Skip Connection
        u2 = self.up_conv2(u2)
        
        u3 = self.up3(u2)
        u3 = torch.cat([x2, u3], dim=1) # Skip Connection
        u3 = self.up_conv3(u3)
        
        u4 = self.up4(u3)
        u4 = torch.cat([x1, u4], dim=1) # Skip Connection
        u4 = self.up_conv4(u4)
        
        return self.out_conv(u4)
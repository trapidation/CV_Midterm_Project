# 计算机视觉期中作业 (CV Midterm Project)

**姓名**：李茗瑄  
**学号**：24300980036  
**分工情况**：本作业由本人独立完成（100%）

---
## 🔗 模型权重与实验数据下载
* **Google Drive 下载链接**：[https://drive.google.com/drive/folders/10uflr_85MMmWWlkj1jdED17_AXvseVqw?usp=drive_link]
* **云盘目录结构说明**：
  * `Task 1 权重/`：包含 ResNet-18 和 ViT-Tiny 训练好的 `.pth` 权重文件。
  * `Task 2 视频与权重/`：包含 YOLOv8 微调权重 `best.pt`，原测试视频 `test_video.mp4`，以及渲染好的追踪结果视频 `output_tracking.mp4`。
  * `Task 3 权重/`：包含手写 U-Net 训练的最佳 `.pth` 权重文件。
* **测试说明**：如需本地运行测试代码，请从网盘对应任务的文件夹中下载文件，并放入您本地对应 Task 的代码文件夹内。
## 📁 仓库目录结构
本项目包含三个独立的代码模块，分别对应三个任务：
* `Task1_ImageNet_Finetune/`：微调CNN与ViT实现宠物识别
* `Task2_YOLOv8_Tracking/`：YOLOv8目标检测与视频追踪计数
* `Task3_UNet_Segmentation/`：从零手写U-Net与损失函数对比

---

## 🛠️ 环境配置 (Environment Setup)

推荐使用 Anaconda 创建虚拟环境。在终端中依次运行以下命令完成环境安装：

1. 创建并激活虚拟环境：
   conda create -n cv_midterm python=3.9 -y
   conda activate cv_midterm

2. 安装 PyTorch (以 CUDA 11.8 为例)：
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

3. 安装本项目的核心依赖库：
   pip install ultralytics timm swanlab opencv-python

---

## 🚀 任务一：微调卷积神经网络实现宠物识别 (Task 1)
**进入工作目录**：`cd Task1_ImageNet_Finetune`

本任务在 Oxford-IIIT Pet 数据集上进行了预训练模型微调、从零训练消融实验、超参数网格搜索以及引入 ViT 注意力机制的对比实验。

**运行代码指引：**
* 运行 Baseline 微调 (ResNet-18, 差异化学习率)：
  python train_baseline.py

* 运行预训练消融实验 (随机初始化参数)：
  python train_scrach.py

* 运行轻量级 Vision Transformer (ViT-Tiny)：
  python train_vit.py

* 运行超参数自动化网格搜索 (Sweep)：
  python train_sweep.py

*(注：data_loader.py 为数据预处理模块，会被上述脚本自动调用，无需单独运行)*

---

## 🛸 任务二：场景目标检测与视频多目标追踪 (Task 2)
**进入工作目录**：`cd Task2_YOLOv8_Tracking`

本任务使用 VisDrone 数据集微调了 YOLOv8s，并结合 ByteTrack 算法与自定义越线计数逻辑（基于 track_history 时空坐标比对），实现了视频流的精准检测与目标追踪。

**运行代码指引：**
* 步骤 1：微调训练模型（已配置自动下载 VisDrone 数据集）：
  python train.py

* 步骤 2：视频追踪与越线计数测试：
  （运行前请确保已将网盘中的 best.pt 以及 test_video.mp4 放入当前目录）
  python track_and_count.py

运行完毕后，当前目录下会生成渲染完成的 output_tracking.mp4 视频文件。

---

## 🖼️ 任务三：从零搭建与损失函数工程 (Task 3)
**进入工作目录**：`cd Task3_UNet_Segmentation`

本任务未使用任何预训练权重，纯底层 API 手写搭建了包含 Skip Connection 的 4 层下采样 U-Net。自定义实现了 Dice Loss 等机制以解决类别不平衡问题。

**运行代码指引：**
本模块通过命令行参数 `--loss` 实现无代码修改的损失函数切换。

* 使用 标准交叉熵损失 (CE Loss) 进行训练：
  python train.py --loss ce

* 使用 手动实现的 Dice Loss 进行训练：
  python train.py --loss dice

* 使用 组合损失 (CE Loss + Dice Loss) 进行训练以达到最佳 mIoU：
  python train.py --loss combined
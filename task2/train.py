import swanlab
from ultralytics import YOLO
from swanlab.integration.ultralytics import add_swanlab_callback

if __name__ == '__main__':
    # 1. 初始化 SwanLab 实验
    swanlab.init(
        project="Midterm-Task2",      # SwanLab 中的项目名称（类似文件夹）
        experiment_name="YOLOv8s_VisDrone", # 本次实验的名字
        description="使用YOLOv8s在VisDrone数据集上进行微调"
    )

    # 2. 加载 YOLOv8 预训练模型 (推荐使用 's' 版本，适合大多数显卡)
    model = YOLO('yolov8s.pt')

    # 3. 将 SwanLab 的回调函数绑定到 YOLO 模型上，这样它就会自动收集数据并画图！
    add_swanlab_callback(model)

    # 4. 开始训练
    # VisDrone.yaml 是 Ultralytics 内置的，会自动下载数据集（约 1.5GB）
    results = model.train(
        data='VisDrone.yaml',  
        epochs=50,            # 训练轮数（建议 50-100，根据你的显卡速度决定）
        imgsz=640,            # 输入图像的尺寸
        batch=16,             # 批次大小，如果显存不够报 OOM 错误，改成 8 或 4
        device=0,             # 使用第一张显卡，如果用 CPU 填 'cpu'
        workers=4             # 数据加载线程数
    )
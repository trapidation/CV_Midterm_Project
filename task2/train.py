import swanlab
from ultralytics import YOLO
from swanlab.integration.ultralytics import add_swanlab_callback

if __name__ == '__main__':
    swanlab.init(
        project="Midterm-Task2",     
        experiment_name="YOLOv8s_VisDrone", 
        description="使用YOLOv8s在VisDrone数据集上进行微调"
    )

    model = YOLO('yolov8s.pt')

    add_swanlab_callback(model)

    results = model.train(
        data='VisDrone.yaml',  
        epochs=50,            # 训练轮数（建议 50-100，根据你的显卡速度决定）
        imgsz=640,            # 输入图像的尺寸
        batch=16,             # 批次大小，如果显存不够报 OOM 错误，改成 8 或 4
        device=0,             # 使用第一张显卡，如果用 CPU 填 'cpu'
        workers=4             # 数据加载线程数
    )
import cv2
from ultralytics import YOLO

# 1. 加载你训练好的专属模型
# 注意：路径要换成你刚刚训练出来的 best.pt 所在的真实路径
model = YOLO('runs/detect/train/weights/best.pt') 

# 2. 读取你拍好的测试视频
video_path = "test_video.mp4"
cap = cv2.VideoCapture(video_path)

# 3. 越线计数的设置
# 假设你的视频高度是 1080，我们在中间画一条线，所以 Y=540。你可以根据视频自己调整。
LINE_Y = 540 
crossed_ids = set()    # 用于记录已经算过的人/车的 ID，防止一个人被数多次
track_history = {}     # 用于记录每个人/车在上一帧的 Y 坐标：{ID: 上一帧的Y坐标}

# 开始逐帧处理视频
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("视频处理完毕！")
        break

    # 4. 让模型去跟踪当前帧画面
    # persist=True 表示保持跟踪的历史信息，tracker 使用内置的 bytetrack
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
    
    # 获取画好了检测框和 ID 的新画面
    annotated_frame = results[0].plot()

    # 5. 如果画面中检测到了物体，开始越线判断逻辑
    if results[0].boxes is not None and results[0].boxes.id is not None:
        # 获取所有框的坐标 (x_center, y_center, width, height)
        boxes = results[0].boxes.xywh.cpu()
        # 获取对应的跟踪 ID
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            center_y = int(y) # 我们只关心中心点的上下高度 (Y坐标)

            # 判断逻辑：上一帧在记录中吗？
            if track_id in track_history:
                prev_y = track_history[track_id]
                
                # 【核心逻辑】：如果它的上一帧在红线上方(prev_y < LINE_Y)
                # 并且这一帧跑到了红线下方(center_y >= LINE_Y)，说明他越线了！
                if prev_y < LINE_Y and center_y >= LINE_Y:
                    crossed_ids.add(track_id) # 记录该ID，跨越人数 +1
                
                # (进阶：如果你想双向计数，加上下面这句即可)
                # elif prev_y > LINE_Y and center_y <= LINE_Y:
                #     crossed_ids.add(track_id)

            # 把当前的 Y 坐标存起来，留给下一帧做对比
            track_history[track_id] = center_y

    # 6. 可视化绘制
    # 在画面上画出那条判定用的红线 (BGR格式，(0, 0, 255) 是红色)
    cv2.line(annotated_frame, (0, LINE_Y), (annotated_frame.shape[1], LINE_Y), (0, 0, 255), 2)
    
    # 在画面左上角显示统计数字
    count_text = f"Crossed Count: {len(crossed_ids)}"
    cv2.putText(annotated_frame, count_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    # 7. 实时显示画面
    # 缩小显示窗口以便在屏幕上查看 (可选)
    display_frame = cv2.resize(annotated_frame, (1280, 720))
    cv2.imshow("Tracking and Counting", display_frame)
    
    # 按下 'q' 键退出播放
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
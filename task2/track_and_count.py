import cv2
from ultralytics import YOLO

# 1. 加载你刚刚炼好的仙丹 (请确保 best.pt 和 test_video.mp4 都在当前这个文件夹里)
model = YOLO('best.pt')

# 2. 读取你的测试视频
video_path = "test_video.mp4"
cap = cv2.VideoCapture(video_path)

# 获取原视频的宽度、高度和帧率，为了等下原画质保存输出视频
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# 设置保存视频的格式和文件名
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter('output_tracking.mp4', fourcc, fps, (width, height))

# 3. 越线计数的设置
# 假设你想在画面偏中间画一条线。你可以根据你的视频画面把 540 改大(往下移)或改小(往上移)
LINE_Y = int(height * 0.5)  # 默认画在画面正中间的水平线
crossed_ids = set()    # 记录已经跨过线的人/车，防止重复数
track_history = {}     # 记录每个人/车在上一帧的位置

print("🚀 开始处理视频，按 'q' 键可提前退出...")

# 开始逐帧处理视频
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("✅ 视频处理完毕！已保存为 output_tracking.mp4")
        break

    # 4. 调用强大的追踪器 (persist=True 保持记忆，tracker 使用自带的 bytetrack)
    results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
    
    # 拿到画好框和 ID 的当前画面
    annotated_frame = results[0].plot()

    # 5. 核心逻辑：越线判断
    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu() # 获取坐标
        track_ids = results[0].boxes.id.int().cpu().tolist() # 获取 ID

        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            center_y = int(y) # 只看中心点的高低位置

            # 如果这个物体上一帧也出现过
            if track_id in track_history:
                prev_y = track_history[track_id]
                
                # 【判断越线】：上一帧在线上面，这一帧跑到线下面了 (从上往下走)
                if prev_y < LINE_Y and center_y >= LINE_Y:
                    crossed_ids.add(track_id)
                # (如果你拍的视频是从下往上走，把上面的逻辑反过来，或者两个都写上)
                elif prev_y > LINE_Y and center_y <= LINE_Y:
                    crossed_ids.add(track_id)

            # 更新记忆，为下一帧做准备
            track_history[track_id] = center_y

    # 6. 画图与显示
    # 在画面上画出那条红色的虚拟判定线
    cv2.line(annotated_frame, (0, LINE_Y), (width, LINE_Y), (0, 0, 255), 3)
    
    # 在画面左上角写上统计人数 (绿色大字)
    count_text = f"Total Crossed: {len(crossed_ids)}"
    cv2.putText(annotated_frame, count_text, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)

    # 写入到输出视频文件
    out_video.write(annotated_frame)

    # 在屏幕上实时预览 (为了防止原视频太大撑爆屏幕，预览时缩小一点)
    display_frame = cv2.resize(annotated_frame, (1280, 720))
    cv2.imshow("Tracking & Counting", display_frame)
    
    # 按下 'q' 键可中断播放
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 释放资源释放内存
cap.release()
out_video.release()
cv2.destroyAllWindows()
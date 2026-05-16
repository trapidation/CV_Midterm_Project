import cv2
from ultralytics import YOLO

model = YOLO('runs/detect/train/weights/best.pt') 

video_path = "test_video.mp4"
cap = cv2.VideoCapture(video_path)

LINE_Y = 540 
crossed_ids = set()  
track_history = {}     

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("视频处理完毕！")
        break

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

            if track_id in track_history:
                prev_y = track_history[track_id]
                
                if prev_y < LINE_Y and center_y >= LINE_Y:
                    crossed_ids.add(track_id) 

            track_history[track_id] = center_y

    cv2.line(annotated_frame, (0, LINE_Y), (annotated_frame.shape[1], LINE_Y), (0, 0, 255), 2)
    
    # 在画面左上角显示统计数字
    count_text = f"Crossed Count: {len(crossed_ids)}"
    cv2.putText(annotated_frame, count_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    display_frame = cv2.resize(annotated_frame, (1280, 720))
    cv2.imshow("Tracking and Counting", display_frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
import cv2
from ultralytics import YOLO
model = YOLO('best.pt')

video_path = "test_video.mp4"
cap = cv2.VideoCapture(video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter('output_tracking.mp4', fourcc, fps, (width, height))

LINE_Y = int(height * 0.5)  
crossed_ids = set()    
track_history = {}     

print("开始处理视频，按 'q' 键可提前退出...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("视频处理完毕！已保存为 output_tracking.mp4")
        break

    results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
    
    annotated_frame = results[0].plot()

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu() 
        track_ids = results[0].boxes.id.int().cpu().tolist() 

        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            center_y = int(y) 

            if track_id in track_history:
                prev_y = track_history[track_id]
                
                if prev_y < LINE_Y and center_y >= LINE_Y:
                    crossed_ids.add(track_id)
                elif prev_y > LINE_Y and center_y <= LINE_Y:
                    crossed_ids.add(track_id)

            track_history[track_id] = center_y

    cv2.line(annotated_frame, (0, LINE_Y), (width, LINE_Y), (0, 0, 255), 3)
    
    count_text = f"Total Crossed: {len(crossed_ids)}"
    cv2.putText(annotated_frame, count_text, (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)

    out_video.write(annotated_frame)

    display_frame = cv2.resize(annotated_frame, (1280, 720))
    cv2.imshow("Tracking & Counting", display_frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out_video.release()
cv2.destroyAllWindows()
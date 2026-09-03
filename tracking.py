# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
from ultralytics import YOLO

model=YOLO("yolo11m.pt")
video= cv2.VideoCapture("video/IMG_5386.mp4")

if not video.isOpened():
    print("video could not be opened")
    exit()
while True:
    ret, frame=video.read()

    if not ret:
        print("video finished")
        break
    results= model.track(frame,persist=True,conf=0.5)
    result=results[0]

    if result.boxes.id is not None:
        track_ids= result.boxes.id.int().cpu().tolist()

        for track_id,box in zip(track_ids,result.boxes):
            class_id= int(box.cls[0])
            class_name= result.names[class_id]

            confidece= float(box.conf[0])

            print(
                f"ID: {track_id}|"
                f" Object: {class_name} |"
                f" Confidence: {confidece: .2f}"
            )

    annotated_frame=results[0].plot()

    cv2.imshow("object tracking",annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
        
video.release()
cv2.destroyAllWindows()
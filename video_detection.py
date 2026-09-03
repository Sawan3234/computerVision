
# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
from ultralytics import YOLO

model= YOLO("yolo11n.pt")
video= cv2.VideoCapture("video/IMG_5386.mp4")

if not video.isOpened():
    print("could not open video")
    exit()
while True:
    ret,frame=video.read()

    if not ret:
        print("video finished")
        break
    results=model(frame,conf=0.5)
    result=results[0]
    for box in result.boxes:
        class_id = int(box.cls[0])  
        class_name = model.names[class_id]

        confidence = float(box.conf[0])

    x1, y1, x2, y2 = map(int, box.xyxy[0])

    print(
        f"Object: {class_name} | "
        f"Confidence: {confidence:.2f} | "
        f"Coordinates: ({x1}, {y1}, {x2}, {y2})"
    )

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (255, 0, 0),
        2
    )

    label = f"{class_name} {confidence:.2f}"

    cv2.putText(
        frame,
        label,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 0, 0),
        2
    )
    cv2.imshow("video Detection",frame)

    if cv2.waitKey(1) & 0xFF== ord("q"):
        break

video.release()
cv2.destroyAllWindows() 

# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
from ultralytics import YOLO
import time

model=YOLO("yolo11n.pt")
camera= cv2.VideoCapture(0)
prev_time=time.time()


if not camera.isOpened():
    print("could not open camera")
    exit()
allowed_classes={"person","bottle","chair"}
while True:
    ret,frame=camera.read()

    if not ret:
        print("could not open frame")
        break

    results= model(frame,conf=0.5)

    result=results[0]
    curr_time=time.time()
    fps= 1/(curr_time-prev_time)
    prev_time=curr_time


    for box in result.boxes:
        class_id= int(box.cls[0])
        class_name= model.names[class_id]

        confidence= float(box.conf[0])
       

        if class_name not in allowed_classes:
            continue

        x1,y1,x2,y2= map(int,box.xyxy[0])
        print(
         f"Object: {class_name} | "
         f"Confidence: {confidence:.2f} | "
         f"Coordinates: ({x1}, {y1}, {x2}, {y2})"
        )

        cv2.rectangle(
            frame,
            (x1,y1),
            (x2,y2),
            (255,0,0),
            2

        )
        label= f"{class_name},{confidence:.2f}"

        cv2.putText(
            frame,
            label,
            (x1,y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,0,0),
            2
        )
        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (10,30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0),
            2
        )
    cv2.imshow("manual detection", frame)
    

    if cv2.waitKey(1)& 0xFF== ord("q"):
        break
camera.release()
cv2.destroyAllWindows()

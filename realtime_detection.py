# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
from ultralytics import YOLO

model=YOLO("yolo11n.pt")
camera= cv2.VideoCapture(0)

if not camera.isOpened():
    print("could not open camera")
    exit()
while True:
    ret,frame=camera.read()

    if not ret:
        print("could not open frame")
        break

    results= model(frame)

    result=results[0]

    annotated_frame= result.plot()
    cv2.imshow("yolo object detect", annotated_frame)
    
    if cv2.waitKey(1)& 0xFF== ord("q"):
        break
camera.release()
cv2.destroyAllWindows()

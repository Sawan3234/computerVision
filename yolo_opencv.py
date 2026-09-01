# pyrefly: ignore [missing-import]
import cv2
from ultralytics import YOLO

model=YOLO("yolo11n.pt")

image=cv2.imread("image/test.jpg")
results=model(image)
result=results[0]

for box in result.boxes:
    class_id= int(box.cls[0])
    class_name= model.names[class_id]

    confidence= float(box.conf[0])

    x1,y1,x2,y2= map(int,box.xyxy[0])

    cv2.rectangle(
        image,
        (x1,y1),
        (x2,y2),
        (255,0,0),
        2

    )
    label= f"{class_name},{confidence:.2f}"

    cv2.putText(
        image,
        label,
        (x1,y1-10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255,0,0),
        2
    )
cv2.imshow("YOLO + OpenCV",image)

cv2.waitKey(0)
cv2.destroyAllWindows()

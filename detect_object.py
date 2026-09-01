# pyrefly: ignore [missing-import]
from ultralytics import YOLO

model=YOLO("yolo11n.pt")
results=model("image/img1.jpeg")
result=results[0]

for box in result.boxes:
    class_id=int(box.cls[0])
    confidence=float(box.conf[0])

    class_name=model.names[class_id]
    x1, y1, x2, y2= map(int,box.xyxy[0])

    print(
        f"object:{class_name}|"
        f"confidence:{confidence:.2f}"
    )
    print(
        f"Coordinates:"
        f"x1={x1}, y1={y1},"
        f"x2={x2}, y2={y2}"
    )
    
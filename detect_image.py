# pyrefly: ignore [missing-import]
from ultralytics import YOLO

model= YOLO("yolo11n.pt")

result=model("image/img1.jpeg")

result[0].show()
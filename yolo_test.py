# pyrefly: ignore [missing-import]
from ultralytics import YOLO

model=YOLO("yolo11n.pt")
print("yolo model loaded sucessfully")
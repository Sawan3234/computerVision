# pyrefly: ignore [missing-import]
from ultralytics import YOLO

model= YOLO("yolo11n.pt")

results=model("image/testimg.jpg")

result=results[0]

print("Boxes:")
print(result.boxes)

print("\n Class IDs:")
print(result.boxes.cls)

print("\nConfidence:")
print(result.boxes.conf)

print("\nCoordinates:")
print(result.boxes.xyxy)
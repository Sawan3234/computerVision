# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
from ultralytics import YOLO

model = YOLO("yolo11l.pt")
video = cv2.VideoCapture("video/IMG_5386.mp4")

class_count = {}

if not video.isOpened():
    print("video could not be opened")
    exit()

while True:
    ret, frame = video.read()

    if not ret:
        print("video finished")
        break

    results = model.track(frame, persist=True, conf=0.5)
    result = results[0]

    if result.boxes.id is not None:

        track_ids = result.boxes.id.int().cpu().tolist()

        for track_id, box in zip(track_ids, result.boxes):

            class_id = int(box.cls[0])
            class_name = result.names[class_id]

            confidence = float(box.conf[0])

            if class_name not in class_count:
                class_count[class_name] = set()

            class_count[class_name].add(track_id)

            label = f"{class_name} | ID:{track_id} | {confidence:.2f}"

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.4,
                (255, 0, 0),
                3
            )

    y = 40

    for class_name, ids in class_count.items():

        text = f"{class_name}: {len(ids)}"

        cv2.putText(
            frame,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.5,
            (255, 0, 0),
            5
        )

        y += 45

    cv2.imshow("object tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()


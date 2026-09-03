# pyrefly: ignore [missing-import]
import cv2

camera=cv2.VideoCapture(0)

if not camera.isOpened():
    print("Could not open camera")
    exit()
while True:
    ret,frame=camera.read()

    if not ret:
        print("could not read frame")
        break
    cv2.rectangle(
        frame,
        (100,100),
        (400,400),
        (255,0,0),
        2
    )

    cv2.putText(
        frame,
        "Object",
        (100,90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,0,0),
        2,
    )
    cv2.imshow("Object Detection", frame)
    if cv2.waitKey(1)& 0xFF== ord("q"):
        break
camera.release()
cv2.destroyAllWindows()

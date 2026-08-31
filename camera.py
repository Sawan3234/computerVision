# pyrefly: ignore [missing-import]
import cv2

camera=cv2.VideoCapture(0)

if not camera.isOpened():
    print("could not open camera")
    exit()
while True:
    ret, frame= camera.read()

    if not ret:
        print("Could not read frame")
        break
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Grayscale", cv2.WINDOW_NORMAL)

    cv2.imshow("Original",frame)
    cv2.imshow("grayscale",gray)
    if cv2.waitKey(1)& 0xFF == ord("q"):
        break
camera.release()
cv2.destroyAllWindows()


# pyrefly: ignore [missing-import]
import cv2

image=cv2.imread("image/testimg.jpg")
if image is None:
    print("could not load image")
else:
    print("image loaded successfully")
    print("image dimension:",image.shape)

    cv2.imshow("test image",image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
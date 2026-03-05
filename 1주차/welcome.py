import cv2 as cv
import numpy as np 

img = cv.imread("welcome.jpg")
if img is None:
    print("이미지 없음")
    exit()

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray_bgr = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

out = np.hstack((img, gray_bgr))

# 화면에 맞게 축소 (예: 0.5배)
scale = 0.5
out_small = cv.resize(out, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)

cv.imshow("Color | Gray", out_small)
cv.waitKey(0)
cv.destroyAllWindows()
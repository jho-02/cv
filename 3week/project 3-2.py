import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("3week/image/dabo.jpg")

if img is None:
    print("이미지를 불러오지 못했습니다.")
    exit()

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

edges = cv.Canny(gray, 100, 200)

line_img = img.copy()

lines = cv.HoughLinesP(
    edges,
    rho=1,
    theta=np.pi / 180,
    threshold=140,
    minLineLength=45,
    maxLineGap=3
)

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
line_img_rgb = cv.cvtColor(line_img, cv.COLOR_BGR2RGB)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(line_img_rgb)
plt.title("Detected Lines")
plt.axis("off")

plt.tight_layout()
plt.show()
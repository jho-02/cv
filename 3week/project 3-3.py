import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img = cv.imread("3week/image/coffee cup.JPG")

if img is None:
    print("이미지를 불러오지 못했습니다.")
    exit()

img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

h, w = img.shape[:2]

mask = np.zeros((h, w), np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

rect = (
    50,
    50,
    w - 100,
    h - 100
)

cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)

mask2 = np.where(
    (mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD),
    0,
    1
).astype("uint8")

result = img_rgb * mask2[:, :, np.newaxis]

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(mask2 * 255, cmap="gray")
plt.title("Mask Image")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(result)
plt.title("Background Removed")
plt.axis("off")

plt.tight_layout()
plt.show()
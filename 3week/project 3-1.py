import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread("3week/image/edgeDetectionImage.jpg")

if img is None:
    print("이미지를 불러오지 못했습니다.")
    exit()

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

grad_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
grad_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)

magnitude = cv.magnitude(grad_x, grad_y)
edge_strength = cv.convertScaleAbs(magnitude)

img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(edge_strength, cmap="gray")
plt.title("Edge Strength")
plt.axis("off")

plt.tight_layout()
plt.show()
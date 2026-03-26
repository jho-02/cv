import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread("mot_color70.jpg")

if img is None:
    print("이미지를 불러오지 못했습니다.")
    exit()

img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

sift = cv.SIFT_create(nfeatures=300)
keypoints, descriptors = sift.detectAndCompute(img, None)

img_keypoints = cv.drawKeypoints(
    img,
    keypoints,
    None,
    flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

img_keypoints_rgb = cv.cvtColor(img_keypoints, cv.COLOR_BGR2RGB)

plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(img_keypoints_rgb)
plt.title(f"SIFT Keypoints ({len(keypoints)} points)")
plt.axis("off")

plt.tight_layout()
plt.show()
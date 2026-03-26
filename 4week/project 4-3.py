import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img1 = cv.imread("4week/img1.jpg")
img2 = cv.imread("4week/img2.jpg")

if img1 is None or img2 is None:
    print("이미지를 불러오지 못했습니다.")
    exit()

sift = cv.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

bf = cv.BFMatcher(cv.NORM_L2)
matches = bf.knnMatch(des2, des1, k=2)

good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

if len(good_matches) < 4:
    print("호모그래피 계산에 필요한 매칭점이 부족합니다.")
    exit()

src_pts = np.float32([kp2[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

if H is None:
    print("호모그래피 계산 실패")
    exit()

h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]

warped = cv.warpPerspective(img2, H, (w1 + w2, max(h1, h2)))
warped[0:h1, 0:w1] = img1

matches_mask = mask.ravel().tolist()

matching_result = cv.drawMatches(
    img2, kp2,
    img1, kp1,
    good_matches, None,
    matchesMask=matches_mask,
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

warped_rgb = cv.cvtColor(warped, cv.COLOR_BGR2RGB)
matching_rgb = cv.cvtColor(matching_result, cv.COLOR_BGR2RGB)

plt.figure(figsize=(20, 8))

plt.subplot(1, 2, 1)
plt.imshow(matching_rgb)
plt.title(f"Matching Result (Inliers: {sum(matches_mask)}/{len(good_matches)})")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(warped_rgb)
plt.title("Warped Image")
plt.axis("off")

plt.tight_layout()
plt.show()


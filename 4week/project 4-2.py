import cv2 as cv
import matplotlib.pyplot as plt

img1 = cv.imread("4week/mot_color70.jpg")
img2 = cv.imread("4week/mot_color83.jpg")

if img1 is None or img2 is None:
    print("이미지를 불러오지 못했습니다.")
    exit()

sift = cv.SIFT_create()

kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

good_matches = matches[:50]

result = cv.drawMatches(
    img1, kp1,
    img2, kp2,
    good_matches, None,
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

result_rgb = cv.cvtColor(result, cv.COLOR_BGR2RGB)

plt.figure(figsize=(18, 8))
plt.imshow(result_rgb)
plt.title(f"SIFT Matching Result ({len(good_matches)} matches)")
plt.axis("off")
plt.tight_layout()
plt.show()

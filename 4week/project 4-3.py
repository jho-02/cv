import cv2 as cv  # OpenCV 라이브러리를 cv라는 이름으로 사용합니다.
import numpy as np  # 좌표 배열을 다루기 위해 NumPy를 사용합니다.
import matplotlib.pyplot as plt  # 결과 이미지를 시각화하기 위해 matplotlib을 사용합니다.

# 정합에 사용할 두 장의 이미지를 불러옵니다.
# 여기서는 img1을 기준 이미지로, img2를 변환할 이미지로 사용합니다.
img1 = cv.imread("4week/img1.jpg")
img2 = cv.imread("4week/img2.jpg")

# 두 이미지 중 하나라도 정상적으로 불러오지 못한 경우
# 이후 연산을 진행할 수 없으므로 프로그램을 종료합니다.
if img1 is None or img2 is None:
    print("이미지를 불러오지 못했습니다.")
    exit()

# SIFT 객체를 생성합니다.
# 이 객체를 이용하여 특징점과 기술자를 추출합니다.
sift = cv.SIFT_create()

# 첫 번째 이미지에서 특징점과 기술자를 추출합니다.
kp1, des1 = sift.detectAndCompute(img1, None)

# 두 번째 이미지에서 특징점과 기술자를 추출합니다.
kp2, des2 = sift.detectAndCompute(img2, None)

# BFMatcher 객체를 생성합니다.
# SIFT 기술자는 실수형 벡터이므로 거리 계산 방식으로 NORM_L2를 사용합니다.
bf = cv.BFMatcher(cv.NORM_L2)

# 각 특징점에 대해 가장 가까운 이웃 2개를 찾습니다.
# 이를 통해 ratio test를 적용할 수 있습니다.
matches = bf.knnMatch(des2, des1, k=2)

# 좋은 매칭만 저장할 리스트를 생성합니다.
good_matches = []

# knnMatch로 구한 최근접 이웃 2개의 거리를 비교하여
# 더 신뢰도 높은 매칭만 선별합니다.
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

# 호모그래피를 계산하려면 최소 4개의 대응점이 필요하므로,
# 좋은 매칭 개수가 4개 미만이면 더 이상 진행하지 않습니다.
if len(good_matches) < 4:
    print("호모그래피 계산에 필요한 매칭점이 부족합니다.")
    exit()

# 두 번째 이미지에서의 대응 좌표들을 추출합니다.
# queryIdx는 knnMatch 기준으로 현재 질의 이미지의 특징점을 의미합니다.
src_pts = np.float32([kp2[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# 첫 번째 이미지에서의 대응 좌표들을 추출합니다.
# trainIdx는 비교 대상 이미지에서 매칭된 특징점을 의미합니다.
dst_pts = np.float32([kp1[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# 좋은 매칭점들을 이용하여 호모그래피 행렬을 계산합니다.
# RANSAC을 사용하여 잘못된 매칭점(outlier)의 영향을 줄입니다.
H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)

# 호모그래피 계산에 실패한 경우 프로그램을 종료합니다.
if H is None:
    print("호모그래피 계산 실패")
    exit()

# 두 이미지의 높이와 너비를 각각 구합니다.
h1, w1 = img1.shape[:2]
h2, w2 = img2.shape[:2]

# 계산된 호모그래피를 이용하여 img2를 img1의 좌표계로 변환합니다.
# 출력 크기는 두 이미지를 합친 파노라마 크기로 설정합니다.
warped = cv.warpPerspective(img2, H, (w1 + w2, max(h1, h2)))

# 기준 이미지인 img1을 결과 이미지의 왼쪽 영역에 그대로 배치합니다.
warped[0:h1, 0:w1] = img1

# RANSAC 결과로 얻은 inlier / outlier 정보를 리스트 형태로 변환합니다.
matches_mask = mask.ravel().tolist()

# 특징점 매칭 결과를 시각화합니다.
# matchesMask를 사용하면 RANSAC에서 inlier로 판단된 매칭만 표시할 수 있습니다.
matching_result = cv.drawMatches(
    img2, kp2,
    img1, kp1,
    good_matches, None,
    matchesMask=matches_mask,
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# OpenCV는 BGR 형식으로 이미지를 처리하므로,
# matplotlib에서 올바른 색상을 출력하기 위해 RGB로 변환합니다.
warped_rgb = cv.cvtColor(warped, cv.COLOR_BGR2RGB)
matching_rgb = cv.cvtColor(matching_result, cv.COLOR_BGR2RGB)

# 출력 창의 크기를 설정합니다.
plt.figure(figsize=(20, 8))

# 왼쪽 subplot에 특징점 매칭 결과를 출력합니다.
plt.subplot(1, 2, 1)
plt.imshow(matching_rgb)
plt.title(f"Matching Result (Inliers: {sum(matches_mask)}/{len(good_matches)})")
plt.axis("off")

# 오른쪽 subplot에 정합된 이미지를 출력합니다.
plt.subplot(1, 2, 2)
plt.imshow(warped_rgb)
plt.title("Warped Image")
plt.axis("off")

# subplot 간 간격을 자동으로 정리합니다.
plt.tight_layout()

# 최종 결과를 화면에 출력합니다.
plt.show()
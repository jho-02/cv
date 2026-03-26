import cv2 as cv  # OpenCV 라이브러리를 cv라는 이름으로 사용합니다.
import matplotlib.pyplot as plt  # 결과 이미지를 출력하기 위해 matplotlib을 사용합니다.

# 두 장의 이미지를 불러옵니다.
# 첫 번째 이미지는 기준 이미지로 사용하고,
# 두 번째 이미지는 이와 비교할 이미지로 사용합니다.
img1 = cv.imread("4week/mot_color70.jpg")
img2 = cv.imread("4week/mot_color83.jpg")

# 두 이미지 중 하나라도 정상적으로 불러오지 못한 경우
# 이후 작업을 진행할 수 없으므로 프로그램을 종료합니다.
if img1 is None or img2 is None:
    print("이미지를 불러오지 못했습니다.")
    exit()

# SIFT 객체를 생성합니다.
# 이 객체를 이용하여 특징점과 기술자를 추출할 수 있습니다.
sift = cv.SIFT_create()

# 첫 번째 이미지에서 특징점과 기술자를 추출합니다.
kp1, des1 = sift.detectAndCompute(img1, None)

# 두 번째 이미지에서 특징점과 기술자를 추출합니다.
kp2, des2 = sift.detectAndCompute(img2, None)

# BFMatcher 객체를 생성합니다.
# SIFT 기술자는 실수형 벡터이므로 거리 계산 방식으로 NORM_L2를 사용합니다.
# crossCheck=True로 설정하여 서로 일치하는 매칭만 남기도록 합니다.
bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)

# 두 이미지의 기술자를 비교하여 특징점 매칭을 수행합니다.
matches = bf.match(des1, des2)

# 매칭 결과를 거리(distance) 기준으로 오름차순 정렬합니다.
# 거리가 작을수록 더 유사한 특징점이라고 볼 수 있습니다.
matches = sorted(matches, key=lambda x: x.distance)

# 매칭 결과가 너무 많으면 화면이 복잡해질 수 있으므로
# 상위 50개의 매칭만 선택하여 시각화합니다.
good_matches = matches[:50]

# 선택된 매칭 결과를 시각화합니다.
# 두 이미지를 나란히 놓고, 대응되는 특징점들을 선으로 연결합니다.
result = cv.drawMatches(
    img1, kp1,
    img2, kp2,
    good_matches, None,
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# OpenCV는 BGR 형식으로 이미지를 다루기 때문에,
# matplotlib에서 올바른 색으로 출력하기 위해 RGB로 변환합니다.
result_rgb = cv.cvtColor(result, cv.COLOR_BGR2RGB)

# 출력 창의 크기를 설정합니다.
plt.figure(figsize=(18, 8))

# 매칭 결과 이미지를 출력합니다.
plt.imshow(result_rgb)

# 제목에 현재 표시된 매칭 개수를 함께 표시합니다.
plt.title(f"SIFT Matching Result ({len(good_matches)} matches)")

# 축 정보는 필요하지 않으므로 제거합니다.
plt.axis("off")

# 레이아웃이 겹치지 않도록 자동으로 정렬합니다.
plt.tight_layout()

# 최종 결과를 화면에 출력합니다.
plt.show()
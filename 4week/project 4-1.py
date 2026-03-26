import cv2 as cv  # OpenCV 라이브러리를 cv라는 이름으로 사용합니다.
import matplotlib.pyplot as plt  # 결과 이미지를 시각화하기 위해 matplotlib을 사용합니다.

# 이미지 파일을 불러옵니다.
img = cv.imread("mot_color70.jpg")

# 이미지가 정상적으로 불러와졌는지 확인합니다.
# 만약 None이라면 경로 문제 등으로 이미지를 읽지 못한 경우입니다.
if img is None:
    print("이미지를 불러오지 못했습니다.")
    exit()

# OpenCV는 기본적으로 BGR 형식으로 이미지를 읽기 때문에,
# matplotlib에서 올바른 색으로 출력하기 위해 RGB로 변환합니다.
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# SIFT 객체를 생성합니다.
# nfeatures=300으로 설정하여 특징점 개수를 최대 300개로 제한합니다.
sift = cv.SIFT_create(nfeatures=300)

# detectAndCompute 함수를 사용하여 특징점을 검출하고,
# 각 특징점에 대한 기술자(descriptor)를 함께 계산합니다.
keypoints, descriptors = sift.detectAndCompute(img, None)

# 검출된 특징점을 원본 이미지 위에 시각화합니다.
# DRAW_RICH_KEYPOINTS 옵션을 사용하여 특징점의 크기와 방향까지 함께 표시합니다.
img_keypoints = cv.drawKeypoints(
    img,
    keypoints,
    None,
    flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# 시각화된 이미지도 matplotlib 출력에 맞게 RGB로 변환합니다.
img_keypoints_rgb = cv.cvtColor(img_keypoints, cv.COLOR_BGR2RGB)

# 출력 이미지 크기를 설정합니다.
plt.figure(figsize=(16, 8))

# 첫 번째 subplot에 원본 이미지를 출력합니다.
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("Original Image")  # 제목 설정
plt.axis("off")  # 축 정보 제거

# 두 번째 subplot에 특징점이 시각화된 이미지를 출력합니다.
plt.subplot(1, 2, 2)
plt.imshow(img_keypoints_rgb)
plt.title(f"SIFT Keypoints ({len(keypoints)} points)")  # 특징점 개수를 함께 표시
plt.axis("off")  # 축 정보 제거

# subplot 간 간격을 자동으로 조정합니다.
plt.tight_layout()

# 최종 결과를 화면에 출력합니다.
plt.show()
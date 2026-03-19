import cv2 as cv  # OpenCV 라이브러리 불러오기
import matplotlib.pyplot as plt  # 결과 이미지를 출력하기 위한 matplotlib 불러오기

# 실습에 사용할 이미지 파일을 읽어옴
img = cv.imread("3week/image/edgeDetectionImage.jpg")

# 이미지가 정상적으로 불러와졌는지 확인
# 파일 경로가 잘못되었거나 이미지가 없으면 None이 반환됨
if img is None:
    print("이미지를 불러오지 못했습니다.")
    exit()  # 더 이상 진행할 수 없으므로 프로그램 종료

# Sobel 연산은 밝기 변화량을 기준으로 계산하므로
# 컬러 이미지를 그레이스케일 이미지로 변환
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# x 방향 Sobel 연산
# (1, 0)은 x축 방향 미분을 의미하며, 세로 경계를 강조하는 효과가 있음
# CV_64F를 사용하여 음수와 실수 형태의 gradient 값을 그대로 유지
grad_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)

# y 방향 Sobel 연산
# (0, 1)은 y축 방향 미분을 의미하며, 가로 경계를 강조하는 효과가 있음
grad_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)

# x 방향 gradient와 y 방향 gradient를 결합하여
# 최종적인 에지 강도(Edge Strength)를 계산
magnitude = cv.magnitude(grad_x, grad_y)

# magnitude 결과는 실수형이므로 화면에 보기 쉽게
# 8비트 이미지 형식으로 변환
edge_strength = cv.convertScaleAbs(magnitude)

# OpenCV는 이미지를 BGR 형식으로 읽지만
# matplotlib는 RGB 형식으로 출력하므로 색상이 올바르게 보이게 변환
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# 출력 창 크기 설정
plt.figure(figsize=(10, 5))

# 첫 번째 subplot : 원본 이미지 출력
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")  # 축 눈금은 필요 없으므로 숨김

# 두 번째 subplot : Sobel로 계산한 에지 강도 이미지 출력
# 흑백 이미지이므로 cmap="gray" 옵션 사용
plt.subplot(1, 2, 2)
plt.imshow(edge_strength, cmap="gray")
plt.title("Edge Strength")
plt.axis("off")  # 축 눈금 숨김

# subplot 간 간격을 자동으로 정리
plt.tight_layout()

# 최종 결과 화면에 출력
plt.show()
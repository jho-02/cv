import cv2 as cv  # OpenCV 라이브러리를 cv라는 이름으로 불러옵니다.
import numpy as np  # 배열 생성 및 연산을 위해 NumPy를 불러옵니다.
import matplotlib.pyplot as plt  # 이미지 출력 및 시각화를 위해 matplotlib를 불러옵니다.

img = cv.imread("3week/image/coffee cup.JPG")  # 분석에 사용할 이미지를 파일 경로에서 불러옵니다.

if img is None:  # 이미지가 정상적으로 불러와졌는지 확인합니다.
    print("이미지를 불러오지 못했습니다.")
    exit()  # 이미지가 없으면 이후 처리를 할 수 없으므로 프로그램을 종료합니다.

img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# OpenCV는 이미지를 BGR 형식으로 읽기 때문에,
# matplotlib에서 올바른 색상으로 출력하기 위해 RGB로 변환합니다.

h, w = img.shape[:2]
# 이미지의 높이(h)와 너비(w)를 가져옵니다.
# 이후 마스크 생성과 영역 설정에 사용됩니다.

mask = np.zeros((h, w), np.uint8)
# GrabCut에서 사용할 마스크를 생성합니다.
# 처음에는 모든 값을 0(배경)으로 초기화합니다.

bgdModel = np.zeros((1, 65), np.float64)
# 배경 모델을 저장하기 위한 배열을 생성합니다.
# GrabCut 내부에서 배경 색상 분포를 학습하는 데 사용됩니다.

fgdModel = np.zeros((1, 65), np.float64)
# 전경 모델을 저장하기 위한 배열을 생성합니다.
# GrabCut 내부에서 전경 색상 분포를 학습하는 데 사용됩니다.

rect = (
    50,
    50,
    w - 100,
    h - 100
)
# 전경이 포함될 것으로 예상되는 초기 사각형 영역을 설정합니다.
# (x, y, width, height) 형태이며, 이미지 가장자리에서 일정 부분을 제외한 영역입니다.

cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
# GrabCut 알고리즘을 실행합니다.
# 사각형(rect) 내부는 전경 후보, 외부는 배경으로 가정하여 분할을 시작합니다.
# 5는 반복 횟수이며, 반복을 통해 더 정교한 결과를 얻습니다.

mask2 = np.where(
    (mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD),
    0,
    1
).astype("uint8")
# GrabCut 결과 마스크를 0과 1로 변환합니다.
# 확실한 배경과 배경 가능 영역은 0,
# 전경과 전경 가능 영역은 1로 설정하여 새로운 마스크를 생성합니다.

result = img_rgb * mask2[:, :, np.newaxis]
# 생성한 마스크를 원본 이미지에 적용합니다.
# mask2는 2차원 배열이므로 np.newaxis를 이용해 3채널로 확장한 뒤 곱합니다.
# 결과적으로 전경만 남고 배경은 제거됩니다.

plt.figure(figsize=(15, 5))
# 출력 화면의 전체 크기를 설정합니다.

plt.subplot(1, 3, 1)
# 1행 3열 중 첫 번째 위치에 원본 이미지를 출력합니다.
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")
# 축 정보는 필요 없으므로 숨깁니다.

plt.subplot(1, 3, 2)
# 두 번째 위치에 마스크 이미지를 출력합니다.
plt.imshow(mask2 * 255, cmap="gray")
# mask2는 0과 1 값이므로 255를 곱해 흑백 이미지로 보기 쉽게 변환합니다.
plt.title("Mask Image")
plt.axis("off")

plt.subplot(1, 3, 3)
# 세 번째 위치에 배경이 제거된 결과 이미지를 출력합니다.
plt.imshow(result)
plt.title("Background Removed")
plt.axis("off")

plt.tight_layout()
# subplot 간의 간격을 자동으로 조정하여 겹치지 않도록 합니다.

plt.show()
# 최종 결과를 화면에 출력합니다.
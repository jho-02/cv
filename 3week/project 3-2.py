import cv2 as cv  # OpenCV 라이브러리를 cv라는 이름으로 불러옵니다.
import numpy as np  # pi 값과 수치 계산에 사용할 NumPy를 불러옵니다.
import matplotlib.pyplot as plt  # 결과 이미지를 화면에 출력하기 위해 matplotlib를 불러옵니다.

img = cv.imread("3week/image/dabo.jpg")  # 직선 검출에 사용할 원본 이미지를 불러옵니다.

if img is None:  # 이미지가 정상적으로 불러와졌는지 확인합니다.
    print("이미지를 불러오지 못했습니다.")
    exit()  # 이미지가 없으면 이후 연산을 진행할 수 없으므로 프로그램을 종료합니다.

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  
# Canny 에지 검출은 밝기 변화량을 기준으로 수행되므로
# 컬러 이미지를 그레이스케일 이미지로 변환합니다.

edges = cv.Canny(gray, 100, 200)  
# Canny 알고리즘을 이용하여 에지를 검출합니다.
# 100은 낮은 임계값, 200은 높은 임계값입니다.

line_img = img.copy()  
# 검출된 직선을 그릴 이미지를 만들기 위해 원본 이미지를 복사합니다.
# 원본 이미지를 그대로 유지하기 위해 copy()를 사용합니다.

lines = cv.HoughLinesP(
    edges,                  # 입력 에지 이미지
    rho=1,                  # 거리 해상도: 1픽셀 단위로 직선을 검출합니다.
    theta=np.pi / 180,      # 각도 해상도: 1도 단위로 직선을 검출합니다.
    threshold=140,          # 직선으로 인정하기 위한 최소 투표 수입니다.
    minLineLength=45,       # 직선으로 인정할 최소 선 길이입니다.
    maxLineGap=3            # 끊어진 선분을 하나의 직선으로 연결할 최대 간격입니다.
)

if lines is not None:  # 직선이 하나라도 검출되었는지 확인합니다.
    for line in lines:  # 검출된 모든 직선에 대해 반복합니다.
        x1, y1, x2, y2 = line[0]  
        # HoughLinesP의 결과는 [[x1, y1, x2, y2]] 형태이므로
        # line[0]에서 시작점과 끝점 좌표를 꺼냅니다.

        cv.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 2)  
        # 복사한 이미지 위에 검출된 직선을 그립니다.
        # (0, 0, 255)는 빨간색, 2는 선 두께를 의미합니다.

img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)  
# OpenCV는 이미지를 BGR 형식으로 읽기 때문에
# matplotlib에서 올바른 색으로 출력하기 위해 RGB 형식으로 변환합니다.

line_img_rgb = cv.cvtColor(line_img, cv.COLOR_BGR2RGB)  
# 직선이 그려진 결과 이미지 역시 matplotlib 출력용으로 RGB 형식으로 변환합니다.

plt.figure(figsize=(12, 5))  
# 전체 출력 창의 크기를 설정합니다.

plt.subplot(1, 2, 1)  
# 1행 2열 중 첫 번째 영역에 원본 이미지를 출력합니다.
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")  
# 축 눈금은 필요하지 않으므로 숨깁니다.

plt.subplot(1, 2, 2)  
# 1행 2열 중 두 번째 영역에 직선 검출 결과 이미지를 출력합니다.
plt.imshow(line_img_rgb)
plt.title("Detected Lines")
plt.axis("off")  
# 축 눈금을 숨겨 결과만 깔끔하게 보이도록 합니다.

plt.tight_layout()  
# subplot 사이의 간격을 자동으로 정리합니다.

plt.show()  
# 최종 결과를 화면에 출력합니다.
import cv2  # OpenCV 라이브러리를 불러옵니다.
import numpy as np  # 배열이나 좌표 계산에 사용하는 NumPy를 불러옵니다.

# 이미지 불러오기
img = cv2.imread("2week/rose.png")  # 실습에 사용할 이미지를 파일에서 읽어옵니다.

# 이미지 크기 줄이기
img = cv2.resize(img, None, fx=0.5, fy=0.5)  # 화면에 너무 크게 나오지 않도록 가로, 세로를 50%로 줄입니다.

if img is None:  # 이미지가 정상적으로 불러와졌는지 확인합니다.
    print("이미지를 찾을 수 없습니다.")  # 파일을 읽지 못했을 때 오류 메시지를 출력합니다.
    exit()  # 프로그램 실행을 종료합니다.

# 이미지 크기
h, w = img.shape[:2]  # 이미지의 높이와 너비를 가져옵니다.

# 이미지 중심
center = (w // 2, h // 2)  # 회전 기준이 될 이미지 중심 좌표를 계산합니다.

# 회전 + 스케일
M = cv2.getRotationMatrix2D(center, 30, 0.8)  # 중심을 기준으로 이미지를 30도 회전하고, 크기를 0.8배로 줄이는 변환 행렬을 만듭니다.

# 평행 이동
M[0, 2] += 80  # x축 방향으로 80픽셀 오른쪽으로 이동하도록 행렬 값을 조정합니다.
M[1, 2] += -40  # y축 방향으로 40픽셀 위쪽으로 이동하도록 행렬 값을 조정합니다.

# 변환 적용
result = cv2.warpAffine(img, M, (w, h))  # 계산한 변환 행렬을 이용해 실제로 이미지를 변환합니다.

# 출력
cv2.imshow("Original Image", img)  # 원본 이미지를 화면에 출력합니다.
cv2.imshow("Transformed Image", result)  # 변환이 적용된 이미지를 화면에 출력합니다.

cv2.waitKey(0)  # 아무 키나 입력할 때까지 창이 닫히지 않도록 기다립니다.
cv2.destroyAllWindows()  # 열려 있는 모든 OpenCV 창을 닫습니다.
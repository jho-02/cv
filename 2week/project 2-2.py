import cv2
import numpy as np

# 이미지 불러오기
img = cv2.imread("2week/rose.png")

# 이미지 크기 줄이기
img = cv2.resize(img, None, fx=0.5, fy=0.5)

if img is None:
    print("이미지를 찾을 수 없습니다.")
    exit()

# 이미지 크기
h, w = img.shape[:2]

# 이미지 중심
center = (w // 2, h // 2)

# 회전 + 스케일
M = cv2.getRotationMatrix2D(center, 30, 0.8)

# 평행 이동
M[0, 2] += 80
M[1, 2] += -40

# 변환 적용
result = cv2.warpAffine(img, M, (w, h))

# 출력
cv2.imshow("Original Image", img)
cv2.imshow("Transformed Image", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
import cv2 as cv
import numpy as np

img = cv.imread("welcome.jpg") # 이미지 불러오기
if img is None: # 이미지가 정상적으로 불러와졌는지 확인
    print("이미지 없음") # 이미지가 없으면 메시지 출력
    exit() # 프로그램 종료

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # 이미지를 그레이스케일로 변환 (컬러 이미지를 흑백으로 변환)
gray_bgr = cv.cvtColor(gray, cv.COLOR_GRAY2BGR) # 그레이스케일 이미지를 BGR 형식으로 변환 (컬러 이미지와 나란히 붙이기 위해)

out = np.hstack((img, gray_bgr)) # 원본 컬러 이미지와 그레이스케일 이미지를 수평으로 붙이기 (hstack: 수평으로 배열을 합치는 함수)

# 화면에 맞게 축소 (예: 0.5배)
scale = 0.5 # 축소된 이미지 생성 (interpolation=cv.INTER_AREA: 축소할 때 좋은 품질을 제공하는 보간 방법)
out_small = cv.resize(out, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA) # 축소된 이미지 화면에 표시

cv.imshow("Color | Gray", out_small) # "Color | Gray"라는 제목으로 축소된 이미지 표시
cv.waitKey(0) # 키 입력 대기 (0: 무한 대기, 양수: 지정된 시간(ms) 동안 대기)
cv.destroyAllWindows() # 모든 OpenCV 윈도우 닫기 (프로그램 종료
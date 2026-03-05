# OpenCV Practice 01  
이미지 불러오기 및 Grayscale 변환

본 실습의 목적은 OpenCV를 이용하여 이미지를 불러오고, 해당 이미지를 Grayscale(흑백 이미지)로 변환한 뒤 원본 이미지와 함께 화면에 출력하는 것입니다.

Computer Vision은 컴퓨터가 이미지나 영상을 분석하여 의미 있는 정보를 이해하도록 하는 기술입니다. 사람은 이미지를 보면 자연스럽게 사람, 건물, 자동차, 풍경 등의 의미를 이해할 수 있지만 컴퓨터는 이미지를 단순한 숫자 데이터로 이루어진 픽셀 배열로 인식합니다.
따라서 Computer Vision의 목표는 이러한 픽셀 데이터와 실제 의미 사이의 간극을 줄이는 것입니다. 아래 그림은 사람이 보는 이미지와 컴퓨터가 인식하는 픽셀 데이터의 차이를 보여줍니다.

<img width="1158" height="646" alt="image" src="https://github.com/user-attachments/assets/c1ab8ded-991e-4f35-99a2-352c46867e31" />


OpenCV에서 이미지는 NumPy 배열 형태로 저장됩니다. 
예를 들어 이미지의 구조는 `(높이, 너비, 채널)` 형태로 표현됩니다. 예를 들어 `(510, 800, 3)`이라는 값은 높이가 510픽셀, 너비가 800픽셀이며 색상 채널이 3개라는 의미입니다. 컬러 이미지는 일반적으로 3개의 색상 채널을 가지며 OpenCV에서는 RGB 순서가 아니라 BGR(Blue, Green, Red) 순서를 사용합니다. 따라서 하나의 픽셀 값은 `(Blue, Green, Red)`입니다.
<img width="1073" height="619" alt="image" src="https://github.com/user-attachments/assets/52d371c9-58ba-40e4-93b4-c02c519cff68" />

Grayscale 이미지는 색상 정보를 제거하고 밝기 정보만 남긴 이미지입니다. 컬러 이미지는 R, G, B 세 개의 채널을 사용하지만 Grayscale 이미지는 하나의 채널만 사용합니다. 
Grayscale 이미지를 사용하는 이유는 이미지 처리 과정에서 계산량을 줄이고 특징 추출을 쉽게 만들기 때문입니다. 사람의 눈은 색상 중에서도 Green 색상에 가장 민감하기 때문에 RGB 채널을 동일한 비율로 사용하는 것이 아니라 서로 다른 가중치를 적용하여 Grayscale 값을 계산합니다. 
<img width="1150" height="649" alt="image" src="https://github.com/user-attachments/assets/46f4a84a-f02c-4240-b1a2-5ae38bfaebc4" />


이번 실습에서 구현한 코드는 다음과 같습니다.
1번 이미지 생성하기 
```python
import cv2 as cv

img = cv.imread("welcome.jpg") # 이미지 불러오기

if img is None: # 이미지가 정상적으로 불러와졌는지 확인
    print("이미지 없음")
    exit()

cv.imshow("Image", img) # 이미지 출력

cv.waitKey(0) # 키 입력 대기
cv.destroyAllWindows() # 모든 창 닫기
```
출력 이미지
<img width="1120" height="620" alt="image" src="https://github.com/user-attachments/assets/9ccaad7a-5d88-4865-83c1-6cc537fec519" />

2번 이미지를 흑백으로 출력하기 
```python
mport cv2 as cv

img = cv.imread("welcome.jpg") # 이미지 불러오기

if img is None:
    print("이미지 없음")
    exit()

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # 컬러 이미지를 흑백 이미지로 변환

cv.imshow("Gray Image", gray) # 흑백 이미지 출력

cv.waitKey(0)
cv.destroyAllWindows()
```
출력 사진
<img width="1110" height="585" alt="image" src="https://github.com/user-attachments/assets/76408641-5370-43c4-957e-5390312204ab" />

3번 컬러 이미지와 흑백이미지 출력후 이어 붙이기

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
```


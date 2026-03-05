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


첫번째 과제 실습에서 구현한 코드는 다음과 같습니다.

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
```python
import cv2 as cv
import numpy as np

img = cv.imread("welcome.jpg") # 이미지 불러오기
if img is None: # 이미지가 정상적으로 불러와졌는지 확인
    print("이미지 없음") # 이미지가 없으면 메시지 출력
    exit() # 프로그램 종료

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # 이미지를 그레이스케일로 변환 (컬러 이미지를 흑백으로 변환)
gray_bgr = cv.cvtColor(gray, cv.COLOR_GRAY2BGR) # 그레이스케일 이미지를 BGR 형식으로 변환 (컬러 이미지와 나란히 붙이기 위해)

out = np.hstack((img, gray_bgr)) # 원본 컬러 이미지와 그레이스케일 이미지를 수평으로 붙이기 (hstack: 수평으로 배열을 합치는 함수)
# 화면에 맞게 축소
scale = 0.5 # 축소된 이미지 생성 (interpolation=cv.INTER_AREA: 축소할 때 좋은 품질을 제공하는 보간 방법)
out_small = cv.resize(out, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA) # 축소된 이미지 화면에 표시

cv.imshow("Color | Gray", out_small) # "Color | Gray"라는 제목으로 축소된 이미지 표시
cv.waitKey(0) # 키 입력 대기 (0: 무한 대기, 양수: 지정된 시간(ms) 동안 대기)
cv.destroyAllWindows() # 모든 OpenCV 윈도우 닫기 (프로그램 종료
```


# OpenCV Practice 02  
페인팅 붓(Brush) 기능 구현

본 실습의 목적은 OpenCV에서 마우스 이벤트와 키보드 입력을 처리하여 이미지 위에 직접 그림을 그릴 수 있는 간단한 페인팅 프로그램을 구현하는 것입니다. 사용자는 마우스 좌클릭/우클릭으로 색상을 바꿔 그림을 그릴 수 있고, 키보드 입력을 통해 브러시 크기를 조절할 수 있습니다.

OpenCV에서 마우스 입력은 언제든 발생할 수 있기 때문에(클릭/이동/드래그 등), 이를 처리하려면 콜백 함수(callback function)가 필요합니다. OpenCV는 `cv.setMouseCallback()` 함수를 통해 특정 윈도우에서 발생하는 마우스 이벤트를 콜백 함수로 전달합니다. 
<img width="1144" height="609" alt="image" src="https://github.com/user-attachments/assets/69ebb20a-804c-4485-b47a-0487060532ed" />


또한 화면에 글자를 표시할 때는 `cv.putText()`를, 도형을 그릴 때는 `cv.circle()`/`cv.rectangle()` 등을 사용할 수 있습니다. 본 실습에서는 브러시 크기 정보를 화면에 표시하기 위해 `cv.putText()`를 사용했습니다.
<img width="1152" height="656" alt="image" src="https://github.com/user-attachments/assets/bedbb5a6-0674-4a13-98c3-51619da345af" />


2번째 과제 실습에서 구현한 코드는 다음과 같습니다

1번 그레이 화면 캔버스 생성
```python
import cv2 as cv
import numpy as np

# 그레이 캔버스 생성: (height, width) 형태
h, w = 600, 800
canvas_gray = np.full((h, w), 200, dtype=np.uint8)  # 0=검정, 255=흰색, 200=연한 회색

cv.imshow("Canvas Gray", canvas_gray)

while True:
    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv.destroyAllWindows()
```
실행화면
<img width="847" height="660" alt="image" src="https://github.com/user-attachments/assets/09dfcfcd-f9e9-4072-b08e-9b7c8c34a1e6" />


2번 그레이 화면 캔버스 생성 + 선 그리기 기능

```python
import cv2 as cv
import numpy as np

h, w = 600, 800
canvas = np.full((h, w), 200, dtype=np.uint8)  # 그레이 캔버스

drawing = False  # 드로잉 중인지 여부
prev_x, prev_y = -1, -1  # 이전 좌표(선 그리기용)

def draw_line(event, x, y, flags, param):
    global drawing, prev_x, prev_y, canvas

    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        prev_x, prev_y = x, y
        # 점 하나 찍어서 클릭한 위치도 표시되게
        cv.circle(canvas, (x, y), 1, 255, -1)

    elif event == cv.EVENT_MOUSEMOVE and drawing:
        # 이전 점과 현재 점을 선으로 연결
        cv.line(canvas, (prev_x, prev_y), (x, y), 255, 2)  # 255=흰색, 두께=2
        prev_x, prev_y = x, y

    elif event == cv.EVENT_LBUTTONUP:
        drawing = False

cv.namedWindow("Canvas Gray")
cv.setMouseCallback("Canvas Gray", draw_line)

while True:
    cv.imshow("Canvas Gray", canvas)
    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv.destroyAllWindows()
```
<img width="858" height="663" alt="image" src="https://github.com/user-attachments/assets/d900f350-5bb5-45fc-a0ac-f666fef9fb4a" />

3번 그레이 화면 캔버스 생성 + 선 그리기 기능과 선 크기 조절기능
```python
import cv2 as cv
import numpy as np

h, w = 600, 800
canvas = np.full((h, w), 200, dtype=np.uint8)  # 그레이 캔버스

drawing = False
prev_x, prev_y = -1, -1

brush = 5  # 선 두께(브러시 크기)

def draw_line(event, x, y, flags, param):
    global drawing, prev_x, prev_y, canvas, brush

    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        prev_x, prev_y = x, y
        cv.circle(canvas, (x, y), max(1, brush // 2), 255, -1)

    elif event == cv.EVENT_MOUSEMOVE and drawing:
        cv.line(canvas, (prev_x, prev_y), (x, y), 255, brush)  # 255=흰색
        prev_x, prev_y = x, y

    elif event == cv.EVENT_LBUTTONUP:
        drawing = False

cv.namedWindow("Canvas Gray")
cv.setMouseCallback("Canvas Gray", draw_line)

while True:
    temp = canvas.copy()  # 표시용 복사본(텍스트 올리기)
    cv.putText(
        temp,
        f"brush:{brush}  (+/-)  q:quit",
        (10, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        0.9,
        0,   # 글씨색(검정)
        2
    )

    cv.imshow("Canvas Gray", temp)

    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('+') or key == ord('='):
        brush = min(15, brush + 1)
    elif key == ord('-') or key == ord('_'):
        brush = max(1, brush - 1)

cv.destroyAllWindows()
```
<img width="837" height="657" alt="image" src="https://github.com/user-attachments/assets/c628fce0-f5b5-4007-bf81-a6ce92878469" />


4번 완성본 
```python
import cv2 as cv

# 이미지 불러오기
img = cv.imread("welcome.jpg")

if img is None: # 이미지가 정상적으로 불러와졌는지 확인
    print("이미지 없음") 
    exit() #  프로그램 종료

brush = 5 # 브러시 크기 초기값
drawing = False # 마우스 드로잉 상태
color = (255,0,0) # 초기 색상 (파랑) - OpenCV는 BGR 순서로 색상을 표현

def draw(event, x, y, flags, param): # 마우스 이벤트 콜백 함수
    global drawing, color, brush, img # 전역 변수 사용 선언

    if event == cv.EVENT_LBUTTONDOWN: # 왼쪽 버튼 클릭 시 드로잉 시작
        drawing = True # 드로잉 상태로 변경
        color = (255,0,0)  # 파랑
        cv.circle(img,(x,y),brush,color,-1) # 원 그리기 (이미지, 중심 좌표, 반지름, 색상, 채우기)

    elif event == cv.EVENT_RBUTTONDOWN: # 오른쪽 버튼 클릭 시 드로잉 시작
        drawing = True # 드로잉 상태로 변경
        color = (0,0,255)  # 빨강
        cv.circle(img,(x,y),brush,color,-1) # 원 그리기 (이미지, 중심 좌표, 반지름, 색상, 채우기)

    elif event == cv.EVENT_MOUSEMOVE and drawing: # 마우스 이동 중 드로잉 상태일 때
        cv.circle(img,(x,y),brush,color,-1) # 원 그리기 (이미지, 중심 좌표, 반지름, 색상, 채우기)

    elif event == cv.EVENT_LBUTTONUP or event == cv.EVENT_RBUTTONUP: # 왼쪽 또는 오른쪽 버튼에서 손을 뗄 때
        drawing = False # 드로잉 상태 종료


cv.namedWindow("Paint") # 윈도우 생성
cv.setMouseCallback("Paint", draw) # "Paint" 윈도우에 마우스 이벤트 콜백 함수 등록

while True: # 메인 루프

    temp = img.copy() # 원본 이미지를 복사하여 임시 이미지 생성

    cv.putText(temp,f"brush:{brush}",(10,30), # 텍스트 추가 (이미지, 텍스트, 위치, 폰트, 크기, 색상, 두께)
               cv.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)# 텍스트 추가 (이미지, 텍스트, 위치, 폰트, 크기, 색상, 두께)

    cv.imshow("Paint",temp) # "Paint" 윈도우에 임시 이미지 표시

    key = cv.waitKey(1) & 0xFF # 키 입력 대기 및 하위 8비트만 추출

    if key == ord('q'): # 'q' 키를 누르면 루프 종료
        break # 프로그램 종료

    elif key == ord('+') or key == ord('='): # '+' 또는 '=' 키를 누르면 브러시 크기 증가
        brush = min(15, brush+1) # 브러시 크기 최대값 15로 제한

    elif key == ord('-') or key == ord('_'): # '-' 또는 '_' 키를 누르면 브러시 크기 감소
        brush = max(1, brush-1) # 브러시 크기 최소값 1로 제한

cv.destroyAllWindows() # 모든 OpenCV 윈도우 닫기
```
# OpenCV Practice 03  
ROI(Region of Interest) 선택 및 저장

본 실습의 목적은 OpenCV에서 마우스 드래그 이벤트를 이용해 이미지의 특정 영역(ROI, 관심영역)을 선택하고 선택된 영역을 별도의 창으로 출력하거나 파일로 저장하는 기능을 구현하는 것입니다. 
또한 키보드 입력을 통해 ROI를 초기화하거나 저장하도록 하여 마우스 이벤트와 키보드 이벤트를 함께 처리하는 흐름을 익히는 것을 목표로 합니다.

ROI(Region of Interest)는 이미지 전체가 아니라 분석에 필요한 특정 영역만 선택하여 처리하는 방법입니다. 예를 들어 얼굴 인식, 객체 검출, 문자 인식과 같은 작업에서는 이미지 전체를 처리하기보다 관심영역만 선택하여 연산량을 줄이고 필요한 정보에 집중할 수 있습니다.
OpenCV에서 이미지는 NumPy 배열 형태로 저장되므로 ROI는 배열 슬라이싱을 이용해 간단히 추출할 수 있습니다(예: `roi = image[y1:y2, x1:x2]`). 또한 마우스 드래그 기반 선택을 구현하기 위해 `cv.setMouseCallback()`을 사용하며, 드래그 중 선택 영역을 화면에 표시하기 위해 `cv.rectangle()`을 활용할 수 있습니다.




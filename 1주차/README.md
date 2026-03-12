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

3번 완성코드 + 코드 설명
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


4번 완성코드 + 코드 설명 
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

마우스를 이용한 ROI(관심 영역) 선택

본 실습의 목적은 OpenCV에서 마우스 이벤트를 이용하여 이미지에서 특정 영역을 선택하고, 선택된 영역을 별도의 이미지로 추출하는 기능을 구현하는 것입니다.

Computer Vision에서는 이미지 전체가 아니라 특정 관심 영역(ROI, Region Of Interest)만 분석하는 경우가 많습니다. 예를 들어 얼굴 인식, 객체 검출, 영상 추적 등의 알고리즘에서는 이미지 전체를 처리하기보다 필요한 부분만 선택하여 처리하는 것이 효율적입니다.

ROI(Region Of Interest)는 이미지에서 특정 부분만 선택하여 처리하는 방법입니다. 
전체 이미지를 처리하는 것보다 필요한 영역만 분석하면 계산량을 줄이고 처리 속도를 높일 수 있습니다. 
예를 들어 얼굴 인식에서는 얼굴 부분만 ROI로 선택하여 분석하고, 객체 검출에서는 자동차나 사람과 같은 특정 객체 영역만 추출하여 처리합니다. 
따라서 ROI는 다양한 Computer Vision 알고리즘에서 매우 중요한 개념입니다.

OpenCV에서는 마우스 이벤트를 이용하여 사용자가 직접 이미지 위에서 영역을 선택할 수 있습니다. 
마우스를 클릭하면 시작 좌표가 기록되고 드래그하면 사각형 영역이 표시됩니다. 마우스를 놓으면 해당 영역이 ROI로 추출됩니다.

아래 그림은 마우스를 클릭한 위치에서 사각형 영역을 선택하는 개념을 보여줍니다.
<img width="1147" height="638" alt="image" src="https://github.com/user-attachments/assets/534028d4-4c1f-4127-8595-dd923737b511" />


또한 마우스를 드래그하면 사각형의 크기가 변경되며 선택 영역을 시각적으로 확인할 수 있습니다.

<img width="1138" height="637" alt="image" src="https://github.com/user-attachments/assets/c27d33d5-bbfe-4f5d-aba6-1f63d21502de" />


3번째 과제 실습에서 구현한 코드는 다음과 같습니다.
1단계 이미지 생성하기

```python
import cv2 as cv

img = cv.imread("welcome.jpg")  # 이미지 불러오기
if img is None:
    print("이미지 없음")
    exit()

img = cv.resize(img, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)  # 화면에 맞게 축소

cv.imshow("Image", img)  # 이미지 출력
cv.waitKey(0)            # 키 입력 대기
cv.destroyAllWindows()   # 창 닫기
```

<img width="901" height="629" alt="image" src="https://github.com/user-attachments/assets/b352fd9d-6d67-4092-a912-79fbc94f4852" />

2단계 드레그 및 추출하기
```python
import cv2 as cv

img = cv.imread("welcome.jpg")
if img is None:
    print("이미지 없음")
    exit()

img = cv.resize(img, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)

orig = img.copy()     # 기준 이미지
disp = img.copy()     # 표시용 이미지

drawing = False
x0, y0 = -1, -1
roi = None

def mouse(event, x, y, flags, param):
    global drawing, x0, y0, disp, roi

    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        x0, y0 = x, y

    elif event == cv.EVENT_MOUSEMOVE and drawing:
        temp = orig.copy()
        cv.rectangle(temp, (x0, y0), (x, y), (0, 255, 0), 2)
        cv.imshow("Image", temp)

    elif event == cv.EVENT_LBUTTONUP and drawing:
        drawing = False
        x1, y1 = x, y

        x_min, x_max = min(x0, x1), max(x0, x1)
        y_min, y_max = min(y0, y1), max(y0, y1)

        roi = orig[y_min:y_max, x_min:x_max]

        if roi.size == 0:
            roi = None
            disp = orig.copy()
            print("ROI가 너무 작음")
            return

        disp = orig.copy()
        cv.rectangle(disp, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        cv.imshow("ROI", roi)

cv.namedWindow("Image")
cv.setMouseCallback("Image", mouse)

while True:
    cv.imshow("Image", disp)
    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv.destroyAllWindows()
```
<img width="896" height="627" alt="image" src="https://github.com/user-attachments/assets/e3f22936-0025-4869-a90d-5d257b96065c" />

3단계 완성코드 + 코드 설명
```python
import cv2 as cv 

img = cv.imread("welcome.jpg") # 이미지 불러오기

if img is None: # 이미지가 정상적으로 불러와졌는지 확인
    print("이미지 없음") # 이미지가 없으면 메시지 출력
    exit() # 프로그램 종료


img = cv.resize(img, None, fx=0.5, fy=0.5) # 이미지 축소 (fx, fy: 가로, 세로 축소 비율, interpolation=cv.INTER_AREA: 축소할 때 좋은 품질을 제공하는 보간 방법)

orig = img.copy() # 원본 이미지 복사 (ROI 선택 후 원본 이미지를 유지하기 위해)

drawing = False #   마우스 드래그 상태를 나타내는 변수 (False: 드래그 중 아님, True: 드래그 중)
x0, y0 = -1, -1 # 드래그 시작점의 좌표를 저장하는 변수 (초기값은 -1로 설정하여 유효하지 않은 좌표로 표시)
roi = None # 선택된 ROI(Region of Interest)를 저장하는 변수 (초기값은 None으로 설정하여 ROI가 선택되지 않았음을 나타냄)


def mouse(event, x, y, flags, param): # 마우스 이벤트를 처리하는 콜백 함수 (event: 마우스 이벤트 종류, x, y: 마우스 좌표, flags: 이벤트 플래그, param: 추가 매개변수)
    global drawing, x0, y0, img, roi # 전역 변수 사용 (drawing, x0, y0, img, roi 변수를 함수 내에서 수정하기 위해 global 키워드 사용)

   
    if event == cv.EVENT_LBUTTONDOWN and roi is None: # 왼쪽 버튼이 눌렸고 ROI가 선택되지 않은 경우 (드래그 시작)
        drawing = True # 드래그 상태로 변경 (True로 설정하여 드래그 중임을 나타냄)
        x0, y0 = x, y # 드래그 시작점의 좌표 저장 (x, y는 마우스 이벤트로 전달된 현재 좌표)

    elif event == cv.EVENT_MOUSEMOVE and drawing: # 마우스가 움직이고 드래그 중인 경우 (드래그 진행)
        temp = orig.copy() # 원본 이미지 복사 (드래그 진행 중에 원본 이미지를 유지하기 위해)
        cv.rectangle(temp, (x0, y0), (x, y), (0,255,0), 2) # 드래그 시작점 (x0, y0)과 현재 마우스 좌표 (x, y)를 이용하여 사각형 그리기 (초록색, 두께 2)
        cv.imshow("Image", temp) # 드래그 진행 중인 이미지를 화면에 표시 (temp 이미지에는 현재 드래그 상태가 반영된 사각형이 그려져 있음)

    elif event == cv.EVENT_LBUTTONUP and drawing: # 왼쪽 버튼이 떼어지고 드래그 중인 경우 (드래그 종료)
        drawing = False # 드래그 상태 종료 (False로 설정하여 드래그 중이 아님을 나타냄)

        x1, y1 = x, y # 드래그 종료점의 좌표 저장 (x, y는 마우스 이벤트로 전달된 현재 좌표)

        x_min, x_max = min(x0,x1), max(x0,x1) # 드래그 시작점과 종료점의 x 좌표를 이용하여 최소값과 최대값 계산 (x_min: 드래그 영역의 왼쪽 경계, x_max: 드래그 영역의 오른쪽 경계)
        y_min, y_max = min(y0,y1), max(y0,y1) # 드래그 시작점과 종료점의 y 좌표를 이용하여 최소값과 최대값 계산 (y_min: 드래그 영역의 상단 경계, y_max: 드래그 영역의 하단 경계)

        roi = orig[y_min:y_max, x_min:x_max] # 선택된 ROI(Region of Interest)를 원본 이미지에서 추출 (y_min:y_max, x_min:x_max 영역을 슬라이싱하여 roi 변수에 저장)

        img = orig.copy() #  원본 이미지 복사 (ROI 선택 후 원본 이미지를 유지하기 위해)
        cv.rectangle(img, (x_min,y_min), (x_max,y_max), (0,255,0), 2) # 선택된 ROI 영역을 원본 이미지에 사각형으로 표시 (초록색, 두께 2)

        cv.imshow("ROI", roi) # 선택된 ROI를 화면에 표시 (roi 이미지에는 선택된 영역이 포함되어 있음)


cv.namedWindow("Image") # "Image"라는 이름의 윈도우 생성 (마우스 이벤트를 처리하기 위해 윈도우가 필요함)
cv.setMouseCallback("Image", mouse) # "Image" 윈도우에 마우스 콜백 함수 등록 (마우스 이벤트가 발생할 때 mouse 함수가 호출되도록 설정)

while True: # 무한 루프를 사용하여 프로그램이 종료될 때까지 계속 실행 (사용자가 'q' 키를 눌러 종료할 때까지)

    cv.imshow("Image", img) # 현재 이미지를 화면에 표시 (img 이미지에는 ROI 선택 상태가 반영되어 있음)

    key = cv.waitKey(1) & 0xFF # 키 입력 대기 (1ms 동안 대기, & 0xFF: 키 입력을 8비트로 제한하여 처리)

    if key == ord('q'):# 'q' 키가 눌렸을 때 루프 종료 (프로그램 종료)
        break # 루프 종료 (프로그램 종료)
 
    elif key == ord('r'):# 'r' 키가 눌렸을 때 이미지와 ROI 초기화 (리셋)
        img = orig.copy() # 원본 이미지 복사 (이미지와 ROI를 초기 상태로 되돌리기 위해)
        roi = None # ROI 초기화 (None으로 설정하여 ROI가 선택되지 않은 상태로 되돌리기)
        print("리셋") # 리셋 메시지 출력 (이미지와 ROI가 초기화되었음을 알림)

    elif key == ord('s'): # 's' 키가 눌렸을 때 선택된 ROI 저장 (save)
        if roi is not None: # ROI가 선택된 경우 (roi 변수가 None이 아닌 경우)
            cv.imwrite("roi.png", roi) # 선택된 ROI를 "roi.png" 파일로 저장 (roi 이미지가 roi.png 파일로 저장됨)
            print("roi.png 저장됨") # 저장 완료 메시지 출력 (roi.png 파일이 저장되었음을 알림)
        else:  # ROI가 선택되지 않은 경우 (roi 변수가 None인 경우)
            print("ROI 없음") # ROI가 선택되지 않았음을 알리는 메시지 출력 (저장할 ROI가 없음을 알림)

cv.destroyAllWindows() # 모든 OpenCV 윈도우 닫기 (프로그램 종료)


```

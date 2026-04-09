과제 1 : Sobel Edge Detection

과제 설명

Edge Detection은 이미지에서 물체의 경계나 밝기 변화가 큰 부분을 찾아내는 과정입니다.
사람은 이미지를 보면 사물의 형태와 윤곽을 자연스럽게 구분할 수 있지만, 컴퓨터는 이미지를 단순한 픽셀 값의 집합으로 인식하기 때문에 이러한 구조를 직접 이해하지 못합니다. 따라서 Computer Vision에서는 픽셀 값이 급격하게 변하는 부분을 찾아 물체의 형태를 파악하는 과정이 중요하며, 이를 위해 Edge Detection이 널리 사용됩니다.
과제에서는 OpenCV를 이용하여 이미지에서 에지를 검출하였습니다.
먼저 입력 이미지를 Grayscale로 변환한 뒤, Sobel 연산자를 사용하여 x 방향과 y 방향의 밝기 변화량을 각각 계산하였습니다. 이후 두 방향의 기울기 값을 결합하여 최종적인 에지 강도를 구하고, 이를 시각화하여 원본 이미지와 비교하였습니다. 실습 과제에서도 edgeDetectionImage를 그레이스케일로 변환하고, Sobel 필터를 사용해 x축과 y축 방향의 에지를 검출한 후, cv.magnitude()를 통해 에지 강도를 계산하여 원본 이미지와 함께 시각화하도록 요구하고 있습니다.


배경 지식
Edge Detection

에지(edge)는 이미지에서 밝기 값이 급격하게 변하는 지점을 의미합니다. 일반적으로 물체의 내부는 비교적 천천히 변하지만, 물체와 배경이 만나는 경계에서는 명암 차이가 크게 나타나기 때문에 에지가 형성됩니다. 강의 자료에서도 에지 검출 알고리즘은 “물체 내부는 명암이 서서히 변하고, 물체 경계는 명암이 급격히 변한다”는 특성을 이용한다고 설명합니다.
에지 검출은 다음과 같은 작업의 기본 단계로 사용됩니다.

- 객체의 윤곽 추출
- 영상 분할
- 직선 및 형태 검출
- 특징점 추출
- 객체 인식 및 장면 분석

에지 검출은 이미지 전체를 그대로 해석하는 것이 아니라, 그중에서도 구조를 이해하는 데 중요한 경계 정보만 선택적으로 추출하는 과정이라고 볼 수 있습니다.

Gradient

에지 검출에서 가장 중요한 개념 중 하나는 Gradient(기울기)입니다.
Gradient는 이미지에서 픽셀 값이 얼마나 빠르게 변하는지를 나타내는 값으로, 변화가 클수록 에지일 가능성이 높다고 볼 수 있습니다.

일반적으로 기울기는 두 방향으로 계산됩니다.

<img width="217" height="69" alt="image" src="https://github.com/user-attachments/assets/c7b1f4ea-bc50-4301-900f-cebed50955dd" />

이 두 값을 이용하면 전체 에지 강도를 다음과 같이 계산할 수 있습니다.

<img width="229" height="66" alt="image" src="https://github.com/user-attachments/assets/789cbce0-cc1c-4ce2-84bf-25a398e1c5b6" />

강의 자료에서는 이 값을 Edge Strength라고 설명하며, x축과 y축 방향의 에지 정보를 함께 고려한 최종 에지 강도라고 제시하고 있습니다.

Sobel Operator

Sobel 연산자는 영상의 1차 미분을 근사하여 에지를 검출하는 대표적인 방법입니다.
단순한 미분 연산은 노이즈에 민감하고 방향 정보가 부족하다는 한계가 있지만, 강의 자료에서는 이를 3×3 필터로 확장한 Prewitt, Sobel 같은 연산자를 사용하면 더 안정적이고 방향 정보를 제공할 수 있다고 설명합니다. 또한 가까운 픽셀에 더 큰 가중치를 주기 때문에 노이즈를 줄이고 더 부드러운 gradient를 계산할 수 있다고 정리하고 있습니다.

Sobel 연산의 특징은 다음과 같습니다.
- x 방향과 y 방향의 변화량을 각각 계산할 수 있음
- 에지의 방향성을 일부 파악할 수 있음
- 단순 미분보다 안정적임
- 구현이 직관적이고 계산량이 비교적 작음
이번 과제에서 사용한 Sobel 연산은 3×3 커널을 기반으로 수행되며, 실습 과제에서도 cv.Sobel()의 ksize를 3 또는 5로 설정하도록 나와있습니다.

Grayscale 변환

컬러 이미지는 일반적으로 B, G, R 세 개의 채널로 구성됩니다.
하지만 에지 검출은 색상 자체보다 밝기 변화량을 기반으로 수행되기 때문에, 먼저 이미지를 Grayscale로 변환하여 하나의 밝기 정보만 사용하는 것이 일반적입니다. 
이전 OpenCV 실습 자료에서도 컬러 이미지를 그레이스케일로 변환하는 과정이 기본적인 전처리 단계로 제시되며
밝기 기반 분석을 위해 cv.cvtColor()를 사용하는 방법이 소개되어 있습니다.
그레이스케일로 변환하면 연산이 단순해지고, 에지 검출에 필요한 핵심 정보인 밝기 변화에 집중할 수 있습니다.

Edge Strength와 자료형

Sobel 연산 결과는 단순한 0과 255 값이 아니라, 음수와 실수를 포함하는 gradient 값으로 계산됩니다.
강의 자료에서도

<img width="64" height="35" alt="image" src="https://github.com/user-attachments/assets/f02c65d5-8235-45c9-8ccd-0901e06ce313" />


에지 강도, 에지 방향은 음수를 포함하는 실수이므로 32비트 실수형(cv.CV_32F) 또는 이에 준하는 실수형 자료형을 사용하는 것이 안전하다고 설명합니다.

실습 과제에서는 cv.CV_64F를 사용하여 Sobel 값을 계산하고, 이후 cv.convertScaleAbs()를 통해 시각화 가능한 uint8 이미지로 변환하도록 요구하고 있습니다.
이 과정을 통해 계산 단계에서는 정확한 실수값을 유지하고, 출력 단계에서는 사람이 보기 쉬운 형태로 변환할 수 있습니다.

주요 코드 설명

1. 이미지 불러오기
먼저 cv.imread()를 사용하여 입력 이미지를 불러왔습니다.
이미지를 정상적으로 읽지 못하는 경우 이후 연산을 수행할 수 없으므로, None 여부를 확인하여 예외 처리를 하였습니다. 실습 과제에서도 cv.imread()를 사용하여 이미지를 불러오는 것이 첫 번째 요구사항으로 제시되어 있습니다.
```
img = cv.imread("3week/image/edgeDetectionImage.jpg")

if img is None:
    print("이미지를 불러오지 못했습니다.")
    exit()
```

2. 그레이스케일 변환
에지 검출은 밝기 변화량을 기준으로 수행되므로, 컬러 이미지를 그대로 사용하지 않고 cv.cvtColor()를 이용하여 그레이스케일 이미지로 변환하였습니다.
이 과정은 에지 검출의 전처리 단계로, 색상 정보 대신 구조적 밝기 정보만 사용하기 위한 것입니다.
```
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
```


3. Sobel 필터를 이용한 x, y 방향 에지 검출
그레이스케일 이미지에 대해 Sobel 연산을 각각 x 방향과 y 방향으로 수행하였습니다.
- x 방향 Sobel : 세로 경계 검출에 유리
- y 방향 Sobel : 가로 경계 검출에 유리
실습 과제에서는 cv.Sobel()을 사용하여 x축 방향은 (cv.CV_64F, 1, 0), y축 방향은 (cv.CV_64F, 0, 1) 형태로 계산하도록 요구하고 있습니다.
이 과정을 통해 이미지의 각 방향에서 밝기 변화량을 얻을 수 있으며, 특정 방향에 강한 경계가 어디에 있는지 알 수 있습니다.
```
grad_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
grad_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
```

4. 에지 강도 계산
x 방향과 y 방향의 Sobel 결과를 각각 구한 뒤, cv.magnitude()를 사용하여 두 기울기를 결합하였습니다.
이 과정은 한 방향의 정보만 보는 것이 아니라 전체적인 경계 강도를 계산하기 위한 단계입니다.
```
magnitude = cv.magnitude(grad_x, grad_y)
edge_strength = cv.convertScaleAbs(magnitude)
```

5. 결과를 시각화 가능한 형태로 변환
Sobel과 magnitude의 결과는 실수형 값으로 계산되므로, 이를 그대로 출력하면 일반 이미지처럼 보기 어렵습니다.
따라서 cv.convertScaleAbs()를 이용하여 uint8 형태로 변환하였습니다.
```
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
```
6. 원본 이미지와 결과 이미지 시각화
마지막으로 Matplotlib를 사용하여 원본 이미지와 에지 강도 이미지를 나란히 출력하였습니다.
이때 에지 강도 이미지는 흑백 이미지이므로 plt.imshow()에서 cmap='gray'를 사용하여 시각화하였습니다. 이 역시 실습 과제에서 명시적으로 제시된 요구사항입니다.
이를 통해 원본 이미지에서는 보이지 않던 경계 정보가 에지 강도 이미지에서 어떻게 강조되는지를 한눈에 비교할 수 있습니다.
```
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(edge_strength, cmap="gray")
plt.title("Edge Strength")
plt.axis("off")

plt.tight_layout()
plt.show()
```

실습 이미지
<img width="990" height="325" alt="image" src="https://github.com/user-attachments/assets/b7d1cf8a-1d82-47ff-a78e-5cfd1da5337b" />

전체코드
```
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
```

과제 2 : Canny Edge Detection과 Hough Transform을 이용한 직선 검출

과제 설명

본 과제에서는 이미지에서 직선 구조를 검출하기 위해 Canny Edge Detection과 Hough Transform을 활용하였습니다.
이미지에는 다양한 색상과 질감 정보가 포함되어 있기 때문에, 바로 직선을 검출하기보다는 먼저 경계 정보를 추출하는 과정이 필요합니다.

따라서 먼저 입력 이미지를 그레이스케일로 변환한 후, Canny 알고리즘을 이용하여 에지를 검출하였습니다. 이후 HoughLinesP 함수를 사용하여 에지 이미지에서 직선으로 판단되는 선분을 추출하였으며
검출된 직선을 원본 이미지 위에 표시하였습니다. 마지막으로 원본 이미지와 직선 검출 결과 이미지를 함께 출력하여 비교하였습니다.

배경 지식

Canny Edge Detection

Canny Edge Detection은 이미지에서 경계를 안정적으로 검출하기 위한 대표적인 알고리즘입니다.
단순히 밝기 변화가 큰 부분을 찾는 것이 아니라, 노이즈를 줄이고 실제 경계일 가능성이 높은 부분만 선택적으로 남기는 특징을 가지고 있습니다.
일반적으로 다음과 같은 과정을 통해 에지를 검출합니다.
- 밝기 변화 계산
- 강한 에지와 약한 에지 구분
- 연결된 에지만 최종적으로 유지
이 방법은 이후 직선 검출과 같은 고차원 처리 과정에서 중요한 전처리 단계로 사용됩니다.

Hough Transform

Hough Transform은 이미지에서 직선과 같은 기하학적 구조를 검출하기 위한 방법입니다.
에지로 검출된 픽셀들이 특정 직선을 형성할 가능성을 누적하여 가장 가능성이 높은 직선을 찾아내는 방식으로 동작합니다.

본 과제에서는 Probabilistic Hough Transform 방식인 cv.HoughLinesP()를 사용하였습니다.
이 방법은 직선을 선분 형태로 반환하므로, 이미지 위에 직접 시각화하기에 적합합니다.

주요 코드 설명
1. 이미지 불러오기

이미지를 불러온 뒤, 정상적으로 로드되었는지 확인합니다.
```
img = cv.imread("3week/image/dabo.jpg")

if img is None:
    print("이미지를 불러오지 못했습니다.")
    exit()
```
2. 그레이스케일 변환

에지 검출은 밝기 기반으로 수행되므로, 컬러 이미지를 그레이스케일로 변환합니다.
```
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
```

3. Canny Edge Detection

Canny 알고리즘을 사용하여 이미지의 경계(에지)를 추출합니다.
```
edges = cv.Canny(gray, 100, 200)
```

4. 직선 검출을 위한 이미지 복사

원본 이미지를 보존하기 위해 복사본을 생성합니다.
```
line_img = img.copy()
```

5. Hough Transform을 이용한 직선 검출
```
lines = cv.HoughLinesP(
    edges,
    rho=1,
    theta=np.pi / 180,
    threshold=140,
    minLineLength=45,
    maxLineGap=3
)
```
Canny로 검출된 에지 이미지에서 직선 형태를 갖는 선분을 찾아냅니다.

6. 검출된 직선 그리기 
```
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
```

검출된 직선을 원본 이미지 복사본 위에 빨간색 선으로 표시합니다.

7. 이미지 출력을 위한 색상 변환
```
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
line_img_rgb = cv.cvtColor(line_img, cv.COLOR_BGR2RGB)
```
Matplotlib에서 올바른 색으로 출력하기 위해 BGR을 RGB로 변환합니다.

8. 결과 시각화
```
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(line_img_rgb)
plt.title("Detected Lines")
plt.axis("off")

plt.tight_layout()
plt.show()
```
원본 이미지와 직선 검출 결과를 나란히 출력하여 비교할 수 있도록 합니다.

실습 결과

<img width="1186" height="479" alt="image" src="https://github.com/user-attachments/assets/2a978279-59ce-44e0-a368-548ebe45dfd4" />

과제 2 전체코드
```
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
```

과제 3 : GrabCut을 이용한 영역 분할

과제 설명

3번 과제에서는 이미지에서 전경 객체와 배경을 분리하기 위해 GrabCut 알고리즘을 사용하였습니다.
이미지 분할(Image Segmentation)은 컴퓨터 비전에서 매우 중요한 작업 중 하나로 이미지 안에 있는 객체와 배경을 구분하여 의미 있는 영역을 추출하는 과정입니다. 
사람은 이미지를 보면 컵, 접시, 숟가락과 같은 물체를 자연스럽게 구분할 수 있지만 컴퓨터는 이를 픽셀 단위의 데이터로 인식하기 때문에 전경과 배경을 나누는 별도의 과정이 필요합니다.
또한 OpenCV의 grabCut() 함수를 이용하여 컵 이미지에서 전경 객체를 분리하였습니다.
먼저 이미지를 불러오고, GrabCut 수행에 필요한 마스크와 모델 배열을 초기화하였습니다. 이후 객체가 포함될 것으로 예상되는 초기 사각형 영역을 지정한 뒤
GrabCut 알고리즘을 이용하여 전경과 배경을 분리하였습니다. 마지막으로 생성된 마스크를 이용하여 배경이 제거된 결과 이미지를 만들고, 원본 이미지, 마스크 이미지, 배경 제거 이미지를 함께 출력하여 결과를 확인하였습니다.

배경 지식

Image Segmentation

Image Segmentation은 이미지를 여러 개의 의미 있는 영역으로 나누는 과정입니다.
예를 들어 하나의 이미지 안에 컵, 접시, 숟가락, 배경이 함께 있을 때, 이들을 각각 구분하여 객체 단위로 나누는 것이 영상 분할의 목적입니다.
영상 분할은 다음과 같은 분야에서 활용됩니다.

- 객체 추출
- 배경 제거
- 이미지 편집
- 의료 영상 분석
- 자율주행 및 장면 이해
  
Image Segmentation은 단순히 픽셀을 처리하는 것이 아니라 이미지 안에서 어떤 부분이 하나의 객체를 이루는지 구분하는 과정이라고 볼 수 있습니다

GrabCut 알고리즘

GrabCut은 전경과 배경을 분리하기 위한 대표적인 대화식(Interactive) 영상 분할 알고리즘입니다.
사용자가 객체가 포함된 대략적인 사각형 영역을 지정하면, 알고리즘이 이 정보를 바탕으로 전경과 배경을 반복적으로 추정하여 더 정교한 분할 결과를 만들어냅니다.

GrabCut의 기본 동작 과정은 다음과 같습니다.

- 초기 사각형 영역 설정
- 사각형 바깥 영역을 배경으로 가정
- 사각형 내부를 전경 후보로 가정
- 전경과 배경의 색상 분포를 모델링
- 반복적으로 전경/배경을 갱신하며 최적의 분할 결과 생성

이 방법은 사용자가 픽셀 단위로 직접 라벨링하지 않아도 된다는 장점이 있습니다.

Gaussian Mixture Model (GMM)

GrabCut은 전경과 배경의 색상 분포를 표현하기 위해 Gaussian Mixture Model을 사용합니다.
이는 픽셀 값의 분포를 하나의 단순한 값으로 보는 것이 아니라 여러 개의 가우시안 분포를 조합하여 보다 유연하게 표현하는 방법입니다.
전경과 배경은 일반적으로 서로 다른 색상 분포를 가지므로 GrabCut은 이러한 차이를 이용하여 어떤 픽셀이 전경에 속하는지 어떤 픽셀이 배경에 속하는지를 추정합니다.

GrabCut 마스크 값

GrabCut 수행 후 생성되는 마스크는 각 픽셀을 다음과 같이 구분합니다.

- cv.GC_BGD : 확실한 배경
- cv.GC_PR_BGD : 배경일 가능성이 높은 영역
- cv.GC_FGD : 확실한 전경
- cv.GC_PR_FGD : 전경일 가능성이 높은 영역

3번 과제에서는 이 값을 다시 0과 1로 변환하여 사용하였습니다.

- 배경 → 0
- 전경 → 1

이 과정을 통해 최종적으로 전경만 남는 마스크 이미지를 만들 수 있습니다.


주요 코드 설명

1. 이미지 불러오기

먼저 cv.imread()를 사용하여 입력 이미지를 불러옵니다.
이미지가 정상적으로 읽히지 않을 경우 이후 연산을 수행할 수 없으므로 None 여부를 확인합니다.
```
img = cv.imread("3week/image/coffee cup.JPG")

if img is None:
    print("이미지를 불러오지 못했습니다.")
    exit()
```

2. RGB 변환

OpenCV는 이미지를 BGR 형식으로 읽어오지만, Matplotlib는 RGB 형식으로 이미지를 출력합니다.
따라서 올바른 색상으로 출력하기 위해 RGB 형식으로 변환합니다.
```
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
```

3. 이미지 크기 저장

이미지의 높이와 너비를 저장하여 이후 마스크 생성과 사각형 영역 설정에 활용합니다.
```
h, w = img.shape[:2]
```

4. 마스크 및 모델 초기화

GrabCut 수행에 필요한 마스크와 두 개의 모델 배열을 초기화합니다.
```
mask = np.zeros((h, w), np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
```

- mask : 각 픽셀이 전경인지 배경인지 저장하는 배열
- bgdModel : 배경 모델
- fgdModel : 전경 모델

이 두 모델은 GrabCut 내부에서 색상 분포를 추정하는 데 사용됩니다.

5. 초기 사각형 영역 설정

객체가 포함될 것으로 예상되는 영역을 사각형 형태로 지정합니다.
```
rect = (
    50,
    50,
    w - 100,
    h - 100
)
```
이 사각형은 (x, y, width, height) 형식이며,
이미지 가장자리에서 일정 거리 안쪽을 전경 후보 영역으로 설정하는 역할을 합니다.

6. GrabCut 수행

설정한 사각형 영역을 기반으로 GrabCut 알고리즘을 적용합니다.
```
cv.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_RECT)
```
여기서 각 인자의 의미는 다음과 같습니다.

- img : 입력 이미지
- mask : 전경/배경 마스크
- rect : 초기 사각형 영역
- bgdModel : 배경 모델
- fgdModel : 전경 모델
- 5 : 반복 횟수
- cv.GC_INIT_WITH_RECT : 사각형 기반 초기화 모드

7. 마스크 값을 0과 1로 변환

GrabCut 결과에서 배경과 전경을 구분하여 새로운 마스크를 생성합니다.
```
mask2 = np.where(
    (mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD),
    0,
    1
).astype("uint8")
```

이 코드는 확실한 배경과 배경 가능성이 높은 영역을 0으로,
그 외 영역을 1로 변환하여 최종 마스크 이미지를 생성합니다.

8. 배경 제거 이미지 생성

생성한 마스크를 원본 이미지에 적용하여 전경만 남긴 결과 이미지를 만듭니다.
```
result = img_rgb * mask2[:, :, np.newaxis]
```
mask2는 2차원 배열이므로, 컬러 이미지와 곱하기 위해 np.newaxis를 사용하여 차원을 확장합니다.

9. 결과 시각화

원본 이미지, 마스크 이미지, 배경 제거 이미지를 나란히 출력하여 결과를 비교합니다.
```
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(mask2 * 255, cmap="gray")
plt.title("Mask Image")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(result)
plt.title("Background Removed")
plt.axis("off")

plt.tight_layout()
plt.show()
```
마스크 이미지는 흑백으로 보기 쉽게 mask2 * 255 형태로 출력하였습니다.

실습결과

<img width="1493" height="429" alt="image" src="https://github.com/user-attachments/assets/eb244075-bdb9-4c33-9290-11da54eb3135" />

전체코드
```
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
```

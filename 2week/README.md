과제 1
1. 과제 1 : Camera Calibration
   
과제 설명
카메라는 3차원 세계(World Coordinates)를 2차원 이미지(Image Plane)로 투영하는 장치입니다. 
이 과정에서 카메라 렌즈의 특성 때문에 이미지에는 렌즈 왜곡(Lens Distortion)이 발생할 수 있습니다.
Camera Calibration은 이러한 왜곡을 보정하기 위해 카메라의 내부 파라미터(Camera Matrix)와 왜곡 계수(Distortion Coefficients)를 계산하는 과정입니다.

본 과제에서는 체크보드 패턴 이미지를 이용하여 체크보드의 코너를 검출하고,
실제 3차원 좌표와 이미지에서의 2차원 좌표를 이용하여 카메라의 내부 파라미터를 계산하였습니다. 
이후 계산된 파라미터를 이용하여 이미지의 왜곡을 보정하고, 원본 이미지와 보정된 이미지를 비교하여 결과를 확인하였습니다.

배경 지식
Camera Calibration

Camera Calibration은 카메라의 내부 파라미터(Intrinsic Parameters)와 렌즈 왜곡(Lens Distortion)을 추정하는 과정입니다.
실제 카메라는 이상적인 핀홀 카메라 모델이 아니기 때문에 렌즈에 의해 이미지가 왜곡될 수 있습니다. 이러한 왜곡은 주로 다음과 같은 형태로 나타납니다.
- Radial Distortion : 이미지 가장자리에서 발생하는 방사형 왜곡

- Tangential Distortion : 렌즈와 이미지 센서의 정렬 문제로 발생하는 왜곡

체크보드 패턴은 일정한 간격을 가진 격자 구조이기 때문에 이미지에서 코너를 쉽게 검출할 수 있어 
Camera Calibration에 널리 사용됩니다.

Camera Matrix
Camera Matrix는 카메라의 내부 파라미터를 나타내는 3×3 행렬입니다.

<img width="227" height="94" alt="image" src="https://github.com/user-attachments/assets/c6cc298a-7daf-47de-9a77-913d899aa714" />

여기서

<img width="350" height="62" alt="image" src="https://github.com/user-attachments/assets/3afc8651-6a37-438d-becf-c094c693fa2a" />

을 의미합니다.

본 과제에서 계산된 Camera Matrix는 다음과 같습니다

<img width="325" height="86" alt="image" src="https://github.com/user-attachments/assets/51e4ec2f-c92a-4474-ae00-4ca0a0457da0" />

Distortion Coefficients

<img width="335" height="66" alt="image" src="https://github.com/user-attachments/assets/ec49b798-90a5-455b-849e-5492187cbc1f" />



렌즈 왜곡은 일반적으로 다음과 같은 계수로 표현됩니다.

<img width="290" height="58" alt="image" src="https://github.com/user-attachments/assets/6ef527bb-f0d9-4978-ab84-45bbbf982d33" />

본 실습에서 계산된 Distortion Coefficients는 다음과 같습니다.

<img width="471" height="44" alt="image" src="https://github.com/user-attachments/assets/5d615ebe-3459-41d7-a64b-9b8783d9f36b" />

이 값을 이용하여 이미지의 왜곡을 보정할 수 있습니다.

주요 코드 설명

1.체크보드 코너 검출
체크보드 패턴의 코너를 검출하기 위해 OpenCV의 findChessboardCorners() 함수를 사용하였습니다.
```
ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
```
이 함수는 이미지에서 체크보드 패턴을 찾아 내부 코너의 위치를 반환합니다.

2. 카메라 캘리브레이션 수행
검출된 코너 좌표를 이용하여 카메라의 내부 행렬과 왜곡 계수를 계산하였습니다.
```
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    img_size,
    None,
    None
)
```
K : Camera Matrix (카메라 내부 행렬)
dist : Distortion Coefficients (렌즈 왜곡 계수)
rvecs : 회전 벡터
tvecs : 이동 벡터

3. 이미지 왜곡 보정
계산된 파라미터를 이용하여 원본 이미지의 렌즈 왜곡을 제거하였습니다.
```
undistorted = cv2.undistort(test_img, K, dist, None, newK)
```
과제 코드
```
import cv2  # OpenCV 라이브러리를 불러옵니다.
import numpy as np  # 배열 계산을 위해 NumPy를 불러옵니다.
import glob  # 여러 이미지 파일 경로를 한 번에 불러오기 위해 사용합니다.

# 체크보드의 내부 코너 개수를 설정합니다.
CHECKERBOARD = (9, 6)

# 체크보드 한 칸의 실제 크기를 mm 단위로 설정합니다.
square_size = 25.0

# 코너 위치를 더 정밀하게 찾기 위한 종료 조건을 설정합니다.
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 체크보드의 실제 3차원 좌표를 저장할 배열을 생성합니다.
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)

# 체크보드 평면 위의 (x, y) 좌표를 격자 형태로 생성합니다.
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# 생성한 좌표에 실제 한 칸 크기를 곱해 실제 크기를 반영합니다.
objp *= square_size

# 모든 이미지의 실제 3차원 좌표를 저장할 리스트입니다.
objpoints = []

# 모든 이미지의 2차원 코너 좌표를 저장할 리스트입니다.
imgpoints = []

# calibration_images 폴더 안의 left로 시작하는 jpg 파일들을 모두 불러옵니다.
images = glob.glob("2week/calibration_images/left*.jpg")

# 이미지 크기를 저장할 변수를 미리 만들어 둡니다.
img_size = None

# -----------------------------
# 1. 체크보드 코너 검출
# -----------------------------
for fname in images:  # 불러온 모든 체크보드 이미지에 대해 반복합니다.
    img = cv2.imread(fname)  # 현재 이미지를 읽어옵니다.

    if img is None:  # 이미지가 정상적으로 읽히지 않은 경우를 확인합니다.
        print("이미지를 읽을 수 없습니다:", fname)  # 어떤 파일에서 문제가 생겼는지 출력합니다.
        continue  # 현재 파일은 건너뛰고 다음 파일로 넘어갑니다.

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 코너 검출을 위해 이미지를 그레이스케일로 변환합니다.
    img_size = gray.shape[::-1]  # 이미지 크기를 (width, height) 형태로 저장합니다.

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)  # 체크보드 코너를 검출합니다.

    if ret:  # 코너 검출에 성공한 경우입니다.
        objpoints.append(objp)  # 실제 3차원 좌표를 저장합니다.

        corners2 = cv2.cornerSubPix(  # 검출한 코너 위치를 더 정확하게 보정합니다.
            gray,  # 입력 그레이스케일 이미지입니다.
            corners,  # 처음 검출한 코너 좌표입니다.
            (11, 11),  # 코너 주변 탐색 윈도우 크기입니다.
            (-1, -1),  # dead zone을 사용하지 않겠다는 의미입니다.
            criteria  # 앞에서 설정한 종료 조건을 사용합니다.
        )

        imgpoints.append(corners2)  # 보정된 2차원 코너 좌표를 저장합니다.

        print("코너 검출 성공:", fname)  # 어떤 이미지에서 코너 검출에 성공했는지 출력합니다.
    else:  # 코너 검출에 실패한 경우입니다.
        print("코너 검출 실패:", fname)  # 실패한 파일 이름을 출력합니다.

if len(objpoints) == 0 or len(imgpoints) == 0:  # 유효한 코너를 찾은 이미지가 하나도 없는지 확인합니다.
    print("유효한 체크보드 코너를 찾은 이미지가 없습니다.")  # 오류 메시지를 출력합니다.
    exit()  # 프로그램을 종료합니다.

# -----------------------------
# 2. 카메라 캘리브레이션
# -----------------------------
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(  # 검출한 좌표들을 이용해 카메라 캘리브레이션을 수행합니다.
    objpoints,  # 실제 3차원 좌표 목록입니다.
    imgpoints,  # 이미지에서 검출한 2차원 좌표 목록입니다.
    img_size,  # 이미지 크기입니다.
    None,  # 초기 카메라 행렬은 지정하지 않습니다.
    None  # 초기 왜곡 계수도 지정하지 않습니다.
)

print("Camera Matrix K:")  # 카메라 내부 행렬 출력을 위한 제목입니다.
print(K)  # 계산된 Camera Matrix를 출력합니다.

print("\nDistortion Coefficients:")  # 왜곡 계수 출력을 위한 제목입니다.
print(dist)  # 계산된 왜곡 계수를 출력합니다.

# -----------------------------
# 3. 왜곡 보정 시각화
# -----------------------------
test_img = cv2.imread(images[0])  # 첫 번째 체크보드 이미지를 예시 이미지로 다시 불러옵니다.

if test_img is None:  # 테스트 이미지를 정상적으로 읽었는지 확인합니다.
    print("왜곡 보정용 테스트 이미지를 읽을 수 없습니다.")  # 읽기 실패 메시지를 출력합니다.
    exit()  # 프로그램을 종료합니다.

h, w = test_img.shape[:2]  # 테스트 이미지의 높이와 너비를 저장합니다.

newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))  
# 왜곡 보정에 사용할 새로운 카메라 행렬을 계산합니다.

undistorted = cv2.undistort(test_img, K, dist, None, newK)  
# 원본 이미지에 왜곡 보정을 적용합니다.

cv2.imwrite("original_image.jpg", test_img)  # 원본 이미지를 파일로 저장합니다.
cv2.imwrite("undistorted_image.jpg", undistorted)  # 왜곡 보정된 이미지를 파일로 저장합니다.

cv2.imshow("Original Image", test_img)  # 원본 이미지를 화면에 출력합니다.
cv2.imshow("Undistorted Image", undistorted)  # 왜곡 보정된 이미지를 화면에 출력합니다.

cv2.waitKey(0)  # 키 입력이 들어올 때까지 창을 유지합니다.
cv2.destroyAllWindows()  # 모든 OpenCV 창을 닫습니다.
```
<img width="1597" height="638" alt="image" src="https://github.com/user-attachments/assets/40c6bc1b-22d0-4542-9c6b-b6de9939a7ed" />

과제 2 : Image Transformation

과제설명
Image Transformation은 이미지의 위치, 방향, 크기 등을 변경하는 과정입니다.
Computer Vision에서는 이러한 변환을 행렬(Matrix)을 이용하여 표현하며
이미지의 픽셀 좌표를 변환 행렬과 곱하여 새로운 위치로 이동시키는 방식으로 이루어집니다.

이번 과제에서는 OpenCV를 이용하여 이미지에 다음과 같은 변환을 수행하였습니다.
- 회전 (Rotation)
- 크기 조정 (Scaling)
- 평행 이동 (Translation)
이러한 변환은 Affine Transformation 으로 표현할 수 있으며 OpenCV의 변환 행렬을 이용하여 이미지를 변환하였습니다.
코드에서는 먼저 이미지 중심을 기준으로 회전과 스케일 변환을 수행한 후 변환 행렬의 이동 값을 수정하여 평행 이동을 추가하였습니다.
마지막으로 warpAffine() 함수를 사용하여 변환 행렬을 이미지에 적용하였습니다.

배경지식

Image Transformation

Image Transformation은 이미지의 좌표를 변환하여 새로운 형태의 이미지를 생성하는 과정입니다.
이미지의 각 픽셀 좌표 (𝑥,𝑦)는 변환 행렬을 이용하여 새로운 좌표 (𝑥′,𝑦′)로 변환됩니다.
이러한 변환은 Computer Vision, 이미지 정렬(Image Alignment), 객체 추적(Object Tracking), 
영상 보정(Image Rectification) 등의 다양한 분야에서 사용됩니다.

Affine Transformation

Affine Transformation은 다음과 같은 변환을 포함하는 기하학적 변환입니다.

- Rotation (회전)
- Scaling (확대 및 축소)
- Translation (평행 이동)
- Shearing (기울이기)

Affine Transformation은 직선과 평행성을 유지하는 특징을 가지고 있습니다.

Affine 변환은 다음과 같은 행렬 형태로 표현됩니다.

<img width="244" height="97" alt="image" src="https://github.com/user-attachments/assets/54434cb3-5000-4a27-b1ea-13fbddb4986c" />

여기서

<img width="311" height="71" alt="image" src="https://github.com/user-attachments/assets/882cd522-c954-44f5-88bc-ab25bd3ebc3a" />

입니다.

주요 코드 설명

1. 회전 및 스케일 변환

이미지 중심을 기준으로 회전과 크기 조정을 수행하였습니다.
```
M = cv2.getRotationMatrix2D(center, angle, scale)
```
각 변수의 의미는 다음과 같습니다.
- center : 회전 중심
- angle : 회전 각도
- scale : 크기 비율
getRotationMatrix2D() 함수는 회전과 스케일 변환을 동시에 적용할 수 있는 Affine 변환 행렬을 생성합니다.

2. 평행 이동 (Translation)

회전 행렬의 이동 요소를 수정하여 이미지의 위치를 이동시켰습니다.
```
M[0,2] += 80
M[1,2] += -40
```
이 코드는 이미지의 x 방향과 y 방향으로 이동을 추가하여 평행 이동 변환을 수행합니다.

3. 이미지 변환 적용

생성된 변환 행렬을 이용하여 실제 이미지 변환을 수행하였습니다.
```
result = cv2.warpAffine(image, M, (width, height))
```
warpAffine() 함수는 변환 행렬을 이용하여 이미지를 새로운 위치와 형태로 변환합니다.
이 함수는 입력 이미지의 모든 픽셀을 변환 행렬에 따라 새로운 좌표로 이동시켜 최종 변환된 이미지를 생성합니다.

```
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
```
<img width="1480" height="535" alt="image" src="https://github.com/user-attachments/assets/273599cb-194d-41b0-b9dd-1a8bc6f0372e" />

과제 3 : Stereo Vision을 이용한 Depth 추정

과제설명
Stereo Vision은 두 개의 카메라로 촬영한 이미지를 이용하여 물체까지의 거리를 계산하는 방법입니다.
사람의 양쪽 눈이 서로 다른 위치에서 사물을 바라보는 것과 같은 원리를 이용하여 깊이 정보를 추정할 수 있습니다.

두 이미지에서 동일한 물체는 서로 다른 위치에 나타나는데, 이러한 위치 차이를 Disparity라고 합니다. 이 disparity 값을 이용하면 물체까지의 거리인 Depth를 계산할 수 있습니다.

Depth는 다음과 같은 관계식으로 계산됩니다.

<img width="224" height="72" alt="image" src="https://github.com/user-attachments/assets/235a5f30-1f5e-4a84-a5c5-266891a24bd4" />

f : 카메라의 초점 거리 (Focal Length)
B : 두 카메라 사이의 거리 (Baseline)
disparity : 좌우 이미지에서의 동일 물체의 위치 차이
3번 과제에서는 좌우 스테레오 이미지를 이용하여 disparity map을 계산하고, 이를 이용하여 depth 정보를 계산하였습니다. 또한 이미지의 특정 영역(Painting, Frog, Teddy)에 대해 평균 disparity와 평균 depth를 계산하여 각 물체의 상대적인 거리를 비교하였습니다.

배경지식

Stereo Vision

Stereo Vision은 두 개의 카메라로 촬영된 이미지를 이용하여 3차원 공간의 깊이 정보를 추정하는 기술입니다.

두 카메라가 서로 다른 위치에서 같은 물체를 촬영하면 이미지에서 물체의 위치가 약간 다르게 나타나게 됩니다. 이 위치 차이를 이용하면 삼각측량(Triangulation)을 통해 물체까지의 거리를 계산할 수 있습니다.

Stereo Vision은 자율주행 자동차, 로봇 비전, 3D 재구성, 증강현실(AR) 같은 분야에서 활용됩니다.

Disparity Map

Disparity Map은 좌우 이미지에서 동일한 픽셀의 위치 차이를 나타내는 이미지입니다.

- disparity 값이 클수록 물체는 카메라에 가까움
- disparity 값이 작을수록 물체는 카메라에서 멀리 있음

disparity map을 시각화하면 이미지의 깊이 구조를 직관적으로 확인할 수 있습니다.

주요 코드 설명

1. Disparity 계산

StereoBM 알고리즘을 이용하여 disparity map을 계산하였습니다.
```
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
disparity = stereo.compute(left_gray, right_gray)
```
StereoBM 알고리즘은 블록 매칭(Block Matching) 방식으로 좌우 이미지의 픽셀 패턴을 비교하여 동일한 물체의 위치 차이를 계산합니다. 이 과정에서 두 이미지 사이의 픽셀 위치 차이가 계산되어 disparity map이 생성됩니다.

2. Depth 계산

계산된 disparity 값을 이용하여 depth 값을 계산하였습니다.
```
depth = (f * B) / disparity
```
Stereo Vision의 기본 공식을 이용하여 각 픽셀의 깊이 값을 계산합니다.

<img width="122" height="40" alt="image" src="https://github.com/user-attachments/assets/bc0ddfb3-f7b4-453b-8751-ec4a2a47dece" />

여기서 disparity 값이 클수록 물체는 카메라에 더 가까운 위치에 있으며 disparity 값이 작을수록 물체는 더 먼 위치에 있습니다.

3. ROI 영역 거리 분석

이미지에서 특정 물체 영역(Painting, Frog, Teddy)을 ROI(Region of Interest)로 설정하고 각 영역의 평균 disparity와 평균 depth를 계산하였습니다.
이를 통해 세 물체 중 어떤 물체가 카메라에 가장 가까운지, 그리고 어떤 물체가 가장 멀리 있는지를 비교할 수 있습니다.

```
import cv2  # OpenCV 라이브러리를 불러옵니다.
import numpy as np  # 배열 계산과 수치 연산을 위해 NumPy를 불러옵니다.
from pathlib import Path  # 출력 폴더를 만들고 경로를 다루기 위해 Path를 사용합니다.

# 출력 폴더 생성
output_dir = Path("./outputs")  # 결과 이미지를 저장할 폴더 경로를 설정합니다.
output_dir.mkdir(parents=True, exist_ok=True)  # outputs 폴더가 없으면 새로 생성합니다.

# 좌/우 이미지 불러오기
left_color = cv2.imread("2week/left.png")  # 왼쪽 이미지를 불러옵니다.
right_color = cv2.imread("2week/right.png")  # 오른쪽 이미지를 불러옵니다.

if left_color is None or right_color is None:  # 두 이미지가 정상적으로 불러와졌는지 확인합니다.
    raise FileNotFoundError("좌/우 이미지를 찾지 못했습니다.")  # 둘 중 하나라도 없으면 오류를 발생시킵니다.


# 카메라 파라미터
f = 700.0  # 카메라의 초점 거리를 설정합니다.
B = 0.12  # 두 카메라 사이의 거리(Baseline)를 설정합니다.

# ROI 설정
rois = {
    "Painting": (55, 50, 130, 110),  # Painting 영역의 x, y, width, height를 설정합니다.
    "Frog": (90, 265, 230, 95),  # Frog 영역의 좌표와 크기를 설정합니다.
    "Teddy": (310, 35, 115, 90)  # Teddy 영역의 좌표와 크기를 설정합니다.
}

# 그레이스케일 변환
left_gray = cv2.cvtColor(left_color, cv2.COLOR_BGR2GRAY)  # 왼쪽 컬러 이미지를 그레이스케일로 변환합니다.
right_gray = cv2.cvtColor(right_color, cv2.COLOR_BGR2GRAY)  # 오른쪽 컬러 이미지를 그레이스케일로 변환합니다.

# -----------------------------
# 1. Disparity 계산
# -----------------------------
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)  # StereoBM 알고리즘 객체를 생성합니다.
disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0  # 좌우 이미지의 disparity map을 계산하고, 실수형으로 변환한 뒤 16으로 나눕니다.

# -----------------------------
# 2. Depth 계산
# Z = fB / d
# -----------------------------
depth_map = np.zeros_like(disparity, dtype=np.float32)  # disparity와 같은 크기의 depth map 배열을 생성합니다.
valid_mask = disparity > 0  # disparity가 0보다 큰 유효한 픽셀만 선택합니다.
depth_map[valid_mask] = (f * B) / disparity[valid_mask]  # 유효한 disparity 값에 대해 depth를 계산합니다.

# -----------------------------
# 3. ROI별 평균 disparity / depth 계산
# -----------------------------
results = {}  # 각 ROI의 평균 disparity와 평균 depth를 저장할 딕셔너리입니다.

for name, (x, y, w, h) in rois.items():  # 설정한 각 ROI에 대해 반복합니다.
    disp_roi = disparity[y:y+h, x:x+w]  # 현재 ROI 영역의 disparity 값을 잘라냅니다.
    depth_roi = depth_map[y:y+h, x:x+w]  # 현재 ROI 영역의 depth 값을 잘라냅니다.

    valid_disp_roi = disp_roi[disp_roi > 0]  # 유효한 disparity 값만 선택합니다.
    valid_depth_roi = depth_roi[depth_roi > 0]  # 유효한 depth 값만 선택합니다.

    mean_disparity = float(np.mean(valid_disp_roi)) if valid_disp_roi.size > 0 else 0.0  # ROI의 평균 disparity를 계산합니다.
    mean_depth = float(np.mean(valid_depth_roi)) if valid_depth_roi.size > 0 else 0.0  # ROI의 평균 depth를 계산합니다.

    results[name] = {  # 계산한 결과를 딕셔너리에 저장합니다.
        "mean_disparity": mean_disparity,
        "mean_depth": mean_depth
    }

# -----------------------------
# 4. 결과 출력
# -----------------------------
print("=== ROI별 평균 Disparity / Depth ===")  # 결과 출력 제목입니다.
for name, value in results.items():  # 각 ROI 결과를 하나씩 출력합니다.
    print(f"{name}")  # ROI 이름을 출력합니다.
    print(f"  Mean Disparity: {value['mean_disparity']:.4f}")  # 평균 disparity를 소수점 넷째 자리까지 출력합니다.
    print(f"  Mean Depth    : {value['mean_depth']:.4f}")  # 평균 depth를 소수점 넷째 자리까지 출력합니다.

closest_roi = max(results, key=lambda k: results[k]["mean_disparity"])  # 평균 disparity가 가장 큰 ROI를 찾습니다.
farthest_roi = min(results, key=lambda k: results[k]["mean_disparity"])  # 평균 disparity가 가장 작은 ROI를 찾습니다.

print("\n=== 거리 해석 ===")  # 거리 해석 제목을 출력합니다.
print(f"가장 가까운 ROI: {closest_roi}")  # 가장 가까운 ROI를 출력합니다.
print(f"가장 먼 ROI   : {farthest_roi}")  # 가장 먼 ROI를 출력합니다.

# -----------------------------
# 5. disparity 시각화
# 가까울수록 빨강 / 멀수록 파랑
# -----------------------------
disp_tmp = disparity.copy()  # disparity 값을 시각화용으로 복사합니다.
disp_tmp[disp_tmp <= 0] = np.nan  # 0 이하의 유효하지 않은 값은 NaN으로 바꿉니다.

if np.all(np.isnan(disp_tmp)):  # 모든 disparity 값이 유효하지 않은지 확인합니다.
    raise ValueError("유효한 disparity 값이 없습니다.")  # 유효한 값이 없으면 오류를 발생시킵니다.

d_min = np.nanpercentile(disp_tmp, 5)  # disparity 값의 하위 5% 지점을 최소값으로 사용합니다.
d_max = np.nanpercentile(disp_tmp, 95)  # disparity 값의 상위 95% 지점을 최대값으로 사용합니다.

if d_max <= d_min:  # 최대값이 최소값보다 작거나 같은 비정상적인 경우를 확인합니다.
    d_max = d_min + 1e-6  # 아주 작은 값을 더해 나눗셈 오류를 방지합니다.

disp_scaled = (disp_tmp - d_min) / (d_max - d_min)  # disparity 값을 0~1 범위로 정규화합니다.
disp_scaled = np.clip(disp_scaled, 0, 1)  # 정규화된 값을 0~1 범위로 제한합니다.

disp_vis = np.zeros_like(disparity, dtype=np.uint8)  # 시각화용 8비트 이미지 배열을 생성합니다.
valid_disp = ~np.isnan(disp_tmp)  # 유효한 disparity 값이 있는 위치를 찾습니다.
disp_vis[valid_disp] = (disp_scaled[valid_disp] * 255).astype(np.uint8)  # 유효한 disparity 값을 0~255 범위로 변환합니다.

disparity_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)  # disparity 이미지를 컬러맵으로 변환합니다.

# -----------------------------
# 6. depth 시각화
# 가까울수록 빨강 / 멀수록 파랑
# -----------------------------
depth_vis = np.zeros_like(depth_map, dtype=np.uint8)  # depth 시각화를 위한 8비트 배열을 생성합니다.

if np.any(valid_mask):  # 유효한 depth 값이 하나라도 있는지 확인합니다.
    depth_valid = depth_map[valid_mask]  # 유효한 depth 값만 추출합니다.

    z_min = np.percentile(depth_valid, 5)  # depth 값의 하위 5% 지점을 최소값으로 사용합니다.
    z_max = np.percentile(depth_valid, 95)  # depth 값의 상위 95% 지점을 최대값으로 사용합니다.

    if z_max <= z_min:  # 최대값과 최소값이 비정상적인 경우를 확인합니다.
        z_max = z_min + 1e-6  # 아주 작은 값을 더해 나눗셈 오류를 방지합니다.

    depth_scaled = (depth_map - z_min) / (z_max - z_min)  # depth 값을 0~1 범위로 정규화합니다.
    depth_scaled = np.clip(depth_scaled, 0, 1)  # 정규화 결과를 0~1 범위로 제한합니다.

    # depth는 클수록 멀기 때문에 반전
    depth_scaled = 1.0 - depth_scaled  # 가까운 물체가 더 뜨겁게 보이도록 값을 반전합니다.
    depth_vis[valid_mask] = (depth_scaled[valid_mask] * 255).astype(np.uint8)  # 유효한 depth 값을 0~255 범위로 변환합니다.

depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)  # depth 이미지를 컬러맵으로 변환합니다.

# -----------------------------
# 7. Left / Right 이미지에 ROI 표시
# -----------------------------
left_vis = left_color.copy()  # 왼쪽 이미지 복사본을 만듭니다.
right_vis = right_color.copy()  # 오른쪽 이미지 복사본을 만듭니다.

# -----------------------------
# 8. 저장
# -----------------------------
cv2.imwrite(str(output_dir / "original.png"), left_color)  # 원본 왼쪽 이미지를 저장합니다.
cv2.imwrite(str(output_dir / "disparity_map_color.png"), disparity_color)  # 컬러 disparity map을 저장합니다.

# -----------------------------
# 9. 출력
# -----------------------------
cv2.imshow("Original", left_color)  # 원본 이미지를 화면에 출력합니다.
cv2.imshow("Disparity map", disparity_color)  # disparity map 결과를 화면에 출력합니다.

cv2.waitKey(0)  # 키 입력이 들어올 때까지 창을 유지합니다.
cv2.destroyAllWindows()  # 모든 OpenCV 창을 닫습니다.
```
<img width="1131" height="510" alt="image" src="https://github.com/user-attachments/assets/887401e2-1065-4054-beae-2cac7a8434c4" />

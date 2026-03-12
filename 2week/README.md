과제 1
1. 과제 1 : Camera Calibration
과제 설명

카메라는 3차원 세계(World Coordinates)를 2차원 이미지(Image Plane)로 투영하는 장치입니다.
이 과정에서 카메라 렌즈의 특성 때문에 이미지에는 렌즈 왜곡(Lens Distortion) 이 발생할 수 있습니다.

Camera Calibration은 이러한 왜곡을 보정하기 위해 카메라의 내부 파라미터(Camera Matrix) 와 왜곡 계수(Distortion Coefficients) 를 계산하는 과정입니다..

본 과제에서는 체크보드 이미지를 이용하여 체크보드의 코너를 검출하고, 실제 3차원 좌표와 이미지에서의 2차원 좌표를 이용하여 카메라의 내부 파라미터를 계산하였다. 이후 계산된 파라미터를 이용하여 이미지의 왜곡을 보정하였습니다.

주요 코드 설명
1.체크보드 코너 검출
체크보드 패턴의 코너를 검출하기 위해 OpenCV의 findChessboardCorners() 함수를 사용하였습니다.
ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
이 함수는 이미지에서 체크보드 패턴을 찾아 내부 코너의 위치를 반환합니다.

2. 카메라 캘리브레이션 수행

검출된 코너 좌표를 이용하여 카메라의 내부 행렬과 왜곡 계수를 계산하였습니다.
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    img_size,
    None,
    None
)
K : Camera Matrix (카메라 내부 행렬)

dist : Distortion Coefficients (렌즈 왜곡 계수)

3. 이미지 왜곡 보정

계산된 파라미터를 이용하여 원본 이미지의 렌즈 왜곡을 제거하였습니다.
undistorted = cv2.undistort(test_img, K, dist, None, newK)
'''python
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
'''
<img width="1597" height="638" alt="image" src="https://github.com/user-attachments/assets/40c6bc1b-22d0-4542-9c6b-b6de9939a7ed" />

과제 2 : Image Transformation

과제 2번은 Image Transformation은 이미지의 위치, 방향, 크기 등을 변경하는 과정입니다.
Computer Vision에서는 이러한 변환을 행렬(Matrix) 을 이용하여 표현합니다.
이번 과제에서는 이미지에 대해 다음과 같은 변환을 수행하였습니다.
- 회전 (Rotation)
- 크기 조정 (Scaling)
- 평행 이동 (Translation)
이러한 변환은 Affine Transformation 으로 표현할 수 있으며, OpenCV의 변환 행렬을 이용하여 이미지를 변환하였습니다.
주요 코드 설명

1. 회전 및 스케일 변환

이미지 중심을 기준으로 회전과 크기 조정을 수행하였습니다.
M = cv2.getRotationMatrix2D(center, angle, scale)
center : 회전 중심
angle : 회전 각도
scale : 크기 비율

2. 이미지 변환 적용

변환 행렬을 이용하여 이미지를 변환하였습니다.
result = cv2.warpAffine(image, M, (width, height))
warpAffine() 함수는 변환 행렬을 이용하여 이미지를 새로운 위치와 형태로 변환합니다.

과제 2
<img width="1480" height="535" alt="image" src="https://github.com/user-attachments/assets/273599cb-194d-41b0-b9dd-1a8bc6f0372e" />

과제 3 : Stereo Vision을 이용한 Depth 추정

3번 과제는 Stereo Vision은 두 개의 카메라로 촬영한 이미지를 이용하여 물체까지의 거리를 계산하는 방법입니다.
두 이미지에서 동일한 물체의 위치 차이를 Disparity 라고 하며, 이 disparity 값을 이용하면 물체까지의 거리(Depth)를 계산할 수 있습니다.
Depth는 다음과 같은 관계식으로 계산됩니다.
Depth = (f × B) / disparity
f : 카메라의 초점 거리
B : 두 카메라 사이 거리(Baseline)
disparity : 좌우 이미지에서의 위치 차이

3번 과제에서는 좌우 스테레오 이미지를 이용하여 disparity map을 계산하고, 이를 이용하여 depth 정보를 계산하였습니다.
또한 이미지의 특정 영역(Painting, Frog, Teddy)에 대해 평균 disparity와 평균 depth를 계산하여 각 물체의 상대적인 거리를 비교하였습니다.
주요 코드 설명

1. Disparity 계산

StereoBM 알고리즘을 이용하여 disparity map을 계산하였습니다.
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
disparity = stereo.compute(left_gray, right_gray)
이 과정에서 두 이미지 사이의 픽셀 위치 차이가 계산됩니다.

2. Depth 계산

계산된 disparity 값을 이용하여 depth 값을 계산하였습니다.
depth = (f * B) / disparity
disparity 값이 클수록 물체는 카메라에 더 가까운 위치에 있습니다.


<img width="1131" height="510" alt="image" src="https://github.com/user-attachments/assets/887401e2-1065-4054-beae-2cac7a8434c4" />

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
import cv2
import numpy as np
import glob

# 체크보드 내부 코너 개수
CHECKERBOARD = (9, 6)

# 체크보드 한 칸 실제 크기 (mm)
square_size = 25.0

# 코너 정밀화 조건
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 실제 좌표 생성
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size

# 저장할 좌표
objpoints = []
imgpoints = []

images = glob.glob("2week/calibration_images/left*.jpg")

img_size = None

# -----------------------------
# 1. 체크보드 코너 검출
# -----------------------------
for fname in images:
    img = cv2.imread(fname)

    if img is None:
        print("이미지를 읽을 수 없습니다:", fname)
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_size = gray.shape[::-1]

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria
        )

        imgpoints.append(corners2)

        print("코너 검출 성공:", fname)
    else:
        print("코너 검출 실패:", fname)

if len(objpoints) == 0 or len(imgpoints) == 0:
    print("유효한 체크보드 코너를 찾은 이미지가 없습니다.")
    exit()

# -----------------------------
# 2. 카메라 캘리브레이션
# -----------------------------
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    img_size,
    None,
    None
)

print("Camera Matrix K:")
print(K)

print("\nDistortion Coefficients:")
print(dist)

# -----------------------------
# 3. 왜곡 보정 시각화
# -----------------------------
test_img = cv2.imread(images[0])

if test_img is None:
    print("왜곡 보정용 테스트 이미지를 읽을 수 없습니다.")
    exit()

h, w = test_img.shape[:2]

newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
undistorted = cv2.undistort(test_img, K, dist, None, newK)

cv2.imwrite("original_image.jpg", test_img)
cv2.imwrite("undistorted_image.jpg", undistorted)

cv2.imshow("Original Image", test_img)
cv2.imshow("Undistorted Image", undistorted)

cv2.waitKey(0)
cv2.destroyAllWindows()
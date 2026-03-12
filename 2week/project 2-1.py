import cv2
import numpy as np
import glob

# 체크보드 내부 코너 개수 (가로 9개, 세로 6개)
CHECKERBOARD = (9, 6)

# 체크보드 한 칸의 실제 크기 (mm)
square_size = 25.0

# 코너 위치를 더 정밀하게 찾기 위한 종료 조건
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 체크보드의 실제 3D 좌표를 저장할 배열 생성
# z축은 0으로 두고 평면 위의 점들만 사용
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)

# (0,0), (1,0), (2,0) ... 형태의 격자 좌표 생성
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# 실제 한 칸 크기를 반영하여 mm 단위로 변환
objp *= square_size

# 모든 이미지에 대한 실제 3D 좌표를 저장할 리스트
objpoints = []

# 모든 이미지에 대한 이미지 좌표(2D 코너 위치)를 저장할 리스트
imgpoints = []

# calibration_images 폴더 안의 left로 시작하는 jpg 파일들을 모두 불러옴
images = glob.glob("calibration_images/left*.jpg")

# 이미지 크기를 저장할 변수
img_size = None

# 불러온 이미지 개수 출력
print("찾은 이미지 개수:", len(images))

# 이미지가 하나도 없으면 프로그램 종료
if len(images) == 0:
    print("calibration_images 폴더에서 이미지를 찾지 못했습니다.")
    print("파이썬 파일 위치와 폴더 구조를 확인하세요.")
    exit()

# -----------------------------
# 1. 체크보드 코너 검출
# -----------------------------
for fname in images:
    # 이미지 읽기
    img = cv2.imread(fname)

    # 이미지가 정상적으로 읽히지 않으면 건너뜀
    if img is None:
        print("이미지를 읽을 수 없습니다:", fname)
        continue

    # 컬러 이미지를 흑백 이미지로 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 이미지 크기 저장 (width, height)
    img_size = gray.shape[::-1]

    # 체크보드 코너 찾기
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    # 코너를 찾은 경우
    if ret:
        # 실제 좌표 저장
        objpoints.append(objp)

        # 찾은 코너를 더 정밀하게 보정
        corners2 = cv2.cornerSubPix(
            gray,              # 입력 흑백 이미지
            corners,           # 초기 코너 위치
            (11, 11),          # 탐색 윈도우 크기
            (-1, -1),          # zero zone
            criteria           # 종료 조건
        )

        # 보정된 이미지 좌표 저장
        imgpoints.append(corners2)

        # 검출된 코너를 원본 이미지에 그리기
        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)

        # 파일명 출력
        print("코너 검출 성공:", fname)

        # 검출 결과 화면에 표시
        cv2.imshow("Detected Corners", img)
        cv2.waitKey(500)

    # 코너를 찾지 못한 경우
    else:
        print("코너 검출 실패:", fname)

# 모든 창 닫기
cv2.destroyAllWindows()

# 유효한 코너가 하나도 검출되지 않으면 종료
if len(objpoints) == 0 or len(imgpoints) == 0:
    print("유효한 체크보드 코너를 찾은 이미지가 없습니다.")
    exit()

# -----------------------------
# 2. 카메라 캘리브레이션
# -----------------------------
# 카메라 내부행렬(K), 왜곡계수(dist), 회전벡터(rvecs), 이동벡터(tvecs) 계산
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,   # 실제 3D 좌표
    imgpoints,   # 이미지 2D 좌표
    img_size,    # 이미지 크기
    None,        # 초기 카메라 행렬
    None         # 초기 왜곡 계수
)

# 결과 출력
print("\n==============================")
print("Camera Matrix K:")
print(K)

print("\nDistortion Coefficients:")
print(dist)

# 재투영 오차 계산
total_error = 0

for i in range(len(objpoints)):
    # 3D 점들을 다시 이미지 평면으로 투영
    projected_points, _ = cv2.projectPoints(
        objpoints[i],   # 실제 3D 좌표
        rvecs[i],       # 회전 벡터
        tvecs[i],       # 이동 벡터
        K,              # 카메라 행렬
        dist            # 왜곡 계수
    )

    # 검출한 코너와 재투영한 점 사이의 평균 오차 계산
    error = cv2.norm(imgpoints[i], projected_points, cv2.NORM_L2) / len(projected_points)
    total_error += error

# 전체 평균 오차 계산
mean_error = total_error / len(objpoints)

print("\nMean Reprojection Error:")
print(mean_error)
print("==============================")

# -----------------------------
# 3. 왜곡 보정 시각화
# -----------------------------
# 첫 번째 이미지를 예시로 사용
test_img = cv2.imread(images[0])

# 테스트 이미지를 읽지 못하면 종료
if test_img is None:
    print("왜곡 보정용 테스트 이미지를 읽을 수 없습니다.")
    exit()

# 이미지 높이와 너비 추출
h, w = test_img.shape[:2]

# 왜곡 보정 후 더 적절한 새 카메라 행렬 계산
newK, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))

# 왜곡 보정 수행
undistorted = cv2.undistort(test_img, K, dist, None, newK)

# ROI 정보 추출
x, y, w_roi, h_roi = roi

# 검은 테두리를 제거하기 위해 ROI 영역만 잘라냄
undistorted_cropped = undistorted[y:y + h_roi, x:x + w_roi]

# 결과 이미지 저장
cv2.imwrite("original_image.jpg", test_img)
cv2.imwrite("undistorted_image.jpg", undistorted)
cv2.imwrite("undistorted_cropped.jpg", undistorted_cropped)

# 결과 화면 출력
cv2.imshow("Original Image", test_img)
cv2.imshow("Undistorted Image", undistorted)
cv2.imshow("Undistorted Cropped", undistorted_cropped)

# 키 입력 대기
cv2.waitKey(0)

# 모든 창 닫기
cv2.destroyAllWindows()
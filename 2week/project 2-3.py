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
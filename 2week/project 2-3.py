import cv2
import numpy as np
from pathlib import Path

# 출력 폴더 생성
output_dir = Path("./outputs")
output_dir.mkdir(parents=True, exist_ok=True)

# 좌/우 이미지 불러오기
left_color = cv2.imread("2week/left.png")
right_color = cv2.imread("2week/right.png")

if left_color is None or right_color is None:
    raise FileNotFoundError("좌/우 이미지를 찾지 못했습니다.")


# 카메라 파라미터
f = 700.0
B = 0.12

# ROI 설정
rois = {
    "Painting": (55, 50, 130, 110),
    "Frog": (90, 265, 230, 95),
    "Teddy": (310, 35, 115, 90)
}

# 그레이스케일 변환
left_gray = cv2.cvtColor(left_color, cv2.COLOR_BGR2GRAY)
right_gray = cv2.cvtColor(right_color, cv2.COLOR_BGR2GRAY)

# -----------------------------
# 1. Disparity 계산
# -----------------------------
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

# -----------------------------
# 2. Depth 계산
# Z = fB / d
# -----------------------------
depth_map = np.zeros_like(disparity, dtype=np.float32)
valid_mask = disparity > 0
depth_map[valid_mask] = (f * B) / disparity[valid_mask]

# -----------------------------
# 3. ROI별 평균 disparity / depth 계산
# -----------------------------
results = {}

for name, (x, y, w, h) in rois.items():
    disp_roi = disparity[y:y+h, x:x+w]
    depth_roi = depth_map[y:y+h, x:x+w]

    valid_disp_roi = disp_roi[disp_roi > 0]
    valid_depth_roi = depth_roi[depth_roi > 0]

    mean_disparity = float(np.mean(valid_disp_roi)) if valid_disp_roi.size > 0 else 0.0
    mean_depth = float(np.mean(valid_depth_roi)) if valid_depth_roi.size > 0 else 0.0

    results[name] = {
        "mean_disparity": mean_disparity,
        "mean_depth": mean_depth
    }

# -----------------------------
# 4. 결과 출력
# -----------------------------
print("=== ROI별 평균 Disparity / Depth ===")
for name, value in results.items():
    print(f"{name}")
    print(f"  Mean Disparity: {value['mean_disparity']:.4f}")
    print(f"  Mean Depth    : {value['mean_depth']:.4f}")

closest_roi = max(results, key=lambda k: results[k]["mean_disparity"])
farthest_roi = min(results, key=lambda k: results[k]["mean_disparity"])

print("\n=== 거리 해석 ===")
print(f"가장 가까운 ROI: {closest_roi}")
print(f"가장 먼 ROI   : {farthest_roi}")

# -----------------------------
# 5. disparity 시각화
# 가까울수록 빨강 / 멀수록 파랑
# -----------------------------
disp_tmp = disparity.copy()
disp_tmp[disp_tmp <= 0] = np.nan

if np.all(np.isnan(disp_tmp)):
    raise ValueError("유효한 disparity 값이 없습니다.")

d_min = np.nanpercentile(disp_tmp, 5)
d_max = np.nanpercentile(disp_tmp, 95)

if d_max <= d_min:
    d_max = d_min + 1e-6

disp_scaled = (disp_tmp - d_min) / (d_max - d_min)
disp_scaled = np.clip(disp_scaled, 0, 1)

disp_vis = np.zeros_like(disparity, dtype=np.uint8)
valid_disp = ~np.isnan(disp_tmp)
disp_vis[valid_disp] = (disp_scaled[valid_disp] * 255).astype(np.uint8)

disparity_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

# -----------------------------
# 6. depth 시각화
# 가까울수록 빨강 / 멀수록 파랑
# -----------------------------
depth_vis = np.zeros_like(depth_map, dtype=np.uint8)

if np.any(valid_mask):
    depth_valid = depth_map[valid_mask]

    z_min = np.percentile(depth_valid, 5)
    z_max = np.percentile(depth_valid, 95)

    if z_max <= z_min:
        z_max = z_min + 1e-6

    depth_scaled = (depth_map - z_min) / (z_max - z_min)
    depth_scaled = np.clip(depth_scaled, 0, 1)

    # depth는 클수록 멀기 때문에 반전
    depth_scaled = 1.0 - depth_scaled
    depth_vis[valid_mask] = (depth_scaled[valid_mask] * 255).astype(np.uint8)

depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

# -----------------------------
# 7. Left / Right 이미지에 ROI 표시
# -----------------------------
left_vis = left_color.copy()
right_vis = right_color.copy()

# -----------------------------
# 8. 저장
# -----------------------------
cv2.imwrite(str(output_dir / "original.png"), left_color)
cv2.imwrite(str(output_dir / "disparity_map_color.png"), disparity_color)

# -----------------------------
# 9. 출력
# -----------------------------
cv2.imshow("Original", left_color)
cv2.imshow("Disparity map", disparity_color)

cv2.waitKey(0)
cv2.destroyAllWindows()
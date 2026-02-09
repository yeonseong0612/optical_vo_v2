import cv2
import numpy as np
from lu_vp_detect import VPDetection

# 1. 설정 및 이미지 로드
img_path = '/home/yskim/projects/vo-labs/data/kitti_odometry/datasets/sequences/00/image_2/000000.png'
calib_path = '/home/yskim/projects/vo-labs/data/kitti_odometry/datasets/sequences/00/calib.txt'

img = cv2.imread(img_path)
h, w = img.shape[:2]

# 캘리브레이션 (아까 사용한 KITTI 00번 값)
fx, fy, cx, cy = 718.8, 718.8, 607.1, 185.2

# 2. lu-vp-detect 실행
vpd = VPDetection(focal_length=fx, principal_point=(cx, cy), length_thresh=60)
vpts_3d = vpd.find_vps(img) # [x, y, z] 단위 벡터 리스트 반환

# [핵심] 여러 소실점 중 정면(Z축 방향)과 가장 일치하는 벡터 찾기
# 정면 벡터 [0, 0, 1]과 내적이 가장 큰(각도가 작은) 소실점을 선택합니다.
front_vector = np.array([0, 0, 1])
best_idx = np.argmax([np.abs(np.dot(v, front_vector)) for v in vpts_3d])

best_vpt = vpts_3d[best_idx]
x, y, z = best_vpt

# 3. 각도 및 픽셀 좌표 재계산
# 만약 z가 음수라면 카메라 뒤쪽이므로 반전시킵니다.
if z < 0: x, y, z = -x, -y, -z

yaw = np.degrees(np.arctan2(x, z))
pitch = np.degrees(np.arctan2(-y, np.sqrt(x**2 + z**2)))

# 실제 KITTI 투영 공식 적용
u = x / (z + 1e-6)
v = y / (z + 1e-6)
x_pixel = u * fx + cx
y_pixel = v * fy + cy

print(f"--- lu-vp-detect 정면 보정 결과 ---")
print(f"Yaw: {yaw:.4f}°, Pitch: {pitch:.4f}°")
print(f"픽셀 좌표: ({x_pixel:.1f}, {y_pixel:.1f})")

# 4. 수동 시각화 (AttributeError 해결)
cv2.circle(img, (int(x_pixel), int(y_pixel)), 10, (0, 0, 255), -1)
cv2.putText(img, f"Traditional VP Yaw:{yaw:.2f}", (int(x_pixel)+10, int(y_pixel)-10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
cv2.imwrite('comparison_traditional.png', img)
print("\n>>> [Success] 'comparison_traditional.png' 생성 완료.")
import sys
import os
import cv2
import torch
import numpy as np

# 1. 경로 설정 (프로젝트 루트 및 외부 라이브러리 등록)
root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root, 'ext/neurvps'))
sys.path.append(os.path.join(root, 'ext/neurvps/neurvps'))
sys.path.append(os.path.join(root, 'src'))

from src.deepvp import VanishingPointDetector, get_rotation_angles

# 테스트 경로 설정 (실제 환경에 맞춰 확인 필요)
CONFIG_PATH = 'ext/neurvps/config/tmm17.yaml'
CKPT_PATH = 'checkpoint/checkpoint_latest.pth.tar'
IMG_PATH = '/home/yskim/projects/vo-labs/data/kitti_odometry/datasets/sequences/00/image_2/000000.png'

def test_detector():
    # 2. Detector 초기화
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    detector = VanishingPointDetector(CONFIG_PATH, CKPT_PATH, device=device)
    print(f">>> [Init] 모델 로드 완료 (Device: {device})")

    # 3. 이미지 로드 및 캘리브레이션 설정
    img_bgr = cv2.imread(IMG_PATH)
    if img_bgr is None:
        print(f"Error: 이미지를 찾을 수 없습니다: {IMG_PATH}")
        return
    
    h, w = img_bgr.shape[:2]
    
    # KITTI 00번 시퀀스 Intrinsics (P0 기준)
    fx = 718.856
    fy = 718.856
    cx, cy = 607.193, 185.216
    
    detector.set_focal_length(fx=fx, img_w=w)
    print(f">>> [Setting] Focal Length {fx:.2f} 주입 완료")

    # 4. 전처리 (Numpy BGR -> Tensor RGB)
    # 
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().unsqueeze(0).to(device)
    img_tensor = img_tensor / 255.0  # 0~1 정규화

    # 모델 내부용 전처리 호출 (메서드 명칭 수정: preprocess_tensor)
    input_vp = detector.preprocess_tensor(img_tensor)
    print(f">>> [Inference] '{os.path.basename(IMG_PATH)}' 분석 시작...")
    
    # 5. 소실점 탐지 실행
    best_vpt, max_score = detector.detect(input_vp, initial_vpt=None)
    
    # 6. 결과 해석 (벡터 -> 각도 -> 픽셀 투영)
    yaw, pitch = get_rotation_angles(best_vpt)
    
    # 투영 공식: 
    # $$x_{px} = f_x \cdot \frac{x}{z} + c_x, \quad y_{px} = f_y \cdot \frac{y}{z} + c_y$$
    z_val = best_vpt[2] + 1e-6
    x_px = int((best_vpt[0] / z_val) * fx + cx)
    y_px = int((best_vpt[1] / z_val) * fy + cy)

    print(f"\n" + "="*30)
    print(f"--- 분석 결과 (Score: {max_score:.4f}) ---")
    print(f"Yaw  : {yaw:.4f}°")
    print(f"Pitch: {pitch:.4f}°")
    print(f"Pixel: ({x_px}, {y_px})")
    print("="*30)

    # 7. 시각화 및 저장
    # 소실점이 이미지 범위 안에 있을 때만 그리기
    if 0 <= x_px < w and 0 <= y_px < h:
        cv2.circle(img_bgr, (x_px, y_px), 10, (0, 255, 0), -1)
        cv2.putText(img_bgr, f"VP Score:{max_score:.2f}", (x_px + 15, y_px - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        print("Warning: 소실점이 이미지 평면 밖에 위치합니다.")
    
    output_name = 'test_result_fixed.png'
    cv2.imwrite(output_name, img_bgr)
    print(f"\n>>> [Success] '{output_name}' 저장 완료.")

if __name__ == "__main__":
    test_detector()
import numpy as np
import torch
# ==========================================
# 1. 기하학 및 회전 연산 (R 추정용)
# ==========================================

def estimate_R(v1, v2):
    """두 소실점 벡터(v_t, v_t+1) 사이의 상대 회전 행렬 R을 계산합니다."""
    v1 = v1 / (np.linalg.norm(v1) + 1e-9)
    v2 = v2 / (np.linalg.norm(v2) + 1e-9)

    axis = np.cross(v1, v2)
    sin_theta = np.linalg.norm(axis)
    cos_theta = np.dot(v1, v2)

    if sin_theta < 1e-9:
        return np.eye(3)
    
    axis /= sin_theta
    # Rodrigues' formula
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) + sin_theta * K + (1 - cos_theta) * (K @ K)
    return R

# ==========================================
# 2. 이동량 추정 (M-estimator 기반 t 추정)
# ==========================================

def estimate_t(pts_3d_t, pts_3d_tp1, R_init, c=1.3998, iters=3):
    """Fair Weight를 이용해 이상치를 방어하며 t를 추정합니다."""
    t = np.zeros(3)
    # t 시점의 점들을 t+1 방향으로 먼저 회전
    pts_t_rot = (R_init @ pts_3d_t.T).T

    for i in range(iters):
        diff = pts_3d_tp1 - (pts_t_rot + t)
        error = np.linalg.norm(diff, axis=1)
        # Fair Weight: 1 / (1 + |e|/c)
        weights = 1.0 / (1.0 + (error / c) + 1e-9)

        t_update = np.sum(diff * weights[:, None], axis=0) / (np.sum(weights) + 1e-9)
        t += t_update

        if np.linalg.norm(t_update) < 1e-4:
            break
    return t

# ==========================================
# 3. 삼각측량 (3D 복원)
# ==========================================

def triangulate_from_indices(kpt_l, kpt_r, intrinsics, baseline, max_depth=60.0):
    disp = kpt_l[:, 0] - kpt_r[:, 0]
    
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    valid = disp > 1.0 
    
    z = (fx * baseline) / (disp + 1e-9)
    x = (kpt_l[:, 0] - cx) * z / fx
    y = (kpt_l[:, 1] - cy) * z / fy
    
    pts_3d = np.stack([x, y, z], axis=1)
    
    mask = valid & (z > 1.0) & (z < max_depth)
    
    return pts_3d, mask

# ==========================================
# 4. 유틸리티 및 저장
# ==========================================

def to_np(a):
    """torch.Tensor를 numpy로 안전하게 변환합니다."""
    if a is None or isinstance(a, np.ndarray):
        return a
    return a.detach().cpu().numpy()

def save_trajectory_kitti(pose_list, save_path):
    """계산된 포즈 리스트를 KITTI 표준 포맷(3x4)으로 저장합니다."""
    with open(save_path, "w") as f:
        for T in pose_list:
            # T: 4x4 matrix -> R(3x3) + t(3x1)를 한 줄로 펼침
            line = T[:3, :].reshape(-1)
            f.write(" ".join(f"{v:.6f}" for v in line) + "\n")

def prepare_for_matcher(kpts, descs, device='cuda'):
    # 1. 리스트인 경우 첫 번째 요소만 추출
    if isinstance(kpts, list): kpts = kpts[0]
    if isinstance(descs, list): descs = descs[0]

    # 2. 넘파이인 경우 텐서로 변환
    if isinstance(kpts, np.ndarray):
        kpts = torch.from_numpy(kpts).to(device)
    if isinstance(descs, np.ndarray):
        descs = torch.from_numpy(descs).to(device)

    # 3. 차원 확인 및 확장 (N, 2) -> (1, N, 2)
    # 텐서의 경우 .dim() 혹은 .ndim 모두 사용 가능합니다.
    if kpts.ndim == 2:
        kpts = kpts.unsqueeze(0)
    if descs.ndim == 2:
        descs = descs.unsqueeze(0)
        
    return kpts, descs

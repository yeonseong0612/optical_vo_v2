import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from src.utils import * # triangulate_from_indices, estimate_t, to_np 등 포함
from src.extractor import SuperPointExtractor
from src.matcher import LightGlueMatcher
from src.deepvp import VanishingPointDetector

class VO(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.extractor = SuperPointExtractor(cfg)
        self.matcher = LightGlueMatcher(feature_type=cfg.feature_type)
        self.vp = VanishingPointDetector(cfg.vp_config_path, cfg.vp_ckpt_path)
        self.prev_vp = None
        self.prev_stable_R = np.eye(3) # VP 실패 시 대비한 백업

    def refine_pose(self, pts_3d, pts_2d, R_init, t_init, K, iterations=10):
        import torch.optim as optim
        from scipy.spatial.transform import Rotation as Rot

        device = self.cfg.device
        # [핵심] 모든 입력을 완전히 독립시킵니다. (미분 에러 방지)
        p3d = torch.from_numpy(pts_3d).float().to(device).detach()
        p2d = torch.from_numpy(pts_2d).float().to(device).detach()
        K_ts = torch.from_numpy(K).float().to(device).detach()

        # 최적화 변수 설정
        r_vec_init = Rot.from_matrix(R_init).as_rotvec()
        r_opt = torch.tensor(r_vec_init, dtype=torch.float32, device=device, requires_grad=True)
        
        t_init_fixed = t_init.copy()
        if t_init_fixed[2] < 0: t_init_fixed[2] = -t_init_fixed[2]
        t_opt = torch.tensor(t_init_fixed, dtype=torch.float32, device=device, requires_grad=True)

        # LBFGS는 이런 소규모 최적화에 가장 빠르고 정확합니다.
        optimizer = optim.LBFGS([r_opt, t_opt], lr=0.1, max_iter=iterations)

        def closure():
            optimizer.zero_grad()
            angle = torch.norm(r_opt + 1e-8)
            axis = r_opt / angle
            
            zero = torch.zeros(1, device=device)
            row1 = torch.stack([zero, -axis[2:3], axis[1:2]], dim=1)
            row2 = torch.stack([axis[2:3], zero, -axis[0:1]], dim=1)
            row3 = torch.stack([-axis[1:2], axis[0:1], zero], dim=1)
            K_skew = torch.cat([row1, row2, row3], dim=0)
            
            R_curr = torch.eye(3, device=device) + torch.sin(angle) * K_skew + (1 - torch.cos(angle)) * (K_skew @ K_skew)
            
            p3d_trans = (R_curr @ p3d.T).T + t_opt
            p2d_proj_h = (K_ts @ p3d_trans.T).T
            p2d_proj = p2d_proj_h[:, :2] / (p2d_proj_h[:, 2:3] + 1e-8)
            
            dist = torch.norm(p2d - p2d_proj, dim=1)
            loss = torch.where(dist < 1.0, 0.5 * dist**2, dist - 0.5).mean()
            
            if t_opt[2] < 0: loss += torch.abs(t_opt[2]) * 100
            
            loss.backward()
            return loss

        optimizer.step(closure)
        return Rot.from_rotvec(r_opt.detach().cpu().numpy()).as_matrix(), t_opt.detach().cpu().numpy()
    def forward(self, batch, mode='train'):
        device = self.cfg.device
        B = batch['L_t'].shape[0]
        W = batch['L_t'].shape[-1]

        # 데이터 준비
        L_t, R_t = batch['L_t'].to(device), batch['R_t'].to(device)
        L_tp1, R_tp1 = batch['L_tp1'].to(device), batch['R_tp1'].to(device)
        intrinsics = batch['intrinsics'].to(device)
        baseline = batch['baseline'].to(device)

        # 1. Vanishing Point 처리 (전처리 포함)
        self.vp.set_focal_length(fx=intrinsics[0, 0, 0].item(), img_w=W)
        L_t_vp = self.vp.preprocess_kitti(L_t[0]) 
        L_tp1_vp = self.vp.preprocess_kitti(L_tp1[0])

        vpt_t, score_t = self.vp.detect(L_t_vp, initial_vpt=self.prev_vp)
        vpt_tp1, score_tp1 = self.vp.detect(L_tp1_vp, initial_vpt=vpt_t)
        self.prev_vp = vpt_tp1

        # 2. R_init 결정 (소실점 신뢰도 기반)
        if score_t > 0.7 and score_tp1 > 0.7:
            # R_init = estimate_R(vpt_t, vpt_tp1) # 추후 구현
            R_init = np.eye(3) 
        else:
            R_init = np.eye(3)

        # 3. 특징점 추출 및 매칭 (Temporal + Stereo)
        kpt_l_t, desc_l_t, _ = self.extractor(L_t)
        kpt_r_t, desc_r_t, _ = self.extractor(R_t)
        kpt_l_tp1, desc_l_tp1, _ = self.extractor(L_tp1)
        kpt_r_tp1, desc_r_tp1, _ = self.extractor(R_tp1)

        kpt_l_t, desc_l_t = prepare_for_matcher(kpt_l_t, desc_l_t)
        kpt_r_t, desc_r_t = prepare_for_matcher(kpt_r_t, desc_r_t)
        kpt_l_tp1, desc_l_tp1 = prepare_for_matcher(kpt_l_tp1, desc_l_tp1)
        kpt_r_tp1, desc_r_tp1 = prepare_for_matcher(kpt_r_tp1, desc_r_tp1)
        
        stereo_t = self.matcher({'keypoints': kpt_l_t, 'descriptors': desc_l_t},
                                {'keypoints': kpt_r_t, 'descriptors': desc_r_t})
        stereo_tp1 = self.matcher({'keypoints': kpt_l_tp1, 'descriptors': desc_l_tp1},
                                  {'keypoints': kpt_r_tp1, 'descriptors': desc_r_tp1})

        temp_l = self.matcher({'keypoints': kpt_l_t, 'descriptors': desc_l_t},
                              {'keypoints': kpt_l_tp1, 'descriptors': desc_l_tp1})
        temp_r = self.matcher({'keypoints': kpt_r_t, 'descriptors': desc_r_t},
                              {'keypoints': kpt_r_tp1, 'descriptors': desc_r_tp1})
        results = []

        for b in range(B):
            idx_lt_rt = stereo_t['matches0'][b]
            idx_lt_ltp1 = temp_l['matches0'][b]
            idx_rt_rtp1 = temp_r['matches0'][b]
            idx_ltp1_rtp1 = stereo_tp1['matches0'][b]

            # L_t -> R_t -> R_tp1  vs  L_t -> L_tp1 -> R_tp1
            initial_mask = (idx_lt_rt > -1) & (idx_lt_ltp1 > -1)
            lt_indices = torch.where(initial_mask)[0]
            
            if len(lt_indices) < 10:
                results.append({'R': np.eye(3), 't': np.zeros(3)})
                continue

            rt_indices = idx_lt_rt[lt_indices]
            rtp1_via_rt = idx_rt_rtp1[rt_indices]
            
            ltp1_indices = idx_lt_ltp1[lt_indices]
            rtp1_via_ltp1 = idx_ltp1_rtp1[ltp1_indices]

            cycle_mask = (rtp1_via_rt > -1) & (rtp1_via_rt == rtp1_via_ltp1)
            f_lt, f_rt = lt_indices[cycle_mask], rt_indices[cycle_mask]
            f_ltp1, f_rtp1 = ltp1_indices[cycle_mask], rtp1_via_rt[cycle_mask]

            if hasattr(self.cfg, 'epipolar_threshold'):
                y_err = torch.abs(kpt_l_t[b][f_lt, 1] - kpt_r_t[b][f_rt, 1])
                geo_mask = y_err < self.cfg.epipolar_threshold
                f_lt, f_rt = f_lt[geo_mask], f_rt[geo_mask]
                f_ltp1, f_rtp1 = f_ltp1[geo_mask], f_rtp1[geo_mask]

            p_lt, p_rt = to_np(kpt_l_t[b][f_lt]), to_np(kpt_r_t[b][f_rt])
            p_ltp1, p_rtp1 = to_np(kpt_l_tp1[b][f_ltp1]), to_np(kpt_r_tp1[b][f_rtp1])
            K_np, base_val = to_np(intrinsics[b]), baseline[b].item()
            
            pts_3d_t, m_t = triangulate_from_indices(p_lt, p_rt, K_np, base_val)
            pts_3d_tp1, m_tp1 = triangulate_from_indices(p_ltp1, p_rtp1, K_np, base_val)
            
            mask_final = m_t & m_tp1
            
            R_final, t_final = R_init.copy(), np.array([0, 0, 0.1])
            
            if mask_final.sum() > 10:
                t_init = estimate_t(pts_3d_t[mask_final], pts_3d_tp1[mask_final], R_init)
                z_init = t_init[2]
                try:
                    # BA 실행
                    R_final, t_final = self.refine_pose(
                        pts_3d_t[mask_final], p_ltp1[mask_final], 
                        R_init, t_init, to_np(intrinsics[b])
                    )
                    
                    # --- [BA 결과 상세 리포트] ---
                    from scipy.spatial.transform import Rotation as R_tool
                    
                    # 오일러 각도 변환 (XYZ 순서, Degree 단위) 
                    euler_init = R_tool.from_matrix(R_init).as_euler('xyz', degrees=True)
                    euler_final = R_tool.from_matrix(R_final).as_euler('xyz', degrees=True)
                    
                    print(f"\n" + "="*60)
                    print(f"Frame Index: {b} | Points used: {mask_final.sum()}")
                    
                    print(f"--- [Rotation (Euler deg)] ---")
                    print(f"  Initial (P, R, Y): {euler_init[0]:.3f}, {euler_init[1]:.3f}, {euler_init[2]:.3f}")
                    print(f"  Refined (P, R, Y): {euler_final[0]:.3f}, {euler_final[1]:.3f}, {euler_final[2]:.3f}")
                    print(f"  Rotation Diff    : {np.linalg.norm(euler_final - euler_init):.4f} deg")
                    
                    print(f"--- [Translation (meters)] ---")
                    print(f"  Initial (x, y, z): {t_init[0]:.4f}, {t_init[1]:.4f}, {t_init[2]:.4f}")
                    print(f"  Refined (x, y, z): {t_final[0]:.4f}, {t_final[1]:.4f}, {t_final[2]:.4f}")
                    
                    # 전진량 보정 수치
                    z_drift = t_final[2] - t_init[2]
                    print(f"  Z-Correction     : {z_drift:+.4f} m ({'Forward Fix' if z_drift > 0 else 'Backward Shift'})")
                    print("="*60 + "\n")
                    # -----------------------------
                except Exception as e:
                    print(f">>> [BA Error at frame {b}] {e}")
                    R_final, t_final = R_init, t_init

            results.append({'R': R_final, 't': t_final})

        # [TypeError 방지]
        if len(results) == 0: return np.eye(3), np.zeros(3)
        return results[0]['R'], results[0]['t']
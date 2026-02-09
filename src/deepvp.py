import torch
import numpy as np
import yaml
import cv2

from torchvision.transforms import transforms
class VanishingPointDetector:
    def __init__(self, config_path, ckpt_path, device='cuda'):
        self.device = torch.device(device)
        self.config = self._load_config(config_path)
        self.model, self.m_ref, self.c_ref = self._build_model(ckpt_path)
        self.transform = self._get_transform()
        self.step = self.m_ref.im2col_step
        self.meta = {}

    def _load_config(self, path):
        with open(path, 'r') as f:
            return yaml.safe_load(f)
        
    def _get_transform(self):
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def preprocess_tensor(self, img_tensor):
        """
        (B, 1, H, W) 형태의 그레이스케일 텐서를 
        (B, 3, 512, 512) 형태의 정규화된 RGB 텐서로 변환합니다.
        """
        if img_tensor.shape[1] == 1:
            img_tensor = img_tensor.repeat(1, 3, 1, 1)
        
        img_tensor = torch.nn.functional.interpolate(
            img_tensor, size=(512, 512), mode='bilinear', align_corners=False
        )
        
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(img_tensor.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(img_tensor.device)
        img_tensor = (img_tensor - mean) / std
        
        return img_tensor

    def _build_model(self, ckpt_path):
        from ext.neurvps.neurvps.models.vanishing_net import VanishingNet
        from ext.neurvps.neurvps.models.hourglass_pose import HourglassNet, Bottleneck2D
        import ext.neurvps.neurvps.models.vanishing_net as vnet_module
        import ext.neurvps.neurvps.config as neurvps_cfg
        from ext.neurvps.neurvps.box import Box
        
        if not hasattr(vnet_module.M, 'conic_6x'):
            vnet_module.M.conic_6x = False
            
        # 전역 설정 동기화 및 초기화
        vnet_module.M.update(self.config['model'])
        vnet_module.C.update(neurvps_cfg.C)
        
        # [추가] C 내부에 io가 누락되는 경우를 대비한 방어 코드
        if 'io' not in vnet_module.C:
            vnet_module.C.io = Box()
        
        model = VanishingNet(vnet_module.C)
        
        # ... (이하 Hourglass 조립 및 SD 로드 로직 동일) ...
        def make_head(in_p, out_p): return torch.nn.Conv2d(in_p, 64, kernel_size=1)
        model.backbone = HourglassNet(
            planes=256, block=Bottleneck2D, head=make_head,
            depth=vnet_module.M.depth, num_stacks=vnet_module.M.num_stacks, num_blocks=vnet_module.M.num_blocks
        )
        
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        new_sd = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
        model.load_state_dict(new_sd, strict=False)
        
        return model.to(self.device).eval(), vnet_module.M, vnet_module.C

    def set_focal_length(self, fx, img_w):
        """외부에서 캘리브레이션을 주입할 수 있는 안전한 통로"""
        f_norm = fx / (img_w / 2.0)
        from ext.neurvps.neurvps.box import Box
        if 'io' not in self.c_ref:
            self.c_ref.io = Box()
        self.c_ref.io.focal_length = f_norm
            
    def preprocess_kitti(self, img_input, target_size=(512, 512)):
        if torch.is_tensor(img_input):
            # GPU에 있다면 CPU로 복사 후 변환
            img_np = img_input.detach().cpu().numpy()
            
            # (C, H, W) -> (H, W, C)로 변경
            if img_np.ndim == 3:
                img_np = np.transpose(img_np, (1, 2, 0))
            # (H, W) 그레이스케일일 경우 (H, W, 1)로 변경
            elif img_np.ndim == 2:
                img_np = img_np[:, :, np.newaxis]
                
            # OpenCV 처리를 위해 BGR/RGB 3채널 복사 (그레이스케일 대응)
            if img_np.shape[2] == 1:
                img_np = np.repeat(img_np, 3, axis=2)
            
            img_bgr = img_np
        else:
            img_bgr = img_input
        h, w = img_bgr.shape[:2]
        
        # 1. ROI Crop: KITTI 보닛(하단 20%) 및 하늘(상단 10%) 제거
        top, bottom = int(h * 0.1), int(h * 0.8)
        roi_img = img_bgr[top:bottom, :]
        rh, rw = roi_img.shape[:2]

        # 2. Letterbox Resize: 종횡비 유지
        # 
        scale = min(target_size[0] / rh, target_size[1] / rw)
        nh, nw = int(rh * scale), int(rw * scale)
        resized = cv2.resize(roi_img, (nw, nh), interpolation=cv2.INTER_LINEAR)

        # 3. Padding (검은색 배경 중앙 배치)
        canvas = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
        dx, dy = (target_size[1] - nw) // 2, (target_size[0] - nh) // 2
        canvas[dy:dy+nh, dx:dx+nw] = resized

        # 4. 역변환을 위한 정보 저장
        self.meta = {
            'scale': scale,
            'offset': (dx, dy),
            'crop_top': top,
            'orig_size': (h, w),
            'roi_size': (rh, rw)
        }

        # 5. Tensor 변환 및 정규화
        img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().to(self.device)
        img_tensor = img_tensor / 255.0
        
        # ImageNet 정규화 적용
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)
        img_tensor = (img_tensor - mean) / std
        
        return img_tensor.unsqueeze(0)

    @torch.no_grad()
    def detect(self, img_tensor, initial_vpt=None):
        """Coarse-to-Fine 탐색 수행"""
        best_vpt = initial_vpt if initial_vpt is not None else np.array([0.0, 0.1, 1.0])
        best_vpt /= (np.linalg.norm(best_vpt) + 1e-6)

        # 20도부터 단계별로 정밀하게 좁혀나감
        search_stages = [np.radians(20)] + sorted([np.radians(a) for a in self.m_ref.multires], reverse=True)

        for alpha in search_stages:
            candidates = sample_sphere(best_vpt, alpha, 64) #
            all_scores = []

            for i in range(0, len(candidates), self.step):
                batch_vpts = candidates[i : i + self.step]
                actual_len = len(batch_vpts)
                
                if actual_len < self.step: # Padding
                    batch_vpts = np.vstack([batch_vpts, np.tile([0, 0, 1], (self.step - actual_len, 1))])

                input_dict = {"image": img_tensor, "vpts": batch_vpts, "test": True}
                outputs = self.model(input_dict)
                
                preds = outputs['preds']['ys'] if isinstance(outputs, dict) else outputs
                batch_scores = (preds[0, :actual_len, 3] if preds.ndim == 3 else preds[:actual_len, 3]).cpu().numpy()
                all_scores.append(batch_scores)

            scores_concat = np.concatenate(all_scores)
            best_vpt = candidates[np.argmax(scores_concat)]
            max_score = np.max(scores_concat)

        return best_vpt, max_score

def orth(v):
    x, y, z = v
    o = np.array([0.0, -z, y] if abs(x) < abs(y) else [-z, 0.0, x])
    o /= np.linalg.norm(o)
    return o

def sample_sphere(v, alpha, num_pts):
    v1, v2 = orth(v), np.cross(v, orth(v))
    v, v1, v2 = v[:, None], v1[:, None], v2[:, None]
    indices = np.linspace(1, num_pts, num_pts)
    phi = np.arccos(1 + (np.cos(alpha) - 1) * indices / num_pts)
    theta = np.pi * (1 + 5 ** 0.5) * indices
    r = np.sin(phi)
    return (v * np.cos(phi) + r * (v1 * np.cos(theta) + v2 * np.sin(theta))).T

def get_rotation_angles(vpt):
    vpt = vpt / (np.linalg.norm(vpt) + 1e-6)
    x, y, z = vpt
    yaw = np.arctan2(x, z) * 180 / np.pi
    pitch = np.degrees(np.arctan2(y, np.sqrt(x**2 + z**2)))
    return yaw, pitch
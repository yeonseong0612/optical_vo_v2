import torch.utils.data as data
import cv2
import numpy as np
import os
import torch


class DataFactory(data.Dataset):
    def __init__(self, cfg, mode='train'):
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.datalist = []
        self.posesdict = {}
        self.calibdict = {}
        
        # 설정값 로드
        txt_file = cfg.traintxt if mode == 'train' else cfg.valtxt
        sequencelist = cfg.trainsequencelist if mode == 'train' else cfg.valsequencelist

        # 1. 데이터 리스트 로드 (txt 파일 전체 읽기)
        txt_path = os.path.join(cfg.proj_home, 'gendata', txt_file)
        with open(txt_path) as f:
            self.datalist = [line.strip() for line in f.readlines()]

        # 2. 각 시퀀스별 Pose 및 Calibration 정보 로드
        for seq in sequencelist:
            pose_path = os.path.join(cfg.odometry_home, cfg.poses_subdir, f"{seq}.txt")
            with open(pose_path) as p:
                self.posesdict[seq] = [
                    np.array(line.split(), dtype=np.float32).reshape(3, 4) 
                    for line in p.readlines()
                ]
            
            calib_path = os.path.join(cfg.odometry_home, cfg.calib_subdir, seq, "calib.txt")
            with open(calib_path) as f:
                lines = f.readlines()
                P2 = np.array(lines[2].strip().split()[1:], dtype=np.float32).reshape(3, 4)
                P3 = np.array(lines[3].strip().split()[1:], dtype=np.float32).reshape(3, 4)
                
                fx, fy = P2[0, 0], P2[1, 1]
                cx, cy = P2[0, 2], P2[1, 2]
                
                baseline = abs(P3[0, 3] - P2[0, 3]) / fx
                
                self.calibdict[seq] = {
                    'intrinsics': np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32),
                    'baseline': np.array(baseline, dtype=np.float32)
                }
        
    def __len__(self):
        return len(self.datalist)
    

    def __getitem__(self, index):
        seq, imgnum = self.datalist[index].split(' ')
        imgnum = imgnum.strip()
        imgnum = int(imgnum)
        base = os.path.join(self.cfg.odometry_home, self.cfg.color_subdir, seq)
        paths = [
            os.path.join(base, 'image_2', f"{imgnum:06d}.png"),      # L_t
            os.path.join(base, 'image_3', f"{imgnum:06d}.png"),      # R_t
            os.path.join(base, 'image_2', f"{(imgnum+1):06d}.png"),  # L_tp1
            os.path.join(base, 'image_3', f"{(imgnum+1):06d}.png")   # R_tp1
        ]

        images = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in paths]

        for i, img in enumerate(images):
            if img is None:
                raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {paths[i]}")

        H, W= images[0].shape
        crop_top = H % 32

        processed_imgs = [img[crop_top:, :1216] for img in images]

        return {
            "L_t": processed_imgs[0],
            "R_t": processed_imgs[1],
            "L_tp1": processed_imgs[2],
            "R_tp1": processed_imgs[3],
            "pose_t": self.posesdict[seq][imgnum],
            "pose_tp1": self.posesdict[seq][imgnum+1],
            "intrinsics": self.calibdict[seq]['intrinsics'],
            "baseline": self.calibdict[seq]['baseline'],
            "seq": seq,
            "imgnum": imgnum
        }


def vo_collate_fn(batch):
    def prepare_img(img_list):
        t = torch.from_numpy(np.stack(img_list, axis=0))
        return t.unsqueeze(1).float() / 255.0

    l_t = prepare_img([item['L_t'] for item in batch])
    r_t = prepare_img([item['R_t'] for item in batch])
    l_tp1 = prepare_img([item['L_tp1'] for item in batch])
    r_tp1 = prepare_img([item['R_tp1'] for item in batch])

    poses_t = torch.from_numpy(np.stack([item['pose_t'] for item in batch], axis=0))
    poses_tp1 = torch.from_numpy(np.stack([item['pose_tp1'] for item in batch], axis=0))
    intrinsics = torch.from_numpy(np.stack([item['intrinsics'] for item in batch], axis=0))
    baselines = torch.from_numpy(np.stack([item['baseline'] for item in batch], axis=0))

    seqs = [item['seq'] for item in batch]
    imgnums = [item['imgnum'] for item in batch]

    return {
        "L_t": l_t,
        "R_t": r_t,
        "L_tp1": l_tp1,
        "R_tp1": r_tp1,
        "pose_t": poses_t,
        "pose_tp1": poses_tp1,
        "intrinsics": intrinsics,
        "baseline": baselines,
        "seq": seqs,
        "imgnum": imgnums
    }
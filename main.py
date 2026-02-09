import sys
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# 경로 설정
root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root, 'ext/neurvps'))
sys.path.append(os.path.join(root, 'ext/neurvps/neurvps'))
sys.path.append(os.path.join(root, 'src'))

from CFG.cfg import cfg
from src.kitti_dataloader import DataFactory, vo_collate_fn
from src.runner import VO

def main():
    # 'cup' 오타 수정 -> 'cpu'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VO(cfg).to(device)
    model.eval()

    dataset = DataFactory(cfg, mode='train')
    dataset.datalist = [line for line in dataset.datalist if line.startswith('00 ')]
    print(f"필터링 완료: {len(dataset)} frames")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=vo_collate_fn)

    output_path = "results/seq_00_pred.txt"
    os.makedirs("results", exist_ok=True)

    cur_pose = np.eye(4)
    
    with open(output_path, "w") as f:
        first_line = cur_pose[:3, :].flatten()
        f.write(" ".join(f"{v:.6f}" for v in first_line) + "\n")
        f.flush() 

        print(f">>> [Start] KITTI Sequence 00 실시간 기록 시작 (Total: {len(dataset)} frames)")

        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader)):
                R_rel, t_rel = model(batch)

                T_rel = np.eye(4)
                if R_rel is not None and t_rel is not None:
                    T_rel[:3, :3] = R_rel
                    T_rel[:3, 3] = t_rel

                cur_pose = cur_pose @ T_rel

                line = cur_pose[:3, :].flatten()
                f.write(" ".join(f"{v:.6f}" for v in line) + "\n")
                
                f.flush() 
    
    print(f"\n>>> [Finish] 모든 궤적 실시간 저장 완료: {output_path}")

if __name__ == "__main__":
    main()
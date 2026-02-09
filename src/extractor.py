import torch
from ext.superpoint.superpoint import SuperPoint

class SuperPointExtractor:
    def __init__(self, cfg):
        self.device = cfg.device
        self.weights = cfg.weights_path

        self.model = SuperPoint(max_num_keypoints=cfg.max_keypoints).to(self.device)
        state = torch.load(cfg.weights_path, map_location=self.device)
        self.model.load_state_dict(state, strict=False)
        self.model.eval()

    @torch.no_grad()
    def __call__(self, image_tensor):
        t = image_tensor.to(self.device) 
        
        out = self.model({"image": t})
        
        kpts = out["keypoints"][0].cpu().numpy()          # (N,2)  x,y
        desc = out["descriptors"][0].cpu().numpy()        # (N,256)
        scores = out["keypoint_scores"][0].cpu().numpy()  # (N,)

        return kpts, desc, scores
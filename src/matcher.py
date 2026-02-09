import torch
import torch.nn as nn
from ext.lightglue.lightglue import LightGlue

class LightGlueMatcher(nn.Module):
    def __init__(self, device='cuda', feature_type='superpoint'):
        super().__init__()
        self.device = device
        self.model = LightGlue(
            features=feature_type,
            width_confidence=-1,  
            depth_confidence=-1  
        ).to(device).eval()
        
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, data0, data1):
        
        B = data0['keypoints'].shape[0]

        input_dict = {
            "image0": {
                "keypoints": data0["keypoints"],
                "descriptors": data0["descriptors"],
                "image_size": torch.tensor([[1242, 375]], device=self.device).float().repeat(B, 1)
            },
            "image1": {
                "keypoints": data1["keypoints"],
                "descriptors": data1["descriptors"],
                "image_size": torch.tensor([[1242, 375]], device=self.device).float().repeat(B, 1)
            }
        }

        return self.model(input_dict)
    
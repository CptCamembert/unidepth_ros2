import torch
import cv2
import numpy as np
from unidepth.models import UniDepthV2
from unidepth.utils.camera import Pinhole


class DepthProcessor:
    def __init__(self, model_name="unidepth-v2-vitl14"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UniDepthV2.from_pretrained(f"lpiccinelli/{model_name}")
        self.model = self.model.to(self.device)
        self.model.eval()
        
    def setup_camera(self, camera_info):
        """Set up camera intrinsics from ROS CameraInfo message"""
        K = torch.tensor([
            [camera_info.k[0], 0, camera_info.k[2]],
            [0, camera_info.k[4], camera_info.k[5]],
            [0, 0, 1]
        ], dtype=torch.float32)
        return Pinhole(K=K)
    
    def process_depth(self, rgb_tensor, camera):
        """Process depth estimation from RGB tensor"""
        with torch.no_grad():
            depth_predictions = self.model.infer(rgb_tensor, camera)
            depth = depth_predictions["depth"].squeeze().cpu().numpy()
            points = depth_predictions["points"].squeeze().cpu().numpy().transpose(1, 2, 0)
            return depth, points
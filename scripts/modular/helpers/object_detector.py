import numpy as np
import cv2
from ultralytics import YOLO


class DetectedObject:
    def __init__(self, class_name, confidence, x, y, z):
        self.class_name = class_name
        self.confidence = confidence
        self.x = x  # forward
        self.y = y  # left  
        self.z = z  # up


class ObjectDetector:
    def __init__(self, model_path="yolo11n-seg.pt"):
        self.model = YOLO(model_path)
        
    def detect_objects(self, image_rgb, depth, camera_intrinsics):
        """Detect objects and calculate their 3D positions"""
        detection_results = self.model(image_rgb)
        detected_objects = []
        detection_data = []
        
        if hasattr(detection_results[0], 'masks') and detection_results[0].masks is not None:
            masks = detection_results[0].masks
            boxes = detection_results[0].boxes
            
            for mask_tensor, box in zip(masks.data, boxes):
                class_name = self.model.names[int(box.cls)]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                box_center_x = (x1 + x2) // 2
                box_center_y = (y1 + y2) // 2
                
                mask = mask_tensor.cpu().numpy()
                mask_points = np.where(mask > 0.5)
                
                if len(mask_points[0]) > 0:
                    # Find optimal depth sampling point using distance transform
                    binary_mask = np.zeros((depth.shape[0], depth.shape[1]), dtype=np.uint8)
                    binary_mask[mask_points] = 255
                    
                    dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
                    max_dist_loc = np.where(dist_transform == np.max(dist_transform))
                    
                    if len(max_dist_loc[0]) > 0:
                        sample_y = max_dist_loc[0][0]
                        sample_x = max_dist_loc[1][0]
                    else:
                        sample_x = box_center_x
                        sample_y = box_center_y
                    
                    if 0 <= sample_y < depth.shape[0] and 0 <= sample_x < depth.shape[1]:
                        depth_at_point = depth[sample_y, sample_x]
                        
                        # Convert to camera coordinate system
                        x_meters = float(depth_at_point)
                        y_meters = -(box_center_x - camera_intrinsics.k[2]) * depth_at_point / camera_intrinsics.k[0]
                        z_meters = -(box_center_y - camera_intrinsics.k[5]) * depth_at_point / camera_intrinsics.k[4]
                        
                        conf = box.conf[0]
                        
                        detected_objects.append(DetectedObject(
                            class_name=class_name,
                            confidence=conf,
                            x=x_meters,
                            y=y_meters,
                            z=z_meters
                        ))
                        
                        detection_data.append({
                            'class_name': class_name,
                            'confidence': conf,
                            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                            'mask': mask,
                            'class_id': int(box.cls),
                            'depth': depth_at_point,
                            'x': x_meters, 'y': y_meters, 'z': z_meters
                        })
        
        return detected_objects, detection_data
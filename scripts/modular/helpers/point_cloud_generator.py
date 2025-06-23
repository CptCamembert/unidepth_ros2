import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
import config


class PointCloudGenerator:
    def create_point_cloud_msg(self, points, rgb_image, header):
        """Create a PointCloud2 message from 3D points with RGB information, downsampled by configurable factor"""
        height, width, _ = points.shape
        
        # Downsample: take every Nth pixel in both dimensions based on config
        downsample_factor = config.POINT_CLOUD_DOWNSAMPLE_FACTOR
        points_downsampled = points[::downsample_factor, ::downsample_factor, :]
        rgb_downsampled = rgb_image[::downsample_factor, ::downsample_factor, :]
        
        new_height, new_width, _ = points_downsampled.shape
        
        cloud_msg = PointCloud2()
        cloud_msg.header = header
        cloud_msg.height = new_height
        cloud_msg.width = new_width
        cloud_msg.is_dense = False
        cloud_msg.is_bigendian = False
        
        cloud_msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
        ]
        
        cloud_msg.point_step = 16
        cloud_msg.row_step = cloud_msg.point_step * new_width
        
        points_flat = points_downsampled.reshape(-1, 3).astype(np.float32)
        rgb_flat = rgb_downsampled.reshape(-1, 3).astype(np.uint8)
        
        rgb_packed = np.zeros(rgb_flat.shape[0], dtype=np.uint32)
        for i in range(rgb_flat.shape[0]):
            r, g, b = rgb_flat[i]
            rgb_packed[i] = (r << 16) | (g << 8) | b
        
        cloud_data = np.zeros(points_flat.shape[0], dtype=[
            ('x', np.float32),
            ('y', np.float32), 
            ('z', np.float32),
            ('rgb', np.uint32)
        ])
        
        cloud_data['x'] = points_flat[:, 0]
        cloud_data['y'] = points_flat[:, 1] 
        cloud_data['z'] = points_flat[:, 2]
        cloud_data['rgb'] = rgb_packed
        
        cloud_msg.data = cloud_data.tobytes()
        return cloud_msg
    
    def create_detection_array_msg(self, detected_objects, header):
        """Create Detection3DArray message from detected objects"""
        detection_array = Detection3DArray()
        detection_array.header = header
        
        for obj in detected_objects:
            detection = Detection3D()
            detection.header = header
            
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = obj.class_name
            hypothesis.hypothesis.score = float(obj.confidence)
            
            hypothesis.pose.pose.position.x = float(obj.x)
            hypothesis.pose.pose.position.y = float(obj.y)
            hypothesis.pose.pose.position.z = float(obj.z)
            
            hypothesis.pose.pose.orientation.w = 1.0
            hypothesis.pose.pose.orientation.x = 0.0
            hypothesis.pose.pose.orientation.y = 0.0
            hypothesis.pose.pose.orientation.z = 0.0
            
            detection.results.append(hypothesis)
            detection_array.detections.append(detection)
        
        return detection_array
    
    def create_landmark_point_cloud_msg(self, landmarks, colors, header, frame_id="map"):
        """
        Create a PointCloud2 message from landmarks with their associated colors
        
        Args:
            landmarks: List of 3D points [[x, y, z], ...]
            colors: List of RGB colors [[r, g, b], ...] (0-255 range)
            header: ROS message header
            frame_id: Frame ID for the point cloud (default: "map")
        
        Returns:
            PointCloud2 message
        """
        if len(landmarks) != len(colors):
            raise ValueError(f"Number of landmarks ({len(landmarks)}) must match number of colors ({len(colors)})")
        
        cloud_msg = PointCloud2()
        cloud_msg.header = header
        cloud_msg.header.frame_id = frame_id
        cloud_msg.height = 1
        cloud_msg.width = len(landmarks)
        cloud_msg.is_dense = True
        cloud_msg.is_bigendian = False
        
        cloud_msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
        ]
        
        cloud_msg.point_step = 16
        cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width
        
        # Convert landmarks and colors to numpy arrays
        landmarks_array = np.array(landmarks, dtype=np.float32)
        colors_array = np.array(colors, dtype=np.uint8)
        
        # Pack RGB colors into uint32 format
        rgb_packed = np.zeros(len(landmarks), dtype=np.uint32)
        for i in range(len(landmarks)):
            r, g, b = colors_array[i]
            rgb_packed[i] = (r << 16) | (g << 8) | b
        
        # Create structured array for point cloud data
        cloud_data = np.zeros(len(landmarks), dtype=[
            ('x', np.float32),
            ('y', np.float32), 
            ('z', np.float32),
            ('rgb', np.uint32)
        ])
        
        cloud_data['x'] = landmarks_array[:, 0]
        cloud_data['y'] = landmarks_array[:, 1] 
        cloud_data['z'] = landmarks_array[:, 2]
        cloud_data['rgb'] = rgb_packed
        
        cloud_msg.data = cloud_data.tobytes()
        return cloud_msg
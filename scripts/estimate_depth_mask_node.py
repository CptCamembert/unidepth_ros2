import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, CompressedImage, PointCloud2, PointField
from geometry_msgs.msg import Point, Pose, PoseWithCovariance, Vector3
from std_msgs.msg import Header, String, Float32
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
from cv_bridge import CvBridge
import numpy as np
import cv2
import torch
import time
import matplotlib.pyplot as plt
from ultralytics import YOLO
import struct

# Import UniDepth
from unidepth.models import UniDepthV2  # Using V2 for better performance

def colorize_depth(depth, min_depth=-1.0, max_depth=-1.0):
    """Convert depth map to color visualization with fixed range of 0-50 meters"""
    # Clip values to our desired range
    if max_depth == -1.0:
        max_depth = np.max(depth)  # Use max depth from the data if not provided
    if min_depth == -1.0:
        min_depth = np.min(depth)
    depth_clipped = np.clip(depth, min_depth, max_depth)
    
    normalized_depth = (depth_clipped - min_depth) / (max_depth - min_depth)
    
    # Convert to colormap (matplotlib returns RGB)
    colorized = plt.cm.inferno(1- normalized_depth)
    # Convert RGB to BGR for OpenCV and return only the RGB channels (not alpha) as uint8
    rgb_image = (colorized[:, :, :3] * 255).astype(np.uint8)
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    return bgr_image

def sample_depth(depth, x, y):
    """Sample depth at a specific pixel location"""
    if 0 <= y < depth.shape[0] and 0 <= x < depth.shape[1]:
        return depth[y, x]
    else:
        return None

# Custom message structure for detected objects
class DetectedObject:
    def __init__(self, class_name, confidence, x, y, z):
        self.class_name = class_name
        self.confidence = confidence
        self.x = x  # forward
        self.y = y  # left  
        self.z = z  # up

class DepthEstimationNode(Node):
    def __init__(self):
        super().__init__('depth_estimation_node')
        
        # Initialize CV Bridge
        self.bridge = CvBridge()
        
        # Camera intrinsics (will be updated from camera_info)
        self.camera_intrinsics = None
        self.camera = None
        
        # Subscribers
        self.image_subscription = self.create_subscription(
            CompressedImage,
            '/camera/camera/color/image_raw/compressed',
            self.image_callback,
            1)
        
        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',
            self.camera_info_callback,
            1)
        
        # Publishers
        self.depth_image_publisher = self.create_publisher(Image, '/depth/image_raw', 1)
        self.annotated_image_publisher = self.create_publisher(Image, '/depth/annotated_image', 1)
        self.point_cloud_publisher = self.create_publisher(PointCloud2, '/depth/point_cloud', 1)
        self.detected_objects_publisher = self.create_publisher(Detection3DArray, '/detected_objects', 1)
        
        self.get_logger().info("Loading UniDepth model...")
        # Select the model - you can change to any other model from the Model Zoo
        depth_model_name = "unidepth-v2-vitl14"  # or use vits14 for faster performance
        
        # Check if CUDA is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {self.device}")
        
        # Load the model
        self.depth_model = UniDepthV2.from_pretrained(f"lpiccinelli/{depth_model_name}")
        self.depth_model = self.depth_model.to(self.device)
        self.depth_model.eval()  # Set to evaluation mode

        self.get_logger().info("Loading Yolo segmentation model...")
        # Load Yolo model for object segmentation (using a segment model instead of detection)
        self.yolo_model = YOLO("yolo11n-seg.pt")  # Load the YOLOv8 segmentation model
        
        # Initialize frame counter
        self.frame_count = 0
        
        self.get_logger().info("Depth estimation node initialized. Waiting for camera data...")
    
    def camera_info_callback(self, msg):
        """Callback for camera info to get intrinsics"""
        if self.camera_intrinsics is None:
            self.camera_intrinsics = msg
            
            # Set up intrinsics matrix for UniDepth
            K = torch.tensor([
                [msg.k[0], 0, msg.k[2]],  # fx, 0, cx
                [0, msg.k[4], msg.k[5]],  # 0, fy, cy
                [0, 0, 1]
            ], dtype=torch.float32)
            
            from unidepth.utils.camera import Pinhole
            self.camera = Pinhole(K=K)
            
            self.get_logger().info("Camera intrinsics received and initialized")
    
    def create_point_cloud_msg(self, points, rgb_image, header):
        """Create a PointCloud2 message from 3D points with RGB information, downsampled by 4"""
        # Points should be in shape (H, W, 3) where the last dimension is (x, y, z)
        height, width, _ = points.shape
        
        # Downsample: take every 2nd pixel in both dimensions (bottom-left of 2x2 grid)
        points_downsampled = points[1::2, ::2, :]  # Start from row 1, column 0, step by 2
        rgb_downsampled = rgb_image[1::2, ::2, :]
        
        # New dimensions after downsampling
        new_height, new_width, _ = points_downsampled.shape
        
        # Create PointCloud2 message
        cloud_msg = PointCloud2()
        cloud_msg.header = header
        cloud_msg.height = new_height
        cloud_msg.width = new_width
        cloud_msg.is_dense = False
        cloud_msg.is_bigendian = False
        
        # Define point fields (x, y, z coordinates + RGB)
        cloud_msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
        ]
        
        cloud_msg.point_step = 16  # 3 * 4 bytes (float32) + 4 bytes (uint32)
        cloud_msg.row_step = cloud_msg.point_step * new_width
        
        # Prepare data array
        points_flat = points_downsampled.reshape(-1, 3).astype(np.float32)
        rgb_flat = rgb_downsampled.reshape(-1, 3).astype(np.uint8)
        
        # Pack RGB values into uint32
        rgb_packed = np.zeros(rgb_flat.shape[0], dtype=np.uint32)
        for i in range(rgb_flat.shape[0]):
            r, g, b = rgb_flat[i]
            rgb_packed[i] = (r << 16) | (g << 8) | b
        
        # Combine points and RGB data
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
    
    def publish_detected_objects(self, detected_objects, header):
        """Publish detected objects as Detection3DArray message"""
        detection_array = Detection3DArray()
        detection_array.header = header
        
        for obj in detected_objects:
            detection = Detection3D()
            detection.header = header
            
            # Create object hypothesis with pose
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = obj.class_name
            hypothesis.hypothesis.score = float(obj.confidence)
            
            # Set 3D pose (position in camera frame)
            hypothesis.pose.pose.position.x = float(obj.x)  # forward
            hypothesis.pose.pose.position.y = float(obj.y)  # left
            hypothesis.pose.pose.position.z = float(obj.z)  # up
            
            # Set orientation to identity (no rotation information available)
            hypothesis.pose.pose.orientation.w = 1.0
            hypothesis.pose.pose.orientation.x = 0.0
            hypothesis.pose.pose.orientation.y = 0.0
            hypothesis.pose.pose.orientation.z = 0.0
            
            detection.results.append(hypothesis)
            detection_array.detections.append(detection)
        
        self.detected_objects_publisher.publish(detection_array)
    
    def image_callback(self, msg):
        """Callback for processing incoming images"""
        if self.camera is None:
            self.get_logger().warn("Camera intrinsics not yet received, skipping frame")
            return
        
        try:
            # Convert compressed ROS image to OpenCV format
            np_arr = np.frombuffer(msg.data, np.uint8)
            color_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if color_image is None:
                self.get_logger().error("Failed to decode compressed image")
                return
            
            self.frame_count += 1
            
            # Convert to RGB for the model
            color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            
            # Convert to torch tensor format
            rgb_tensor = torch.from_numpy(color_image_rgb).permute(2, 0, 1).float()  # C, H, W
            rgb_tensor = rgb_tensor.to(self.device)
            
            with torch.no_grad():
                # Perform object detection with segmentation
                detection_results = self.yolo_model(color_image_rgb)
                
                # Create a copy of the color image for visualization
                color_display = color_image.copy()
                
                # Pass camera intrinsics to get better results
                depth_predictions = self.depth_model.infer(rgb_tensor, self.camera)
                
                # Get the depth prediction
                depth = depth_predictions["depth"].squeeze().cpu().numpy()
                
                # Colored depth visualization
                depth_colored = colorize_depth(depth)
                
                # Ensure both images have the same dimensions and type
                if color_image.shape[:2] != depth_colored.shape[:2]:
                    depth_colored = cv2.resize(depth_colored, (color_image.shape[1], color_image.shape[0]))
                
                # Create a copy of the depth colored image for annotation
                depth_display = depth_colored.copy()
                
                # Get the center depth
                center_y, center_x = depth.shape[0] // 2, depth.shape[1] // 2
                center_depth = depth[center_y, center_x]
                center_depth_meters = float(center_depth)
                
                # Draw center markers and text
                cv2.drawMarker(depth_display, (center_x, center_y), (255, 255, 255), 
                              markerType=cv2.MARKER_CROSS, markerSize=20, thickness=1)
                cv2.drawMarker(color_display, (center_x, center_y), (0, 255, 0), 
                              markerType=cv2.MARKER_CROSS, markerSize=20, thickness=1)
                
                # Display the center depth value
                depth_text = f"Center Depth: {center_depth_meters:.2f}m"
                cv2.putText(depth_display, depth_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.8, (255, 255, 255), 2, 1)
                
                # List to store detected objects
                detected_objects = []
                
                # Process each detection with mask
                if hasattr(detection_results[0], 'masks') and detection_results[0].masks is not None:
                    masks = detection_results[0].masks
                    boxes = detection_results[0].boxes
                    
                    for i, (mask_tensor, box) in enumerate(zip(masks.data, boxes)):
                        # Get class name
                        class_name = self.yolo_model.names[int(box.cls)]
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        box_center_x = (x1 + x2) // 2
                        box_center_y = (y1 + y2) // 2
                        
                        # Convert mask tensor to numpy array
                        mask = mask_tensor.cpu().numpy()
                        
                        # Find points inside the mask to sample depth
                        mask_points = np.where(mask > 0.5)
                        if len(mask_points[0]) > 0:
                            # First, create a binary mask image
                            binary_mask = np.zeros((depth.shape[0], depth.shape[1]), dtype=np.uint8)
                            binary_mask[mask_points] = 255
                            
                            # Calculate distance transform - each pixel's value is its distance to nearest zero pixel
                            dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
                            
                            # Find the maximum distance and its location
                            # This is the point furthest from any edge of the mask
                            max_dist_val = np.max(dist_transform)
                            max_dist_loc = np.where(dist_transform == max_dist_val)
                            
                            if len(max_dist_loc[0]) > 0:
                                # Take the first point with maximum distance
                                sample_y = max_dist_loc[0][0]
                                sample_x = max_dist_loc[1][0]
                            else:
                                # Fallback box_center_x
                                sample_x = box_center_x
                                sample_y = box_center_y
                            
                            # Get the depth at the sampled point inside the mask
                            if 0 <= sample_y < depth.shape[0] and 0 <= sample_x < depth.shape[1]:
                                depth_at_mask_point = depth[sample_y, sample_x]                              
                                
                                # Convert to camera coordinate system: X=forward(depth), Y=left, Z=up
                                x_meters = float(depth_at_mask_point)  # forward (depth)
                                y_meters = -(box_center_x - self.camera_intrinsics.k[2]) * depth_at_mask_point / self.camera_intrinsics.k[0]  # left
                                z_meters = -(box_center_y - self.camera_intrinsics.k[5]) * depth_at_mask_point / self.camera_intrinsics.k[4]  # up
                                
                                # Get confidence
                                conf = box.conf[0]
                                
                                # Add to detected objects list
                                detected_objects.append(DetectedObject(
                                    class_name=class_name,
                                    confidence=conf,
                                    x=x_meters,
                                    y=y_meters,
                                    z=z_meters
                                ))
                                
                                # Format text with multiple lines
                                text = f"{class_name} ({conf:.2f})\n{depth_at_mask_point:.2f}m\n({x_meters:.2f}, {y_meters:.2f}, {z_meters:.2f})"
                                
                                # Draw a marker at the sampled point
                                cv2.drawMarker(color_display, (box_center_x, box_center_y), (255, 255, 255), 
                                              markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
                                # Draw text with multiple lines
                                y_offset = box_center_y - 30
                                for line in text.split('\n'):
                                    cv2.putText(color_display, line, 
                                                (box_center_x + 5, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                                                0.5, (255, 255, 255), 1, cv2.LINE_AA)
                                    y_offset += 15  # Move down for next line
                        
                        # Draw filled masks with class-based colors
                        class_id = int(box.cls)
                        
                        # Use cv2 colormap to generate colors based on class_id
                        # Map class_id from 0-500 to 0-255 (valid colormap range)
                        color_value = int((class_id*23 % 80) * 255 / 80)
                        
                        # Apply gist_rainbow colormap to get RGB color
                        colormap = cv2.COLORMAP_RAINBOW  # Similar to gist_rainbow
                        color_array = np.zeros((1, 1, 1), dtype=np.uint8)
                        color_array[0, 0, 0] = color_value
                        
                        # Apply the colormap and get the color
                        colored_array = cv2.applyColorMap(color_array, colormap)
                        mask_color = colored_array[0, 0].tolist()  # This is in BGR format
                        
                        # Create a colored mask image
                        colored_mask = np.zeros_like(color_display)
                        
                        # Apply the mask with the specified color
                        for c in range(3):
                            colored_mask[:, :, c] = mask * mask_color[c]
                        
                        # Add the mask to the image with transparency
                        alpha = 0.5
                        
                        # Apply mask to both RGB and depth display
                        mask_bool = mask > 0.5
                        
                        # Apply to color display
                        overlay = color_display.copy()
                        overlay[mask_bool] = (overlay[mask_bool] * (1-alpha) + colored_mask[mask_bool] * alpha).astype(np.uint8)
                        color_display = overlay
                        
                        # Apply to depth display
                        # depth_overlay = depth_display.copy()
                        # depth_overlay[mask_bool] = (depth_overlay[mask_bool] * (1-alpha) + colored_mask[mask_bool] * alpha).astype(np.uint8)
                        # depth_display = depth_overlay
                
                # Process point cloud
                point_cloud = depth_predictions["points"]
                
                # Convert point cloud tensor to numpy and reshape
                points_np = point_cloud.squeeze().cpu().numpy()  # Shape: (3, H, W)
                points_np = points_np.transpose(1, 2, 0)  # Shape: (H, W, 3)
                
                # Create and publish point cloud message with RGB
                point_cloud_msg = self.create_point_cloud_msg(points_np, color_image_rgb, msg.header)
                self.point_cloud_publisher.publish(point_cloud_msg)

                # Publish detected objects
                self.publish_detected_objects(detected_objects, msg.header)

                # Publish results
                # Convert depth to ROS Image message
                depth_msg = self.bridge.cv2_to_imgmsg(depth_display, "bgr8")
                depth_msg.header = msg.header
                self.depth_image_publisher.publish(depth_msg)
                
                # Convert annotated image to ROS Image message
                annotated_msg = self.bridge.cv2_to_imgmsg(color_display, "bgr8")
                annotated_msg.header = msg.header
                self.annotated_image_publisher.publish(annotated_msg)
                
        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")

def main(args=None):
    rclpy.init(args=args)
    
    depth_estimation_node = DepthEstimationNode()
    
    try:
        rclpy.spin(depth_estimation_node)
    except KeyboardInterrupt:
        pass
    finally:
        depth_estimation_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
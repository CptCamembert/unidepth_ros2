#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, CompressedImage, PointCloud2
from vision_msgs.msg import Detection3DArray
from cv_bridge import CvBridge
import numpy as np
import cv2
import torch

from helpers.depth_processor import DepthProcessor
from helpers.object_detector import ObjectDetector
from helpers.point_cloud_generator import PointCloudGenerator
from helpers.visualization import (
    colorize_depth, 
    draw_center_marker, 
    draw_detection_annotations, 
    apply_mask_overlay
)
import config


class ModularDepthEstimationNode(Node):
    def __init__(self):
        super().__init__('modular_depth_estimation_node')
        
        # Initialize components
        self.bridge = CvBridge()
        self.camera_intrinsics = None
        self.camera = None
        
        # Initialize helper classes
        self.get_logger().info("Loading models...")
        self.depth_processor = DepthProcessor(config.DEPTH_MODEL_NAME)
        self.object_detector = ObjectDetector(config.YOLO_MODEL_PATH)
        self.point_cloud_generator = PointCloudGenerator()
        
        # Setup ROS subscriptions and publishers
        self.setup_ros_interface()
        
        self.get_logger().info("Modular depth estimation node initialized. Waiting for camera data...")
    
    def setup_ros_interface(self):
        """Setup ROS subscriptions and publishers"""
        # Subscribers
        self.image_subscription = self.create_subscription(
            CompressedImage,
            config.IMAGE_TOPIC,
            self.image_callback,
            1)
        
        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            config.CAMERA_INFO_TOPIC,
            self.camera_info_callback,
            1)
        
        # Publishers
        self.depth_image_publisher = self.create_publisher(Image, config.DEPTH_IMAGE_TOPIC, 1)
        self.annotated_image_publisher = self.create_publisher(Image, config.ANNOTATED_IMAGE_TOPIC, 1)
        self.point_cloud_publisher = self.create_publisher(PointCloud2, config.POINT_CLOUD_TOPIC, 1)
        self.detected_objects_publisher = self.create_publisher(Detection3DArray, config.DETECTED_OBJECTS_TOPIC, 1)
    
    def camera_info_callback(self, msg):
        """Process camera info to setup intrinsics"""
        if self.camera_intrinsics is None:
            self.camera_intrinsics = msg
            self.camera = self.depth_processor.setup_camera(msg)
            self.get_logger().info("Camera intrinsics initialized")
    
    def process_image(self, color_image, color_image_rgb, msg_header):
        """Main image processing pipeline"""
        # Convert to tensor for depth processing
        rgb_tensor = torch.from_numpy(color_image_rgb).permute(2, 0, 1).float()
        rgb_tensor = rgb_tensor.to(self.depth_processor.device)
        
        # Process depth
        depth, points = self.depth_processor.process_depth(rgb_tensor, self.camera)
        
        # Detect objects
        detected_objects, detection_data = self.object_detector.detect_objects(
            color_image_rgb, depth, self.camera_intrinsics)
        
        # Create visualizations
        depth_colored = colorize_depth(depth)
        if color_image.shape[:2] != depth_colored.shape[:2]:
            depth_colored = cv2.resize(depth_colored, (color_image.shape[1], color_image.shape[0]))
        
        # Create display copies
        color_display = color_image.copy()
        depth_display = depth_colored.copy()
        
        # Draw center markers
        draw_center_marker(depth_display, depth, is_depth_image=True)
        draw_center_marker(color_display, depth, is_depth_image=False)
        
        # Apply detection visualizations
        draw_detection_annotations(color_display, detection_data, self.camera_intrinsics)
        
        # Apply mask overlays
        for detection in detection_data:
            color_display = apply_mask_overlay(
                color_display, detection['mask'], detection['class_id'], config.MASK_OVERLAY_ALPHA)
        
        return depth_display, color_display, points, detected_objects
    
    def publish_results(self, depth_display, color_display, points, color_image_rgb, detected_objects, header):
        """Publish all ROS messages"""
        # Publish depth image
        depth_msg = self.bridge.cv2_to_imgmsg(depth_display, "bgr8")
        depth_msg.header = header
        self.depth_image_publisher.publish(depth_msg)
        
        # Publish annotated image
        annotated_msg = self.bridge.cv2_to_imgmsg(color_display, "bgr8")
        annotated_msg.header = header
        self.annotated_image_publisher.publish(annotated_msg)
        
        # Publish point cloud
        point_cloud_msg = self.point_cloud_generator.create_point_cloud_msg(points, color_image_rgb, header)
        self.point_cloud_publisher.publish(point_cloud_msg)
        
        # Publish detected objects
        detection_array_msg = self.point_cloud_generator.create_detection_array_msg(detected_objects, header)
        self.detected_objects_publisher.publish(detection_array_msg)
    
    def image_callback(self, msg):
        """Main callback for processing incoming images"""
        if self.camera is None:
            return
        
        try:
            # Decode compressed image
            np_arr = np.frombuffer(msg.data, np.uint8)
            color_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if color_image is None:
                return
            
            color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            
            # Process image
            depth_display, color_display, points, detected_objects = self.process_image(
                color_image, color_image_rgb, msg.header)
            
            # Publish results
            self.publish_results(depth_display, color_display, points, color_image_rgb, detected_objects, msg.header)
                
        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")


def main(args=None):
    rclpy.init(args=args)
    node = ModularDepthEstimationNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
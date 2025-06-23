import numpy as np
import cv2
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
import tf2_ros
from tf_transformations import quaternion_from_matrix


class PoseEstimator:
    """
    Pose estimation helper using ORB features and depth-enhanced PnP algorithm.
    Based on the working motion_estimation_node implementation.
    """
    
    def __init__(self, logger=None):
        self.logger = logger
        
        # Camera calibration
        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_info_received = False
        
        # Previous frame data for tracking
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.prev_points_3d = None
        self.current_pose = np.eye(4)  # 4x4 transformation matrix (map to camera_link)
        
        # Motion estimation parameters
        self.motion_params = {
            'min_matches': 30,
            'min_3d_matches': 15,
            'ransac_threshold': 3.0,
            'pnp_ransac_threshold': 5.0,
            'ransac_confidence': 0.99,
            'max_iterations': 2000,
            'min_inliers': 15,
            'max_reprojection_error': 5.0,
            'depth_threshold': 0.1,  # Minimum valid depth in meters
            'max_depth': 10.0,       # Maximum valid depth in meters
            'max_translation_per_frame': 1.0,  # Max 1m per frame
            'max_rotation_per_frame': 0.5      # Max ~28 degrees per frame
        }
        
        # Feature matching parameters
        self.matching_params = {
            'ratio_threshold': 0.75,
            'cross_check': True,
            'max_distance': 100
        }
        
        # Initialize feature matcher
        self._setup_feature_matcher()
        
        # ORB detector
        self.orb = cv2.ORB_create(nfeatures=1000)
        
        # Frame transformation from optical to ROS standard frame
        self.optical_to_ros = np.array([
            [0,  0,  1,  0],  # X_ros = Z_cam (Forward = Forward)
            [-1, 0,  0,  0],  # Y_ros = -X_cam (Left = -Right)
            [0, -1,  0,  0],  # Z_ros = -Y_cam (Up = -Down)
            [0,  0,  0,  1]
        ])
        
        self._log_info("Pose estimator initialized")
    
    def _setup_feature_matcher(self):
        """Initialize feature matchers"""
        # FLANN matcher for ORB features
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                           table_number=6,
                           key_size=12,
                           multi_probe_level=1)
        search_params = dict(checks=50)
        
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Brute force matcher as fallback
        self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    def _log_info(self, message):
        """Log info message if logger is available"""
        if self.logger:
            self.logger.info(message)
    
    def _log_warn(self, message):
        """Log warning message if logger is available"""
        if self.logger:
            self.logger.warn(message)
    
    def _log_error(self, message):
        """Log error message if logger is available"""
        if self.logger:
            self.logger.error(message)
    
    def setup_camera(self, camera_info):
        """Setup camera calibration from ROS CameraInfo message"""
        if not self.camera_info_received:
            # Extract camera matrix
            self.camera_matrix = np.array(camera_info.k).reshape(3, 3).astype(np.float32)
            
            # Extract distortion coefficients
            self.dist_coeffs = np.array(camera_info.d).astype(np.float32)
            
            self.camera_info_received = True
            self._log_info(f"Camera calibration setup: fx={self.camera_matrix[0,0]:.1f}, fy={self.camera_matrix[1,1]:.1f}")
            self._log_info(f"Principal point: cx={self.camera_matrix[0,2]:.1f}, cy={self.camera_matrix[1,2]:.1f}")
            
            return True
        return False
    
    def detect_features(self, gray_image):
        """Detect ORB features in grayscale image"""
        keypoints, descriptors = self.orb.detectAndCompute(gray_image, None)
        return keypoints, descriptors
    
    def reconstruct_3d_points(self, keypoints, depth_image):
        """Reconstruct 3D points from 2D keypoints and depth image"""
        if not self.camera_info_received:
            self._log_warn("Camera not calibrated for 3D reconstruction")
            return np.array([], dtype=np.float32).reshape(0, 3)
        
        points_3d = []
        valid_indices = []
        
        fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
        cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]
        
        for i, kp in enumerate(keypoints):
            # Get pixel coordinates
            u, v = int(kp.pt[0]), int(kp.pt[1])
            
            # Check bounds
            if (0 <= u < depth_image.shape[1] and 0 <= v < depth_image.shape[0]):
                # Sample depth in a small neighborhood for robustness
                depth_patch = depth_image[max(0, v-1):min(depth_image.shape[0], v+2),
                                        max(0, u-1):min(depth_image.shape[1], u+2)]
                
                # Use median depth to handle noise
                valid_depths = depth_patch[depth_patch > 0]
                if len(valid_depths) > 0:
                    depth = np.median(valid_depths)
                    
                    # Validate depth range
                    if (self.motion_params['depth_threshold'] <= depth <= self.motion_params['max_depth']):
                        # Project to 3D in camera coordinate system
                        x = (u - cx) * depth / fx
                        y = (v - cy) * depth / fy
                        z = depth
                        
                        points_3d.append([x, y, z])
                        valid_indices.append(i)
        
        if len(points_3d) > 0:
            self._log_info(f"Reconstructed {len(points_3d)} 3D points from {len(keypoints)} features")
            return np.array(points_3d, dtype=np.float32)
        else:
            self._log_warn("No valid 3D points reconstructed")
            return np.array([], dtype=np.float32).reshape(0, 3)
    
    def match_features(self, desc1, desc2):
        """Match features between two descriptor sets using ratio test"""
        try:
            # Use FLANN matcher
            matches = self.flann.knnMatch(desc1, desc2, k=2)
            
            # Apply ratio test (Lowe's ratio test)
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < self.matching_params['ratio_threshold'] * n.distance:
                        good_matches.append(m)
            
            return good_matches
            
        except Exception as e:
            self._log_warn(f"FLANN matching failed: {e}. Using BF matcher.")
            
            # Fallback to brute force matcher
            matches = self.bf_matcher.knnMatch(desc1, desc2, k=2)
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < self.matching_params['ratio_threshold'] * n.distance:
                        good_matches.append(m)
            
            return good_matches
    
    def estimate_pose(self, rgb_image, depth_image):
        """
        Estimate camera pose from RGB and depth images.
        Returns: (success, pose_matrix, num_matches, num_inliers)
        """
        if not self.camera_info_received:
            self._log_warn("Camera not calibrated for pose estimation")
            return False, None, 0, 0
        
        # Convert to grayscale
        if len(rgb_image.shape) == 3:
            gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = rgb_image
        
        # Detect features in current frame
        current_keypoints, current_descriptors = self.detect_features(gray_image)
        
        if current_descriptors is None:
            self._log_warn("No features detected in current frame")
            return False, None, 0, 0
        
        # If we have a previous frame with 3D points, estimate motion using PnP
        if (self.prev_descriptors is not None and len(self.prev_descriptors) > 0 and 
            self.prev_points_3d is not None and len(self.prev_points_3d) > 0):
            
            success, num_matches, num_inliers = self._estimate_motion_pnp(
                self.prev_keypoints, self.prev_descriptors, self.prev_points_3d,
                current_keypoints, current_descriptors
            )
            
            if success:
                # Reconstruct 3D points from current frame for next iteration
                current_points_3d = self.reconstruct_3d_points(current_keypoints, depth_image)
                
                # Store current frame data for next iteration
                self.prev_frame = gray_image.copy()
                self.prev_keypoints = current_keypoints
                self.prev_descriptors = current_descriptors
                self.prev_points_3d = current_points_3d
                
                return True, self.current_pose.copy(), num_matches, num_inliers
        
        # Initialize or store current frame data for first frame or when tracking fails
        current_points_3d = self.reconstruct_3d_points(current_keypoints, depth_image)
        
        self.prev_frame = gray_image.copy()
        self.prev_keypoints = current_keypoints
        self.prev_descriptors = current_descriptors
        self.prev_points_3d = current_points_3d
        
        # Return current pose (no motion estimated)
        return False, self.current_pose.copy(), 0, 0
    
    def _estimate_motion_pnp(self, prev_kp, prev_desc, prev_3d_points, curr_kp, curr_desc):
        """Estimate camera motion using 3D-2D point correspondences (PnP)"""
        try:
            # Match features between frames
            matches = self.match_features(prev_desc, curr_desc)
            
            if len(matches) < self.motion_params['min_matches']:
                self._log_warn(f"Not enough matches: {len(matches)} < {self.motion_params['min_matches']}")
                return False, len(matches), 0
            
            # Extract corresponding 3D points and 2D points
            object_points = []  # 3D points from previous frame
            image_points = []   # 2D points in current frame
            valid_matches = []
            
            for match in matches:
                prev_idx = match.queryIdx
                curr_idx = match.trainIdx
                
                # Check if we have a valid 3D point for this match
                if prev_idx < len(prev_3d_points):
                    prev_3d = prev_3d_points[prev_idx]
                    curr_2d = curr_kp[curr_idx].pt
                    
                    # Validate the 3D point
                    if (np.all(np.isfinite(prev_3d)) and 
                        np.linalg.norm(prev_3d) > self.motion_params['depth_threshold']):
                        
                        object_points.append(prev_3d)
                        image_points.append(curr_2d)
                        valid_matches.append(match)
            
            if len(object_points) < self.motion_params['min_3d_matches']:
                self._log_warn(f"Not enough 3D-2D correspondences: {len(object_points)} < {self.motion_params['min_3d_matches']}")
                return False, len(matches), 0
            
            # Convert to numpy arrays
            object_points = np.array(object_points, dtype=np.float32)
            image_points = np.array(image_points, dtype=np.float32)
            
            # Solve PnP problem with RANSAC
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                object_points,
                image_points,
                self.camera_matrix,
                self.dist_coeffs,
                iterationsCount=self.motion_params['max_iterations'],
                reprojectionError=self.motion_params['pnp_ransac_threshold'],
                confidence=self.motion_params['ransac_confidence']
            )
            
            if not success or inliers is None or len(inliers) < self.motion_params['min_inliers']:
                self._log_warn(f"PnP failed or insufficient inliers: {len(inliers) if inliers is not None else 0}")
                return False, len(matches), len(inliers) if inliers is not None else 0
            
            # Convert rotation vector to rotation matrix
            R, _ = cv2.Rodrigues(rvec)
            
            # Validate motion magnitude
            translation_norm = np.linalg.norm(tvec)
            rotation_angle = np.linalg.norm(rvec)
            
            if translation_norm > self.motion_params['max_translation_per_frame']:
                self._log_warn(f"Large translation: {translation_norm:.3f}m - rejecting")
                return False, len(matches), len(inliers)
                
            if rotation_angle > self.motion_params['max_rotation_per_frame']:
                self._log_warn(f"Large rotation: {rotation_angle:.3f} rad - rejecting")
                return False, len(matches), len(inliers)
            
            # Create transformation matrix (camera motion in optical frame)
            camera_motion_optical = np.eye(4)
            camera_motion_optical[:3, :3] = R
            camera_motion_optical[:3, 3] = tvec.flatten()
            
            # Convert camera motion to ROS frame
            camera_motion_ros = self.optical_to_ros @ camera_motion_optical @ np.linalg.inv(self.optical_to_ros)
            
            # For pose tracking, we need to invert this to get world frame motion
            camera_motion_inv = np.linalg.inv(camera_motion_ros)
            
            # Update current pose by applying the inverse motion
            self.current_pose = self.current_pose @ camera_motion_inv
            
            self._log_info(f"PnP motion estimated: {len(valid_matches)} 3D-2D matches, {len(inliers)} inliers")
            self._log_info(f"  Translation: {translation_norm:.4f}m, Rotation: {rotation_angle:.4f}rad")
            self._log_info(f"  Position: [{self.current_pose[0,3]:.3f}, {self.current_pose[1,3]:.3f}, {self.current_pose[2,3]:.3f}]")
            
            return True, len(matches), len(inliers)
            
        except Exception as e:
            self._log_error(f"Error in PnP motion estimation: {e}")
            return False, 0, 0
    
    def get_pose_as_ros_messages(self, header, frame_id="map", child_frame_id="camera_link"):
        """
        Get current pose as ROS messages (PoseStamped and TransformStamped).
        Returns: (pose_msg, transform_msg)
        """
        # Create PoseStamped message
        pose_msg = PoseStamped()
        pose_msg.header = header
        pose_msg.header.frame_id = frame_id
        
        # Extract position
        pose_msg.pose.position.x = self.current_pose[0, 3]
        pose_msg.pose.position.y = self.current_pose[1, 3]
        pose_msg.pose.position.z = self.current_pose[2, 3]
        
        # Extract rotation as quaternion
        rotation_matrix = self.current_pose[:3, :3]
        full_matrix = np.eye(4)
        full_matrix[:3, :3] = rotation_matrix
        
        quaternion = quaternion_from_matrix(full_matrix)
        pose_msg.pose.orientation.x = quaternion[0]
        pose_msg.pose.orientation.y = quaternion[1]
        pose_msg.pose.orientation.z = quaternion[2]
        pose_msg.pose.orientation.w = quaternion[3]
        
        # Create TransformStamped message
        transform_msg = TransformStamped()
        transform_msg.header = header
        transform_msg.header.frame_id = frame_id
        transform_msg.child_frame_id = child_frame_id
        
        transform_msg.transform.translation.x = pose_msg.pose.position.x
        transform_msg.transform.translation.y = pose_msg.pose.position.y
        transform_msg.transform.translation.z = pose_msg.pose.position.z
        
        transform_msg.transform.rotation = pose_msg.pose.orientation
        
        return pose_msg, transform_msg
    
    def get_odometry_message(self, header, frame_id="map", child_frame_id="camera_link"):
        """Get current pose as Odometry message"""
        odom_msg = Odometry()
        odom_msg.header = header
        odom_msg.header.frame_id = frame_id
        odom_msg.child_frame_id = child_frame_id
        
        # Position
        odom_msg.pose.pose.position.x = self.current_pose[0, 3]
        odom_msg.pose.pose.position.y = self.current_pose[1, 3]
        odom_msg.pose.pose.position.z = self.current_pose[2, 3]
        
        # Orientation
        rotation_matrix = self.current_pose[:3, :3]
        full_matrix = np.eye(4)
        full_matrix[:3, :3] = rotation_matrix
        
        quaternion = quaternion_from_matrix(full_matrix)
        odom_msg.pose.pose.orientation.x = quaternion[0]
        odom_msg.pose.pose.orientation.y = quaternion[1]
        odom_msg.pose.pose.orientation.z = quaternion[2]
        odom_msg.pose.pose.orientation.w = quaternion[3]
        
        # TODO: Add velocity estimation
        
        return odom_msg
    
    def reset_pose(self, initial_pose=None):
        """Reset pose to initial position"""
        if initial_pose is not None:
            self.current_pose = initial_pose.copy()
        else:
            self.current_pose = np.eye(4)
        
        # Clear previous frame data
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.prev_points_3d = None
        
        self._log_info("Pose estimator reset")
    
    def get_current_position(self):
        """Get current position as [x, y, z]"""
        return self.current_pose[:3, 3]
    
    def get_current_rotation_matrix(self):
        """Get current rotation matrix"""
        return self.current_pose[:3, :3]
    
    def set_motion_parameters(self, **kwargs):
        """Update motion estimation parameters"""
        for key, value in kwargs.items():
            if key in self.motion_params:
                self.motion_params[key] = value
                self._log_info(f"Updated motion parameter {key} = {value}")
    
    def set_matching_parameters(self, **kwargs):
        """Update feature matching parameters"""
        for key, value in kwargs.items():
            if key in self.matching_params:
                self.matching_params[key] = value
                self._log_info(f"Updated matching parameter {key} = {value}")
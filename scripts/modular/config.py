# Configuration for modular depth estimation node

# Model configurations
DEPTH_MODEL_NAME = "unidepth-v2-vitl14"
YOLO_MODEL_PATH = "yolo11n-seg.pt"

# ROS topic names
IMAGE_TOPIC = "/camera/camera/color/image_raw/compressed"
CAMERA_INFO_TOPIC = "/camera/camera/color/camera_info"
DEPTH_IMAGE_TOPIC = "/depth/image_raw"
ANNOTATED_IMAGE_TOPIC = "/depth/annotated_image"
POINT_CLOUD_TOPIC = "/depth/point_cloud"
DETECTED_OBJECTS_TOPIC = "/detected_objects"

# SLAM topic names
SLAM_POSE_TOPIC = "/slam/pose"
SLAM_POSE_COV_TOPIC = "/slam/pose_with_covariance"
SLAM_PATH_TOPIC = "/slam/path"
SLAM_MATCHES_TOPIC = "/slam/feature_matches"
SLAM_LANDMARKS_TOPIC = "/slam/landmarks"

# SLAM parameters
SLAM_MAX_FEATURES = 1000
SLAM_MATCH_RATIO = 0.7
SLAM_RANSAC_THRESHOLD = 1.0
SLAM_MAX_PATH_LENGTH = 1000

# Landmark parameters
LANDMARK_MIN_DISTANCE = 0.20  # Minimum distance between landmarks in meters
LANDMARK_SAMPLING_PERCENTAGE = 10.0  # Percentage of features to test for landmark creation (2% = every 50th feature)

# Processing parameters
MASK_OVERLAY_ALPHA = 0.5
POINT_CLOUD_DOWNSAMPLE_FACTOR = 4  # Take every Nth pixel for point cloud generation
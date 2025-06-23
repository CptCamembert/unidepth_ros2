import numpy as np
import cv2
import matplotlib.pyplot as plt


def colorize_depth(depth, min_depth=-1.0, max_depth=-1.0):
    """Convert depth map to color visualization"""
    if max_depth == -1.0:
        max_depth = np.max(depth)
    if min_depth == -1.0:
        min_depth = np.min(depth)
    depth_clipped = np.clip(depth, min_depth, max_depth)
    
    normalized_depth = (depth_clipped - min_depth) / (max_depth - min_depth)
    colorized = plt.cm.inferno(1 - normalized_depth)
    rgb_image = (colorized[:, :, :3] * 255).astype(np.uint8)
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    return bgr_image


def draw_center_marker(image, depth, is_depth_image=True):
    """Draw center marker and depth text on image"""
    center_y, center_x = depth.shape[0] // 2, depth.shape[1] // 2
    center_depth = depth[center_y, center_x]
    
    color = (255, 255, 255) if is_depth_image else (0, 255, 0)
    cv2.drawMarker(image, (center_x, center_y), color, 
                  markerType=cv2.MARKER_CROSS, markerSize=20, thickness=1)
    
    if is_depth_image:
        depth_text = f"Center Depth: {float(center_depth):.2f}m"
        cv2.putText(image, depth_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, (255, 255, 255), 2, 1)


def draw_detection_annotations(image, detections, camera_intrinsics):
    """Draw detection annotations on image"""
    for detection in detections:
        box_center_x = int((detection['x1'] + detection['x2']) // 2)
        box_center_y = int((detection['y1'] + detection['y2']) // 2)
        
        # Draw marker
        cv2.drawMarker(image, (box_center_x, box_center_y), (255, 255, 255), 
                      markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
        
        # Draw text
        text = f"{detection['class_name']} ({detection['confidence']:.2f})\n{detection['depth']:.2f}m\n({detection['x']:.2f}, {detection['y']:.2f}, {detection['z']:.2f})"
        y_offset = box_center_y - 30
        for line in text.split('\n'):
            cv2.putText(image, line, (box_center_x + 5, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            y_offset += 15


def apply_mask_overlay(image, mask, class_id, alpha=0.5):
    """Apply colored mask overlay to image"""
    color_value = int((class_id * 23 % 80) * 255 / 80)
    colormap = cv2.COLORMAP_RAINBOW
    color_array = np.zeros((1, 1, 1), dtype=np.uint8)
    color_array[0, 0, 0] = color_value
    
    colored_array = cv2.applyColorMap(color_array, colormap)
    mask_color = colored_array[0, 0].tolist()
    
    colored_mask = np.zeros_like(image)
    for c in range(3):
        colored_mask[:, :, c] = mask * mask_color[c]
    
    mask_bool = mask > 0.5
    overlay = image.copy()
    overlay[mask_bool] = (overlay[mask_bool] * (1-alpha) + colored_mask[mask_bool] * alpha).astype(np.uint8)
    
    return overlay
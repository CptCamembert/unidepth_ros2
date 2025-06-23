# UniDepth ROS2 Implementation

[![Original UniDepth](https://img.shields.io/badge/Based%20on-UniDepth-blue)](https://github.com/lpiccinelli-eth/UniDepth)
[![ROS2](https://img.shields.io/badge/ROS2-Humble-orange)](https://docs.ros.org/en/humble/)

This repository contains a ROS2 implementation of the UniDepth universal monocular metric depth estimation system, designed for real-time depth estimation and object detection using Intel RealSense cameras.

## Original UniDepth Project

This implementation is based on the original UniDepth project by Luigi Piccinelli et al. For detailed information about the underlying model, please refer to the [original UniDepth README](README_UNIDEPTH.md).

**Original Papers:**
- [UniDepthV2: Universal Monocular Metric Depth Estimation Made Simpler](https://arxiv.org/abs/2502.20110)
- [UniDepth: Universal Monocular Metric Depth Estimation](https://arxiv.org/abs/2403.18913) (CVPR 2024)

## Features

- **Real-time depth estimation** using UniDepthV2 models
- **Object detection and segmentation** with YOLOv11
- **3D point cloud generation** with RGB information
- **ROS2 integration** for robotics applications
- **Intel RealSense camera support**
- **Configurable model backends** (ViT-S, ViT-B, ViT-L)

## System Requirements

- **OS:** Linux (tested on Ubuntu 22.04)
- **Python:** 3.10+
- **CUDA:** 11.8+ (recommended for GPU acceleration)
- **ROS2:** Humble Hawksbill
- **Hardware:** Intel RealSense camera (D435, D455, etc.)

## Installation

### 1. Prerequisites

Ensure you have ROS2 Humble installed:
```bash
# Follow the official ROS2 Humble installation guide
# https://docs.ros.org/en/humble/Installation.html
```

### 2. Install Intel RealSense SDK

```bash
# Install RealSense SDK
sudo apt-get update && sudo apt-get upgrade && sudo apt-get dist-upgrade
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" -u
sudo apt-get install librealsense2-dkms librealsense2-utils librealsense2-dev librealsense2-dbg

# Install RealSense ROS2 wrapper
sudo apt install ros-humble-realsense2-*
```

### 3. Set up Python Environment

```bash
# Create a virtual environment (recommended)
python3 -m venv ~/unidepth_env
source ~/unidepth_env/bin/activate

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt

# Install additional ROS2 Python dependencies
pip install cv-bridge sensor-msgs vision-msgs ultralytics
```

### 4. Install UniDepth Package

```bash
# Navigate to the UniDepth directory
cd /path/to/your/UniDepth/folder

# Install UniDepth in development mode
pip install -e .

# Optional: Install Pillow-SIMD for better performance
pip uninstall pillow
CC="cc -mavx2" pip install -U --force-reinstall pillow-simd

# Compile KNN operations (for evaluation only)
cd unidepth/ops/knn && bash compile.sh && cd ../../../
```

## Usage

### 1. Start the RealSense Camera Node

First, launch the RealSense camera node:

```bash
# Source ROS2 environment
source /opt/ros/humble/setup.bash

# Launch RealSense camera with default settings
ros2 launch realsense2_camera rs_launch.py

# Or with custom parameters (recommended)
ros2 launch realsense2_camera rs_launch.py \
    pointcloud.enable:=true \
    align_depth.enable:=true \
    enable_color:=true \
    enable_depth:=true \
    color_width:=640 \
    color_height:=480 \
    color_fps:=30.0
```

### 2. Run the UniDepth Node

Navigate to the UniDepth directory and run one of the main nodes:

```bash
# Activate your Python environment
source ~/unidepth_env/bin/activate

# Navigate to UniDepth folder
cd /path/to/your/UniDepth/folder

# Run the main node (depth estimation only)
python3 scripts/modular/main_node.py

# OR run the pose estimation node (with relative ORB tracking)
python3 scripts/modular/main_node_pose.py
```

### 3. Alternative: Direct Script Execution

You can also run the standalone depth estimation script:

```bash
cd /path/to/your/UniDepth/folder
python3 scripts/estimate_depth_mask_node.py
```

## ROS2 Topics

### Published Topics

- `/depth/image_raw` - Raw depth image
- `/depth/annotated_image` - Color image with object annotations
- `/depth/point_cloud` - 3D point cloud with RGB information
- `/detected_objects` - 3D object detection results

### Subscribed Topics

- `/camera/camera/color/image_raw/compressed` - Compressed RGB image from RealSense
- `/camera/camera/color/camera_info` - Camera intrinsic parameters

## Configuration

### Model Selection

You can change the UniDepth model by modifying the `depth_model_name` variable in the scripts:

```python
# Available models:
# - "unidepth-v2-vits14" (fastest, lower accuracy)
# - "unidepth-v2-vitb14" (balanced)
# - "unidepth-v2-vitl14" (slowest, highest accuracy)

depth_model_name = "unidepth-v2-vitb14"  # Change this line
```

### Performance Optimization

For real-time performance:
1. Use a GPU with CUDA support
2. Choose a smaller model (vits14 or vitb14)
3. Reduce camera resolution if needed
4. Ensure proper cooling for sustained performance

## Troubleshooting

### Common Issues

1. **"No CUDA device found"**
   - Ensure NVIDIA drivers and CUDA are properly installed
   - Check PyTorch CUDA compatibility: `python -c "import torch; print(torch.cuda.is_available())"`

2. **"Camera not detected"**
   - Verify RealSense camera connection: `realsense-viewer`
   - Check USB3 connection and power

3. **"Segmentation Fault"**
   - Try installing PyTorch via conda instead of pip
   - Ensure xformers version compatibility

4. **Slow performance**
   - Switch to a smaller model (vits14)
   - Enable GPU acceleration
   - Reduce camera resolution

### Performance Tips

- **GPU Memory:** ViT-L models require ~8GB VRAM, ViT-B ~4GB, ViT-S ~2GB
- **CPU Performance:** Use Pillow-SIMD for image processing acceleration
- **Real-time Processing:** Consider frame skipping for very slow hardware

## Development

### Adding Custom Nodes

The modular structure allows easy extension:

```bash
scripts/modular/
├── main_node.py          # Basic depth estimation
├── main_node_pose.py     # With ORB tracking
├── config.py             # Configuration settings
└── helpers/              # Utility functions
```

### Contributing

1. Fork this repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project maintains the same license as the original UniDepth project: **Creative Commons BY-NC 4.0 license**.

## Citation

If you use this ROS2 implementation, please cite the original UniDepth papers:

```bibtex
@inproceedings{piccinelli2024unidepth,
    title     = {{U}ni{D}epth: Universal Monocular Metric Depth Estimation},
    author    = {Piccinelli, Luigi and Yang, Yung-Hsu and Sakaridis, Christos and Segu, Mattia and Li, Siyuan and Van Gool, Luc and Yu, Fisher},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2024}
}
```

```bibtex
@misc{piccinelli2025unidepthv2,
      title={{U}ni{D}epth{V2}: Universal Monocular Metric Depth Estimation Made Simpler}, 
      author={Luigi Piccinelli and Christos Sakaridis and Yung-Hsu Yang and Mattia Segu and Siyuan Li and Wim Abbeloos and Luc Van Gool},
      year={2025},
      eprint={2502.20110},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.20110}, 
}
```

## Acknowledgments

- Original UniDepth authors for the excellent depth estimation model
- Intel RealSense team for the camera SDK and ROS2 wrapper
- Ultralytics for YOLOv11 object detection

## Support

For issues related to:
- **Original UniDepth model:** Contact Luigi Piccinelli (lpiccinelli@ethz.ch)
- **ROS2 implementation:** Open an issue in this repository
- **RealSense camera:** Check Intel RealSense documentation
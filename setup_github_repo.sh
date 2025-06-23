#!/bin/bash

# UniDepth ROS2 GitHub Repository Setup Script
# This script helps initialize a GitHub repository for the UniDepth ROS2 implementation

echo "=== UniDepth ROS2 GitHub Repository Setup ==="
echo ""

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ] || [ ! -d "unidepth" ]; then
    echo "Error: Please run this script from the UniDepth root directory"
    echo "Expected files: pyproject.toml, unidepth/ folder"
    exit 1
fi

echo "✓ Detected UniDepth project structure"

# Initialize git repository if not already initialized
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
    echo "✓ Git repository initialized"
else
    echo "✓ Git repository already exists"
fi

# Add files to git
echo "Adding files to git..."
git add .gitignore
git add README_ROS2.md
git add requirements_ros2.txt
git add pyproject.toml
git add unidepth/
git add scripts/
git add assets/ 2>/dev/null || true  # Add assets if they exist
git add LICENSE 2>/dev/null || true  # Add license if it exists

echo "✓ Files added to git staging area"

# Create initial commit if no commits exist
if ! git rev-parse HEAD >/dev/null 2>&1; then
    echo "Creating initial commit..."
    git commit -m "Initial commit: UniDepth ROS2 implementation

- Added ROS2-specific README and requirements
- Included depth estimation and object detection nodes
- Support for Intel RealSense cameras
- Modular architecture for easy extension"
    echo "✓ Initial commit created"
else
    echo "! Git repository already has commits"
    echo "  You may want to commit the new files manually:"
    echo "  git commit -m 'Add ROS2 documentation and setup files'"
fi

echo ""
echo "=== Next Steps ==="
echo ""
echo "1. Create a new repository on GitHub:"
echo "   - Go to https://github.com/new"
echo "   - Repository name: unidepth-ros2 (or your preferred name)"
echo "   - Description: ROS2 implementation of UniDepth for real-time depth estimation"
echo "   - Make it public or private as needed"
echo "   - Do NOT initialize with README, .gitignore, or license (we already have them)"
echo ""
echo "2. Add the GitHub remote and push:"
echo "   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "3. Optional: Set up GitHub Pages for documentation:"
echo "   - Go to your repository settings"
echo "   - Scroll to 'Pages' section"
echo "   - Select 'Deploy from a branch' and choose 'main' branch"
echo ""
echo "=== Repository Structure ==="
echo ""
echo "Your repository will include:"
echo "├── README_ROS2.md          # Main documentation for ROS2 users"
echo "├── README.md               # Original UniDepth documentation"
echo "├── requirements_ros2.txt   # Python dependencies for ROS2"
echo "├── requirements.txt        # Original UniDepth requirements"
echo "├── .gitignore             # Git ignore patterns"
echo "├── pyproject.toml         # Python package configuration"
echo "├── unidepth/              # UniDepth core library"
echo "├── scripts/               # ROS2 nodes and utilities"
echo "│   ├── modular/           # Modular node implementations"
echo "│   │   ├── main_node.py   # Basic depth estimation"
echo "│   │   └── main_node_pose.py # With ORB tracking"
echo "│   └── estimate_depth_mask_node.py # Standalone script"
echo "└── assets/                # Documentation assets (if present)"
echo ""
echo "=== Recommended Repository Settings ==="
echo ""
echo "- Repository name: unidepth-ros2"
echo "- Topics/Tags: ros2, depth-estimation, computer-vision, realsense, yolo, pytorch"
echo "- Description: 'ROS2 implementation of UniDepth for real-time monocular depth estimation with Intel RealSense cameras'"
echo ""
echo "Happy coding! 🚀"
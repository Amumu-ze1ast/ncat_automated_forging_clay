# NCAT Automated Forging Clay

An autonomous robotic system for automated clay forging and shaping using computer vision, motion planning, and 3D reconstruction techniques.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Project Modules](#project-modules)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This project implements an autonomous robotic welding and forging system for clay manipulation and shaping. It combines:
- **Computer Vision**: Real-time marker detection and 3D reconstruction
- **Motion Planning**: Adaptive trajectory generation for complex shapes
- **Robotics Control**: Integration with Kinova Gen3 robotic arm and Clearpath Husky A200
- **State Machine Architecture**: FlexBE-based behavior control

The system can detect hand-drawn or designed patterns on a whiteboard, convert them to 3D trajectories, and execute automated welding/shaping operations.

---

## ✨ Features

- ✅ **Multi-Segment Path Support**: Handle disconnected or multiple drawing paths
- ✅ **Computer Vision Pipeline**: OpenCV-based contour detection and waypoint extraction
- ✅ **Flexible Starting Points**: Control where each path begins (top, bottom, left, right, corners)
- ✅ **Customizable Path Order**: Reorder execution sequence (left-to-right, right-to-left, custom order)
- ✅ **3D Reconstruction**: Convert 2D pixel coordinates to 3D robot coordinates
- ✅ **Depth Integration**: Real-time depth camera integration (RealSense D435, Orbbec Femto Bolt)
- ✅ **Z-Axis Control**: Automatic pen-up/pen-down state management
- ✅ **TF2 Transformation**: Seamless coordinate frame conversion (camera → base_link)
- ✅ **ROS2 Integration**: Full ROS2 Humble compatibility
- ✅ **Visualization Tools**: RViz-compatible marker array visualization

---

## 🏗️ System Architecture

---

## 🛠️ Installation

### Prerequisites

- **OS**: Ubuntu 22.04 LTS
- **ROS**: ROS2 Humble
- **Python**: 3.10+
- **Hardware**: 
  - Kinova Gen3 Robotic Arm
  - Clearpath Husky A200 Mobile Base (optional)
  - RealSense D435 or Orbbec Femto Bolt depth camera

### Step 1: Clone the Repository

```bash
cd ~/ros2_workspace/src
git clone https://github.com/ncat/ncat_automated_forging_clay.git
```

### Step 2: Install Dependencies

```bash
# Update package list
sudo apt update && sudo apt upgrade

# Install required packages
sudo apt install -y python3-pip libopencv-dev python3-opencv

# Install Python dependencies
pip3 install numpy scipy opencv-contrib-python

# Install ROS2 packages
sudo apt install -y ros-humble-kinova-gen3-control ros-humble-realsense2-camera
```

### Step 3: Build the Workspace

```bash
cd ~/ros2_workspace
colcon build --symlink-install

# Source the workspace
source install/setup.bash
```

---

## 🚀 Usage

### 1. Basic Vision Pipeline (Steps 1-4)

Process a whiteboard image to extract waypoints:

```bash
# Step 1: RGB + Depth Image Capture
# (Captured automatically or use existing images in raw_files/)

# Step 2: Generate Black Mask
python3 step2_black_mask_generation.py

# Step 3: Extract Skeleton & Contours
python3 step3_skeleton_contours.py

# Step 4: Extract & Order Waypoints
python3 step4_waypoints_extraction.py
  # Configure: CONTOUR_ORDER, START_FROM
```

### 2. 3D Conversion (Step 5)

Convert pixel coordinates to 3D camera frame:

```bash
python3 step5_pixel_to_3d.py
```

### 3. Coordinate Transformation (Step 6)

Transform from camera frame to robot base_link:

```bash
# Ensure TF2 is broadcasting camera→base_link transform
ros2 launch ncat_automated_forging_clay camera_tf_broadcaster.launch.py

# Run transformation
python3 step6_transform_to_base.py
```

### 4. Visualization (RViz)

Visualize waypoints in RViz:

```bash
# Terminal 1: Start ROS2 nodes
ros2 launch ncat_automated_forging_clay visualization.launch.py

# Terminal 2: Open RViz
rviz2
  - Fixed Frame: base_link
  - Add MarkerArray display
  - Topic: /visualization_marker_array
```

### 5. Robot Execution (Step 7)

Send trajectory to robot:

```bash
python3 step7_ik_solver_execution.py
```

---

## 📦 Project Modules

### flexbe_behavior_engine
Core FlexBE (Flexible Behavior Engine) framework for behavior trees and state machines.

**Key Components:**
- Behavior tree execution engine
- State management
- Action/feedback handling
- Mirror visualization

### flexbe_kinovagen3
Kinova Gen3-specific FlexBE integration.

**Features:**
- Joint control states
- Cartesian motion states
- Gripper control
- Vision integration states
- Collision avoidance

**States Included:**
- `joint_trajectory_state`: Execute joint trajectories
- `cartesian_move_state`: Cartesian motion planning
- `gripper_control_state`: Gripper open/close
- `vision_detect_state`: Object detection via color filtering

### flexbe_webui
Web-based UI for FlexBE behavior monitoring and control.

**Functionality:**
- Real-time state visualization
- Behavior tree editor
- Parameter tuning
- Execution monitoring
- Logging and debugging

---

## 🔧 Configuration

### Camera Intrinsics (step5_pixel_to_3d.py)

```python
fx = 1297.672904  # Focal length X
fy = 1298.631344  # Focal length Y
cx = 620.914026   # Principal point X
cy = 238.280325   # Principal point Y
```

**Calibrate using:**
```bash
ros2 run camera_calibration cameracalibrator.py \
  --size 8x6 --square 0.05 image:=/camera/color/image_raw
```

### Waypoint Configuration (step4_waypoints_extraction.py)

```python
SAMPLE_RATE = 3              # Points per waypoint (lower = more waypoints)
START_FROM = "top"           # Starting point: top, bottom, left, right, etc.
CONTOUR_ORDER = None         # Execution order: None, "reverse", "left_to_right", [custom]
```

### Z-Axis Control (step5_pixel_to_3d.py)

```python
Z_DOWN = 0.0    # Z offset when pen is down (on surface)
Z_UP = 0.05     # Z offset when pen is up (lifted 5cm)
```

---

## 📊 Output Files

Pipeline generates the following outputs:

| Step | Output File | Format | Description |
|------|------------|--------|-------------|
| 2 | `2black_mask.png` | PNG | Binary mask of drawing |
| 3 | `3line_skeleton.png` | PNG | Skeleton after thinning |
| 3 | `3line_contours.npy` | NumPy | Extracted contours |
| 4 | `4waypoints_pixels.npy` | NumPy | [x, y, segment_id] |
| 4 | `4waypoints_visualization.png` | PNG | Visualization with markers |
| 5 | `5waypoints_3d_camera.npy` | NumPy | [X, Y, Z] in camera frame |
| 5 | `5waypoints_3d_camera_with_segments.npy` | NumPy | With segment/pen_down info |
| 6 | `6waypoints_3d_base.npy` | NumPy | [X, Y, Z] in base_link frame |
| 6 | `6waypoints_3d_base_with_segments.npy` | NumPy | With segment/pen_down info |

---

## 📋 Requirements

### Hardware

- **Robot Arm**: Kinova Gen3 (7-DOF, 800mm reach)
- **Mobile Base**: Clearpath Husky A200 (optional)
- **End Effector**: Robotiq 2F-85 adaptive gripper + TSF-85 tactile fingertips
- **Camera**: RealSense D435 or Orbbec Femto Bolt (RGB-D)
- **Force/Torque Sensor**: Bota Systems F/T sensor

### Software

- ROS2 Humble
- OpenCV 4.5.5+
- NumPy 1.21+
- SciPy 1.7+
- PyYAML
- transformations

### Network

- Robot communication on 192.168.1.x subnet
- Camera USB 3.0 connection
- WiFi for ROS2 DDS (QoS configured)

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Camera Not Detected
```bash
# Check USB devices
lsusb | grep Intel/Orbbec

# Grant permissions
sudo usermod -aG dialout $USER
```

#### 2. TF2 Transform Missing
```bash
# Check available transforms
ros2 run tf2_tools view_frames

# Verify broadcaster is running
ros2 topic echo /tf
```

#### 3. IK Solver Failure
- Ensure target is within robot workspace
- Check joint limits and collision geometry
- Verify TRAC-IK parameters

#### 4. Depth Image Issues
```bash
# Check depth image quality
ros2 run rqt_image_view rqt_image_view
  # Subscribe to /camera/depth/image_raw
```

---

### Areas for Contribution

- [ ] Additional end-effector support
- [ ] Mobile base integration improvements
- [ ] Deep learning-based object detection
- [ ] Real-time force feedback control
- [ ] Multi-robot coordination
- [ ] Performance optimization

---

## 👥 Authors

**Amanuel Abrdo Tereda**
- Ph.D. Candidate in Mechanical Engineering
- North Carolina A&T State University
- Advisor: Dr. Sun Yi
- Email: [aatereda@aggies.ncat.edu](mailto:aatereda@aggies.ncat.edu)

---

## 🙏 Acknowledgments

- **NSF** - Research funding
- **HAMMER** - Hardware support
- **NCDOT** - Transportation research collaboration
- **NNL** - National laboratory partnership
- **Dr. Sun Yi** - Academic advisor and guidance

---

## 📞 Support

For issues, questions, or collaboration inquiries:

- 📧 **Email**: aatereda@aggies.ncat.edu


---

## 📈 Project Status

**Current Release**: v0.1.0 (Alpha)

**Development Status**:
- ✅ Vision pipeline (complete)
- ✅ 3D conversion (complete)
- ✅ Coordinate transformation (complete)
- 🔄 IK solver integration (in progress)
- 🔄 Real-world testing (in progress)
- ⏳ Multi-robot coordination (planned)
- ⏳ Learning-based control (planned)

---

## 📊 Citation

If you use this project in your research, please cite:

```bibtex
@software{ncat_automated_forging_2026,
  title={NCAT Automated Forging Clay: Autonomous Robotic System for Clay Manipulation},
  author={Tereda, Amanuel Abrdo and Yi, Sun},
  year={2026},
  url={https://github.com/ncat/ncat_automated_forging_clay}
}
```

---

**Last Updated**: 2026  
**Latest Release**: [v0.1.0](https://github.com/ncat/ncat_automated_forging_clay/releases)  
**Next Release**: v0.2.0 (Q2 2024)

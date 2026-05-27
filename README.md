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

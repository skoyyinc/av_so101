# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup and Testing Commands

### Quick Setup and Testing
```bash
# Initial setup (installs dependencies and creates URDF files)
python quick_setup.py

# Full setup and test (includes comprehensive functionality test)
python setup_and_test.py

# Basic functionality test
python test_tracking.py

# Basic functionality test with visual demo
python test_tracking.py demo

# LeRobot integration test
python test_setup.py
```

### Demo Commands
```bash
# Object tracking demo (default PyBullet backend)
python demo_object_tracking.py

# Active vision system demo 
python demo_active_vision.py

# Demo with manual controls
python demo_with_controls.py

# Occlusion handling demo
python demo_occlusion.py
```

### Environment Testing
```bash
# Test backend availability and performance
python src/env_factory.py

# Test specific backend
python -c "from src.env_factory import create_so101_env; env = create_so101_env('mujoco'); print('✅ MuJoCo works')"
```

## Architecture Overview

### Core Components

**Environment Factory System** (`src/env_factory.py`):
- Supports multiple physics backends: PyBullet, MuJoCo, Gazebo
- Auto-selects best available backend (preference: MuJoCo > PyBullet > Gazebo)
- Provides convenience functions for training and evaluation environments
- Fallback mechanism when preferred backend unavailable

**Robot Environments**:
- **`SO101CameraTrackingEnv`** (`src/so_arm_gym_env.py`): PyBullet-based environment with camera-as-end-effector
- **`SO101MuJoCoTrackingEnv`** (`src/so_arm_mujoco_env.py`): MuJoCo-based equivalent for better ML/RL training
- **`SO101GazeboTrackingEnv`** (`src/so_arm_gazebo_env.py`): ROS/Gazebo integration

**Tracking Policies**:
- **`ImprovedTrackingPolicy`** (`src/tracking_policy.py`): Visual servoing with adaptive gains
- **`OcclusionTrackingPolicy`** (`src/occlusion_tracking_policy.py`): Handles partial occlusions and search patterns

**Vision Components**:
- **`ActiveVisionSystem`** (`src/active_vision_system.py`): Main vision coordination system
- **`ActiveCameraController`** (`src/camera_controller.py`): Multi-camera management and switching
- **`SimpleObjectDetector`** (`src/object_detector.py`): Color-based object detection

### Robot Configuration

**SO-ARM101 Specifications**:
- 6 DOF robotic arm with camera-as-end-effector
- Joint control via position/velocity commands
- Two calibration modes available (new/old calibration)
- URDF files in `urdf/` directory with STL mesh assets

**Camera Setup**:
- Resolution: 640x480 (configurable) 
- Mount: End-effector camera for active vision
- Multiple camera support via `ActiveCameraController`

### Key Directories

```
src/                           # Core implementation
├── so_arm_gym_env.py         # PyBullet environment
├── so_arm_mujoco_env.py      # MuJoCo environment  
├── so_arm_gazebo_env.py      # Gazebo environment
├── tracking_policy.py        # Visual servoing policies
├── active_vision_system.py   # Main vision system
├── camera_controller.py      # Camera management
├── object_detector.py        # Object detection
└── env_factory.py           # Environment creation

urdf/                         # Robot models and assets
├── scene.xml                 # MuJoCo scene description
├── so101_new_calib.xml       # New calibration robot model
├── so101_old_calib.xml       # Old calibration robot model
└── assets/                   # STL mesh files
```

## Environment Backends

The system supports three physics backends with automatic fallback:

1. **MuJoCo** (Preferred for ML/RL): Most accurate physics, optimized for training
2. **PyBullet** (General robotics): Good debugging tools, easier visualization
3. **Gazebo** (ROS integration): Best for ROS-based workflows

Use `src/env_factory.py` functions to automatically select the best available backend or specify manually.

## Integration Points

**LeRobot Integration**: 
- SO101Follower class compatibility (`test_setup.py`)
- Standard Gymnasium interface for all environments
- Ready for imitation learning and RL training pipelines

**Active Vision Pipeline**:
1. `ActiveVisionSystem` coordinates overall behavior
2. `ActiveCameraController` manages camera switching/positioning  
3. `SimpleObjectDetector` provides object detection
4. `TrackingPolicy` converts detections to robot actions

## Development Workflow

1. **Setup**: Run `python quick_setup.py` for basic setup
2. **Test**: Run `python test_tracking.py` to verify functionality  
3. **Develop**: Modify policies in `src/tracking_policy.py` or environments in `src/so_arm_*_env.py`
4. **Debug**: Use `python demo_object_tracking.py` for visual debugging
5. **Train**: Use environment factory to create training environments with optimal backend
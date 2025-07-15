import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
import cv2
import time
from typing import Tuple, Dict, Optional
import math
from pathlib import Path

class SO101MuJoCoTrackingEnv(gym.Env):
    """
    SO-ARM101 with camera end-effector for object tracking using MuJoCo
    Alternative to PyBullet version for testing rendering issues
    """
    
    def __init__(self, render_mode="human", camera_width=640, camera_height=480):
        super().__init__()
        
        # Environment parameters
        self.render_mode = render_mode
        self.camera_width = camera_width
        self.camera_height = camera_height
        
        # MuJoCo simulation
        self.model = None
        self.data = None
        self.viewer = None
        
        # Robot parameters - SO-ARM101 specific
        self.num_joints = 6  # SO-ARM101 has 6 DOF
        self.joint_names = ["shoulder_pan", "shoulder_tilt", "elbow_flex", "wrist_1", "wrist_2", "wrist_3"]
        self.joint_ids = []
        
        # Joint limits (from MuJoCo model)
        self.joint_limits = []
        
        # Action space: joint velocities for 6 DOF arm
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_joints,), dtype=np.float32
        )
        
        # Observation space
        self.observation_space = spaces.Dict({
            'camera_image': spaces.Box(
                low=0, high=255, 
                shape=(camera_height, camera_width, 3), 
                dtype=np.uint8
            ),
            'joint_positions': spaces.Box(
                low=-np.pi, high=np.pi, shape=(self.num_joints,), dtype=np.float32
            ),
            'target_in_view': spaces.Box(
                low=0, high=1, shape=(1,), dtype=np.float32
            ),
            'target_center_distance': spaces.Box(
                low=0, high=1, shape=(1,), dtype=np.float32
            )
        })
        
        # Tracking state
        self.target_position = np.array([0.5, 0.0, 0.1])
        self.camera_center = np.array([camera_width//2, camera_height//2])
        self.max_episode_steps = 500
        self.current_step = 0
        
        # Performance tracking
        self.target_centered_steps = 0
        self.total_tracking_error = 0.0
        
        # Previous state for reward calculation
        self.prev_target_in_view = False
        self.prev_joint_positions = None
        
        # MuJoCo model path
        self.model_path = self._get_model_path()
        
    def _get_model_path(self):
        """Get path to MuJoCo model with fallback options"""
        model_dir = Path("urdf")
        
        # Try different model options
        model_options = [
            model_dir / "so101_new_calib.xml",
            model_dir / "so101_old_calib.xml",
            model_dir / "scene.xml"
        ]
        
        for model_path in model_options:
            if model_path.exists():
                print(f"📁 Found MuJoCo model: {model_path}")
                return str(model_path)
        
        print("⚠️  No MuJoCo model found. Please check urdf directory.")
        return None
        
    def _load_model(self):
        """Load MuJoCo model"""
        if self.model_path and Path(self.model_path).exists():
            print(f"📁 Loading MuJoCo model: {self.model_path}")
            
            try:
                self.model = mujoco.MjModel.from_xml_path(self.model_path)
                self.data = mujoco.MjData(self.model)
                
                # Get joint information
                self._setup_joint_info()
                
                print("✅ MuJoCo model loaded successfully")
                return True
                
            except Exception as e:
                print(f"❌ Failed to load MuJoCo model: {e}")
                return False
        else:
            print("❌ MuJoCo model path not found")
            return False
    
    def _setup_joint_info(self):
        """Setup joint information from MuJoCo model"""
        self.joint_ids = []
        self.joint_limits = []
        
        for joint_name in self.joint_names:
            try:
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if joint_id >= 0:
                    self.joint_ids.append(joint_id)
                    
                    # Get joint limits
                    joint_range = self.model.jnt_range[joint_id]
                    self.joint_limits.append((joint_range[0], joint_range[1]))
                else:
                    print(f"⚠️  Joint {joint_name} not found in model")
            except Exception as e:
                print(f"⚠️  Error getting joint {joint_name}: {e}")
        
        print(f"✅ Found {len(self.joint_ids)} joints in MuJoCo model")
        
    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.target_centered_steps = 0
        self.total_tracking_error = 0.0
        
        # Initialize previous state variables
        self.prev_target_in_view = False
        self.prev_joint_positions = None
        
        # Load MuJoCo model
        if not self._load_model():
            return self._get_dummy_observation(), {}
        
        # Reset to initial pose
        self._reset_robot_pose()
        
        # Create target object
        self._create_target_object()
        
        # Step simulation to settle
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def _reset_robot_pose(self):
        """Reset robot to initial pose"""
        if self.model is None or self.data is None:
            return
            
        # Initial joint positions (neutral pose)
        initial_positions = [0.0, -0.3, 0.5, 0.0, 0.2, 0.0]
        
        # Set joint positions
        for i, joint_id in enumerate(self.joint_ids):
            if i < len(initial_positions):
                self.data.qpos[joint_id] = initial_positions[i]
                self.data.qvel[joint_id] = 0.0
        
        # Reset simulation state
        mujoco.mj_forward(self.model, self.data)
        
        print("✅ Robot reset to initial pose")
    
    def _create_target_object(self):
        """Create target object in MuJoCo scene"""
        # For now, we'll use a simple geometric primitive
        # In MuJoCo, objects are typically defined in the XML
        # For dynamic object creation, we'd need to modify the model
        
        # Set target position for tracking
        self.target_position = np.array([
            0.3 + np.random.uniform(-0.2, 0.2),  # X: in front of robot
            np.random.uniform(-0.3, 0.3),        # Y: left/right
            0.1 + np.random.uniform(-0.1, 0.1)   # Z: table height
        ])
        
        print(f"🎯 Target created at position: {self.target_position}")
    
    def step(self, action):
        """Execute one step in the environment"""
        if self.model is None or self.data is None:
            return self._get_dummy_observation(), 0, True, False, {}
        
        self.current_step += 1
        
        # Apply action
        self._apply_action(action)
        
        # Step simulation
        mujoco.mj_step(self.model, self.data)
        
        # Get observation
        observation = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward(observation)
        
        # Check termination conditions
        terminated = self._is_terminated(observation)
        truncated = self.current_step >= self.max_episode_steps
        
        # Get info
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _apply_action(self, action):
        """Apply action to robot joints"""
        if self.model is None or self.data is None:
            return
        
        # Scale action to reasonable velocities
        max_velocity = 3.0  # rad/s
        target_velocities = np.clip(action, -1.0, 1.0) * max_velocity
        
        # Apply velocity control
        for i, joint_id in enumerate(self.joint_ids):
            if i < len(target_velocities):
                self.data.ctrl[joint_id] = target_velocities[i]
    
    def _get_observation(self):
        """Get current observation"""
        if self.model is None or self.data is None:
            return self._get_dummy_observation()
        
        # Get camera image
        camera_image = self._get_camera_image()
        
        # Get joint positions
        joint_positions = np.zeros(self.num_joints)
        for i, joint_id in enumerate(self.joint_ids):
            if i < len(joint_positions):
                joint_positions[i] = self.data.qpos[joint_id]
        
        # Detect target in camera image
        target_info = self._detect_target_in_image(camera_image)
        
        observation = {
            'camera_image': camera_image,
            'joint_positions': joint_positions,
            'target_in_view': np.array([target_info['in_view']], dtype=np.float32),
            'target_center_distance': np.array([target_info['center_distance']], dtype=np.float32)
        }
        
        return observation
    
    def _get_camera_image(self):
        """Get camera image from MuJoCo"""
        if self.model is None or self.data is None:
            return np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)
        
        # Create camera
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        
        # Set camera to end-effector position (if available)
        try:
            # Try to get end-effector position
            end_effector_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "wrist_3")
            if end_effector_id >= 0:
                camera.lookat = self.data.xpos[end_effector_id]
                camera.distance = 0.5
                camera.elevation = -20
                camera.azimuth = 0
        except:
            # Default camera position
            camera.lookat = np.array([0.3, 0.0, 0.1])
            camera.distance = 0.8
            camera.elevation = -20
            camera.azimuth = 0
        
        # Render image
        context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        viewport = mujoco.MjrRect(0, 0, self.camera_width, self.camera_height)
        scene = mujoco.MjvScene(self.model, maxgeom=1000)
        
        # Update scene
        mujoco.mjv_updateScene(self.model, self.data, mujoco.MjvOption(), None, camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
        
        # Render
        mujoco.mjr_render(viewport, scene, context)
        
        # Get image data
        rgb_array = np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(rgb_array, None, viewport, context)
        
        # Flip image (MuJoCo renders upside down)
        rgb_array = np.flipud(rgb_array)
        
        return rgb_array
    
    def _detect_target_in_image(self, image):
        """Detect target object in camera image"""
        target_info = {
            'in_view': 0.0,
            'center_distance': 1.0,
            'pixel_position': None
        }
        
        if image is None:
            return target_info
        
        try:
            # Simple red object detection (similar to color detection)
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # Red color ranges
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 100, 100])
            upper_red2 = np.array([180, 255, 255])
            
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = cv2.bitwise_or(mask1, mask2)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                
                if area > 300:  # Minimum area threshold
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    center_x = x + w // 2
                    center_y = y + h // 2
                    
                    # Calculate distance from image center
                    image_center = np.array([self.camera_width // 2, self.camera_height // 2])
                    target_center = np.array([center_x, center_y])
                    distance = np.linalg.norm(target_center - image_center)
                    
                    # Normalize distance
                    max_distance = np.linalg.norm(image_center)
                    normalized_distance = min(distance / max_distance, 1.0)
                    
                    target_info['in_view'] = 1.0
                    target_info['center_distance'] = normalized_distance
                    target_info['pixel_position'] = (center_x, center_y)
                    
        except Exception as e:
            print(f"⚠️  Target detection failed: {e}")
        
        return target_info
    
    def _calculate_reward(self, observation):
        """Calculate reward (same as PyBullet version)"""
        reward = 0.0
        
        target_in_view = observation['target_in_view'][0]
        center_distance = observation['target_center_distance'][0]
        joint_positions = observation['joint_positions']
        
        # Base reward for having target in view
        if target_in_view > 0.5:
            reward += 3.0
            
            # Smooth, continuous centering reward
            centering_reward = np.exp(-2.0 * center_distance) * 10.0
            reward += centering_reward
            
            # Progressive precision bonuses
            if center_distance < 0.05:
                self.target_centered_steps += 1
                reward += 15.0
            elif center_distance < 0.1:
                reward += 8.0
            elif center_distance < 0.2:
                reward += 4.0
            
            # Bonus for sustained centering
            if self.target_centered_steps > 10:
                reward += 2.0
                
        else:
            # Progressive penalty for losing target
            reward -= 2.0
            if hasattr(self, 'prev_target_in_view') and self.prev_target_in_view:
                reward -= 3.0
        
        # Movement penalties
        if len(joint_positions) > 0:
            extreme_position_penalty = np.sum(np.abs(joint_positions) > 2.0) * 0.5
            reward -= extreme_position_penalty
            
            if hasattr(self, 'prev_joint_positions') and self.prev_joint_positions is not None:
                joint_velocities = np.abs(joint_positions - self.prev_joint_positions)
                velocity_penalty = np.sum(joint_velocities) * 0.02
                reward -= velocity_penalty
        
        # Energy efficiency bonus
        if target_in_view > 0.5 and center_distance < 0.3:
            efficiency_bonus = 1.0 / (1.0 + np.sum(np.abs(joint_positions)) * 0.1)
            reward += efficiency_bonus
        
        # Track error and update previous state
        self.total_tracking_error += center_distance
        self.prev_target_in_view = target_in_view > 0.5
        if len(joint_positions) > 0:
            self.prev_joint_positions = joint_positions.copy()
        
        return reward
    
    def _is_terminated(self, observation):
        """Check if episode should terminate"""
        # Success condition: target well-centered for extended period
        if self.target_centered_steps > 50:  # 5 seconds at 10 Hz
            return True
        
        # Failure condition: robot in extreme position
        joint_positions = observation['joint_positions']
        if len(joint_positions) > 0:
            if np.any(np.abs(joint_positions) > 3.0):
                return True
        
        return False
    
    def _get_info(self):
        """Get environment info"""
        return {
            'target_centered_steps': self.target_centered_steps,
            'total_tracking_error': self.total_tracking_error,
            'current_step': self.current_step,
            'target_centered': self.target_centered_steps > 0
        }
    
    def _get_dummy_observation(self):
        """Get dummy observation when model fails to load"""
        return {
            'camera_image': np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8),
            'joint_positions': np.zeros(self.num_joints, dtype=np.float32),
            'target_in_view': np.array([0.0], dtype=np.float32),
            'target_center_distance': np.array([1.0], dtype=np.float32)
        }
    
    def render(self):
        """Render environment"""
        if self.render_mode == "human" and self.model is not None and self.data is not None:
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            
            # Update viewer
            self.viewer.sync()
            
    def close(self):
        """Close environment"""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
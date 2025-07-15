#!/usr/bin/env python3
"""
ALOHA-based active vision environment for object tracking with occlusion
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import cv2
from typing import Dict, Tuple, Optional
import json
import os

# LeRobot imports
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import create_lerobot_dataset

class ALOHAActiveVisionEnv(gym.Env):
    """
    ALOHA-based environment for active vision with occlusion handling
    Compatible with LeRobot framework
    """
    
    def __init__(self, config_path: str = None, camera_width: int = 640, camera_height: int = 480):
        super().__init__()
        
        # Environment parameters
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.max_episode_steps = 500
        self.current_step = 0
        
        # Load ALOHA configuration
        self.config = self._load_config(config_path)
        
        # Initialize ALOHA environment
        self._setup_aloha_env()
        
        # Define action space (joint velocities for one arm)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32  # 7 DOF for ALOHA arm
        )
        
        # Define observation space (LeRobot compatible)
        self.observation_space = spaces.Dict({
            "observation.images.cam_high": spaces.Box(
                low=0, high=255, 
                shape=(camera_height, camera_width, 3), 
                dtype=np.uint8
            ),
            "observation.state": spaces.Box(
                low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32  # 2 arms × 7 joints
            ),
            "observation.target_info": spaces.Box(
                low=0, high=1, shape=(3,), dtype=np.float32  # in_view, distance, occluded
            )
        })
        
        # Active vision state
        self.target_position = None
        self.occlusion_position = None
        self.target_centered_steps = 0
        self.episode_data = []
        
        print("🤖 ALOHA Active Vision Environment initialized")
        
    def _load_config(self, config_path: str) -> Dict:
        """Load ALOHA environment configuration"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Default configuration for active vision
        return {
            "env_name": "aloha_active_vision",
            "robot_name": "aloha",
            "control_mode": "joint_velocity",
            "camera_names": ["cam_high", "cam_low"],
            "target_object": {
                "type": "sphere",
                "radius": 0.04,
                "color": [1, 0, 0, 1]  # Red
            },
            "occlusion_object": {
                "type": "box",
                "size": [0.015, 0.015, 0.05],
                "color": [0.5, 0.5, 0.5, 1]  # Gray
            },
            "workspace": {
                "x_range": [0.3, 0.7],
                "y_range": [-0.3, 0.3],
                "z_range": [0.0, 0.2]
            }
        }
    
    def _setup_aloha_env(self):
        """Setup ALOHA MuJoCo environment"""
        try:
            # Import ALOHA environment
            from gym_hil.envs.aloha import ALOHAEnv
            
            # Create ALOHA environment
            self.aloha_env = ALOHAEnv(
                task_name="active_vision",
                control_mode=self.config["control_mode"],
                camera_names=self.config["camera_names"],
                render_mode="rgb_array"
            )
            
            print("✅ ALOHA environment created successfully")
            
        except ImportError as e:
            print(f"❌ Failed to import ALOHA environment: {e}")
            print("Please install: pip install gym-hil")
            raise
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset environment for new episode"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.target_centered_steps = 0
        self.episode_data = []
        
        # Reset ALOHA environment
        aloha_obs = self.aloha_env.reset()
        
        # Create target and occlusion objects
        self._create_objects()
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def _create_objects(self):
        """Create target and occlusion objects in ALOHA scene"""
        # Generate random positions within workspace
        workspace = self.config["workspace"]
        
        # Target object position
        target_x = np.random.uniform(workspace["x_range"][0], workspace["x_range"][1])
        target_y = np.random.uniform(workspace["y_range"][0], workspace["y_range"][1])
        target_z = 0.02  # On table surface
        self.target_position = np.array([target_x, target_y, target_z])
        
        # Occlusion object position (25% coverage)
        occlusion_x = target_x * 0.85  # Between camera and target
        occlusion_y = target_y + np.random.choice([-0.03, 0.03])  # Side offset
        occlusion_z = 0.05  # Slightly higher
        self.occlusion_position = np.array([occlusion_x, occlusion_y, occlusion_z])
        
        # Add objects to ALOHA scene (implementation depends on gym_hil API)
        try:
            self.aloha_env.add_object(
                name="target",
                obj_type="sphere",
                pos=self.target_position,
                size=self.config["target_object"]["radius"],
                rgba=self.config["target_object"]["color"]
            )
            
            self.aloha_env.add_object(
                name="occlusion",
                obj_type="box",
                pos=self.occlusion_position,
                size=self.config["occlusion_object"]["size"],
                rgba=self.config["occlusion_object"]["color"]
            )
            
            print(f"🎯 Objects created - Target: {self.target_position}, Occlusion: {self.occlusion_position}")
            
        except Exception as e:
            print(f"⚠️ Object creation failed: {e}")
            # Fallback to manual object placement if gym_hil doesn't support dynamic objects
    
    def step(self, action: np.ndarray):
        """Execute one step in the environment"""
        self.current_step += 1
        
        # Convert action to ALOHA format (use only one arm for active vision)
        aloha_action = np.zeros(14)  # 2 arms × 7 joints
        aloha_action[:7] = action  # Use first arm for camera control
        
        # Execute action in ALOHA environment
        aloha_obs, aloha_reward, aloha_done, aloha_info = self.aloha_env.step(aloha_action)
        
        # Get our observation
        observation = self._get_observation()
        
        # Calculate reward for active vision task
        reward = self._calculate_reward(observation)
        
        # Check termination
        terminated = self._is_terminated(observation)
        truncated = self.current_step >= self.max_episode_steps
        
        info = self._get_info()
        
        # Store episode data for LeRobot dataset
        self.episode_data.append({
            "observation": observation,
            "action": action,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated
        })
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> Dict:
        """Get current observation in LeRobot format"""
        # Get camera images from ALOHA
        images = self.aloha_env.get_images()
        camera_image = images.get("cam_high", np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8))
        
        # Get robot state (joint positions)
        robot_state = self.aloha_env.get_state()
        
        # Get target information
        target_info = self._get_target_info(camera_image)
        
        # Format observation for LeRobot
        observation = {
            "observation.images.cam_high": camera_image,
            "observation.state": robot_state,
            "observation.target_info": np.array([
                target_info["in_view"],
                target_info["center_distance"],
                target_info["occluded"]
            ], dtype=np.float32)
        }
        
        return observation
    
    def _get_target_info(self, camera_image: np.ndarray) -> Dict:
        """Detect target in camera image"""
        target_info = {
            "in_view": 0.0,
            "center_distance": 1.0,
            "occluded": 0.0
        }
        
        if camera_image is None or camera_image.size == 0:
            return target_info
        
        try:
            # Convert to HSV for color detection
            hsv = cv2.cvtColor(camera_image, cv2.COLOR_RGB2HSV)
            
            # Red color detection
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
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                
                if area > 50:  # Minimum area threshold
                    # Get center
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        target_info["in_view"] = 1.0
                        
                        # Calculate distance from center
                        center_x, center_y = self.camera_width // 2, self.camera_height // 2
                        distance = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
                        max_distance = np.sqrt(center_x**2 + center_y**2)
                        target_info["center_distance"] = min(distance / max_distance, 1.0)
                        
                        # Estimate occlusion based on area
                        expected_area = 400  # Expected area when not occluded
                        occlusion_ratio = max(0, 1.0 - (area / expected_area))
                        target_info["occluded"] = min(occlusion_ratio, 1.0)
                        
        except Exception as e:
            print(f"⚠️ Target detection failed: {e}")
        
        return target_info
    
    def _calculate_reward(self, observation: Dict) -> float:
        """Calculate reward for active vision task"""
        target_info = observation["observation.target_info"]
        target_in_view = target_info[0]
        center_distance = target_info[1]
        target_occluded = target_info[2]
        
        reward = 0.0
        
        # Base reward for target in view
        if target_in_view > 0.5:
            reward += 3.0
            
            # Penalty for occlusion
            reward -= target_occluded * 5.0
            
            # Bonus for clear view
            if target_occluded < 0.3:
                reward += 5.0
            
            # Centering reward
            centering_reward = np.exp(-2.0 * center_distance) * 10.0
            reward += centering_reward
            
            # Precision bonus
            if center_distance < 0.08:
                self.target_centered_steps += 1
                reward += 15.0
            else:
                self.target_centered_steps = 0
        else:
            reward -= 2.0
        
        return reward
    
    def _is_terminated(self, observation: Dict) -> bool:
        """Check if episode should terminate"""
        return self.target_centered_steps > 100
    
    def _get_info(self) -> Dict:
        """Get additional info"""
        return {
            "target_centered_steps": self.target_centered_steps,
            "current_step": self.current_step,
            "target_position": self.target_position.tolist() if self.target_position is not None else None,
            "occlusion_position": self.occlusion_position.tolist() if self.occlusion_position is not None else None
        }
    
    def render(self, mode: str = "rgb_array"):
        """Render environment"""
        return self.aloha_env.render(mode=mode)
    
    def close(self):
        """Close environment"""
        if hasattr(self, 'aloha_env'):
            self.aloha_env.close()
    
    def save_episode_data(self, dataset_name: str = "aloha_active_vision"):
        """Save episode data in LeRobot format"""
        if not self.episode_data:
            return
        
        # Create LeRobot dataset
        dataset = create_lerobot_dataset(
            dataset_name,
            self.episode_data,
            episode_data_index={"from": 0, "to": len(self.episode_data)}
        )
        
        print(f"📁 Episode data saved to LeRobot dataset: {dataset_name}")
        return dataset
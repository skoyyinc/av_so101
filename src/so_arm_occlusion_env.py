import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import cv2
import time
from typing import Tuple, Dict, Optional
import math
from pathlib import Path

class SO101OcclusionTrackingEnv(gym.Env):
    """
    SO-ARM101 with camera end-effector for object tracking with occlusion
    Goal: Move camera to center target object in view despite occlusion
    """
    
    def __init__(self, render_mode="human", camera_width=640, camera_height=480):
        super().__init__()
        
        # Environment parameters
        self.render_mode = render_mode
        self.camera_width = camera_width
        self.camera_height = camera_height
        
        # Physics simulation
        self.physics_client = None
        self.robot_id = None
        self.target_object_id = None
        self.occlusion_object_id = None
        self.plane_id = None
        
        # Robot parameters - SO-ARM101 specific
        self.num_joints = 6  # SO-ARM101 has 6 DOF
        self.joint_indices = []
        self.joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
        
        # Joint limits (from SO-ARM101 specs) - Constrained to prevent looking at sky
        self.joint_limits = [
            (-np.pi, np.pi),      # Base rotation (full range)
            (-np.pi/2, np.pi/6),  # Shoulder (limited upward movement to prevent sky look)
            (-np.pi/2, np.pi/2),  # Elbow  
            (-np.pi/2, np.pi/6),  # Wrist 1 (limited upward movement)
            (-np.pi/2, np.pi/2),  # Wrist 2
            (-np.pi, np.pi)       # Wrist rotation
        ]
        
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
            ),
            'target_occluded': spaces.Box(
                low=0, high=1, shape=(1,), dtype=np.float32
            )
        })
        
        # Tracking state
        self.target_position = np.array([0.5, 0.0, 0.1])
        self.occlusion_position = np.array([0.7, 0.0, 0.1])
        self.camera_center = np.array([camera_width//2, camera_height//2])
        self.max_episode_steps = 500
        self.current_step = 0
        
        # Performance tracking
        self.target_centered_steps = 0
        self.total_tracking_error = 0.0
        
        # URDF path
        self.urdf_path = self._get_urdf_path()
        
    def _get_urdf_path(self):
        """Get path to SO-ARM101 URDF with fallback options"""
        urdf_dir = Path("urdf")
        
        # Try different URDF options in order of preference
        urdf_options = [
            urdf_dir / "so_arm101_camera.urdf",  # With meshes
            urdf_dir / "so_arm101_simple.urdf",  # Simple geometry
            urdf_dir / "so101_new_calib.urdf",   # If you have the original
        ]
        
        for urdf_path in urdf_options:
            if urdf_path.exists():
                print(f"📁 Found URDF: {urdf_path}")
                return str(urdf_path)
        
        print("⚠️  No SO-ARM101 URDF found. Please run setup first.")
        return None

    def _create_so101_robot(self):
        """Create SO-ARM101 robot model with better error handling"""
        if self.urdf_path and Path(self.urdf_path).exists():
            print(f"📁 Loading SO-ARM101 URDF: {self.urdf_path}")
            
            # Set search path for mesh files
            urdf_dir = Path(self.urdf_path).parent
            p.setAdditionalSearchPath(str(urdf_dir))
            
            base_position = [0, 0, 0]
            base_orientation = p.getQuaternionFromEuler([0, 0, 0])
            
            try:
                self.robot_id = p.loadURDF(
                    self.urdf_path, 
                    base_position, 
                    base_orientation,
                    useFixedBase=True,
                    flags=p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_MATERIAL_COLORS_FROM_MTL
                )
                print("✅ SO-ARM101 URDF loaded successfully")
                
                # Set better visual properties
                self._improve_visual_properties()
                
            except Exception as e:
                print(f"❌ Failed to load URDF: {e}")
                print("🔄 Trying fallback robot...")
                self.robot_id = self._create_fallback_robot()
        else:
            print("🔧 Using fallback robot model")
            self.robot_id = self._create_fallback_robot()
            
        # Setup joint control
        self._setup_joint_control()
        
    def _improve_visual_properties(self):
        """Improve visual properties of the robot"""
        if self.robot_id is None:
            return
            
        # Get number of links
        num_links = p.getNumJoints(self.robot_id)
        
        # Set better colors and materials for base link
        p.changeVisualShape(self.robot_id, -1, rgbaColor=[0.8, 0.8, 0.8, 1.0])
        
        # Set colors for all links
        for i in range(num_links):
            link_info = p.getJointInfo(self.robot_id, i)
            link_name = link_info[12].decode('utf-8') if link_info[12] else f"link_{i}"
            
            # Different colors for different parts
            if "base" in link_name.lower():
                color = [0.2, 0.2, 0.2, 1.0]  # Dark gray for base
            elif "shoulder" in link_name.lower() or "elbow" in link_name.lower():
                color = [0.7, 0.3, 0.1, 1.0]  # Orange for main arm
            elif "wrist" in link_name.lower():
                color = [0.3, 0.3, 0.7, 1.0]  # Blue for wrist
            else:
                color = [0.5, 0.5, 0.5, 1.0]  # Gray for other parts
                
            p.changeVisualShape(self.robot_id, i, rgbaColor=color)

    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.target_centered_steps = 0
        self.total_tracking_error = 0.0
        
        # Initialize previous state variables for reward calculation
        self.prev_target_in_view = False
        self.prev_joint_positions = None
        
        # Initialize PyBullet
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            
        if self.render_mode == "human":
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1./240.)  # 240 Hz for stable simulation
        
        # Create environment
        self._create_environment()
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def _create_environment(self):
        """Create the simulation environment"""
        # Load ground plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Create SO-ARM101 robot
        self._create_so101_robot()
        
        # Create target object
        self._create_target_object()
        
        # Create occlusion object
        self._create_occlusion_object()
        
        # Set initial robot configuration
        self._reset_robot_pose()
        
    def _create_fallback_robot(self):
        """Create fallback robot if URDF fails"""
        # Use a simple 6-DOF arm
        try:
            robot_id = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0], useFixedBase=True)
            print("✅ Using KUKA IIWA as fallback")
            return robot_id
        except:
            print("❌ No suitable robot URDF found")
            return None
            
    def _setup_joint_control(self):
        """Setup joint control for the robot"""
        if self.robot_id is None:
            return
            
        # Get all joints
        num_joints = p.getNumJoints(self.robot_id)
        print(f"🔧 Robot has {num_joints} total joints")
        
        self.joint_indices = []
        joint_info = []
        
        for i in range(num_joints):
            info = p.getJointInfo(self.robot_id, i)
            joint_name = info[1].decode('utf-8')
            joint_type = info[2]
            
            # Only use revolute and prismatic joints
            if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                self.joint_indices.append(i)
                joint_info.append((i, joint_name, joint_type))
                print(f"  Joint {i}: {joint_name} (type: {joint_type})")
        
        # Limit to 6 controllable joints for SO-ARM101
        self.joint_indices = self.joint_indices[:6]
        self.num_joints = len(self.joint_indices)
        
        print(f"✅ Using {self.num_joints} controllable joints: {self.joint_indices}")
        
        # Enable force/torque sensors for all joints
        for joint_idx in self.joint_indices:
            p.enableJointForceTorqueSensor(self.robot_id, joint_idx, 1)
        
    def _create_target_object(self):
        """Create target object to track"""
        # Create a smaller bright red sphere
        object_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.04)
        object_visual = p.createVisualShape(
            p.GEOM_SPHERE, 
            radius=0.04,
            rgbaColor=[1, 0, 0, 1]  # Bright red
        )
        
        # Place target object closer to robot on ground level
        x = np.random.uniform(0.4, 0.6)    # Closer forward distance
        y = np.random.uniform(-0.15, 0.15)  # Smaller left/right variation
        z = 0.02  # Lower height on ground level
        self.target_position = np.array([x, y, z])
        
        self.target_object_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=object_shape,
            baseVisualShapeIndex=object_visual,
            basePosition=self.target_position
        )
        
        print(f"🎯 Target object created at {self.target_position}")
        
    def _create_occlusion_object(self):
        """Create minimal occlusion object"""
        # Create a smaller gray box as occlusion (smaller than target)
        box_size = [0.015, 0.015, 0.05]  # Smaller than target sphere (radius 0.04)
        object_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=box_size)
        object_visual = p.createVisualShape(
            p.GEOM_BOX, 
            halfExtents=box_size,
            rgbaColor=[0.5, 0.5, 0.5, 1]  # Gray color
        )
        
        # Place occlusion object between robot and target (25% occlusion)
        target_x = self.target_position[0]
        target_y = self.target_position[1]
        
        # Position occlusion to cover only 25% of target
        occlusion_x = target_x * 0.85  # 85% of the way to target (closer to camera)
        # Offset to cover only edge of target (25% coverage)
        occlusion_y = target_y + np.random.choice([-0.03, 0.03])  # Either left or right edge
        occlusion_z = 0.05  # Same height as target for partial side coverage
        
        self.occlusion_position = np.array([occlusion_x, occlusion_y, occlusion_z])
        
        self.occlusion_object_id = p.createMultiBody(
            baseMass=0.0,  # Static object
            baseCollisionShapeIndex=object_shape,
            baseVisualShapeIndex=object_visual,
            basePosition=self.occlusion_position
        )
        
        print(f"🚧 Occlusion object created at {self.occlusion_position}")
        
    def _reset_robot_pose(self):
        """Reset robot to initial pose"""
        if self.robot_id is None or not self.joint_indices:
            return
            
        # Initial joint positions for SO-ARM101 - adjusted to point camera toward ground within limits
        initial_positions = [0, 0.2, 0.8, -0.3, 0.2, 0]  # Shoulder and wrist1 within sky-prevention limits
        
        # Apply initial positions
        for i, joint_idx in enumerate(self.joint_indices):
            if i < len(initial_positions):
                p.resetJointState(self.robot_id, joint_idx, initial_positions[i])
                
        # Let physics settle
        for _ in range(100):
            p.stepSimulation()
            
        print("✅ Robot reset to initial pose")
        
    def step(self, action):
        """Execute one step in the environment"""
        if self.robot_id is None:
            # Return dummy observation if robot failed to load
            return self._get_dummy_observation(), 0, True, False, {}
            
        self.current_step += 1
        
        # Apply action
        self._apply_action(action)
        
        # Step simulation multiple times for stability
        for _ in range(5):
            p.stepSimulation()
        
        # Get observation
        observation = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward(observation)
        
        # Check termination
        terminated = self._is_terminated(observation)
        truncated = self.current_step >= self.max_episode_steps
        
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
        
    def _apply_action(self, action):
        """Apply action to robot joints with proper control and sky-prevention constraints"""
        if self.robot_id is None or not self.joint_indices:
            return
            
        # Scale action to reasonable velocities - Increased for better responsiveness
        max_velocity = 3.0  # rad/s (increased from 1.5)
        target_velocities = np.clip(action, -1.0, 1.0) * max_velocity
        
        # Get current joint positions to check limits
        current_positions = self._get_joint_positions()
        
        # Apply velocity control to each joint with constraint checking
        for i, joint_idx in enumerate(self.joint_indices):
            if i < len(target_velocities) and i < len(current_positions):
                # Disable camera joint (joint 5) - keep it fixed
                if i == 5:  # Camera joint should not move
                    target_vel = 0
                else:
                    # Check joint limits to prevent looking at sky
                    current_pos = current_positions[i]
                    target_vel = target_velocities[i]
                    
                    # Constrain movement based on joint limits
                    if i < len(self.joint_limits):
                        min_limit, max_limit = self.joint_limits[i]
                        
                        # If approaching upper limit and velocity is positive, reduce/stop
                        if current_pos > max_limit - 0.1 and target_vel > 0:
                            target_vel = 0  # Stop upward movement
                            
                        # If approaching lower limit and velocity is negative, reduce/stop
                        elif current_pos < min_limit + 0.1 and target_vel < 0:
                            target_vel = 0  # Stop downward movement
                
                p.setJointMotorControl2(
                    bodyUniqueId=self.robot_id,
                    jointIndex=joint_idx,
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocity=target_vel,
                    maxVelocity=max_velocity,
                    force=200  # Maximum force/torque (increased)
                )
                
    def _get_observation(self):
        """Get current observation"""
        if self.robot_id is None:
            return self._get_dummy_observation()
            
        # Get camera image
        camera_image = self._get_camera_image()
        
        # Get joint positions
        joint_positions = self._get_joint_positions()
        
        # Get target info
        target_info = self._get_target_info(camera_image)
        
        observation = {
            'camera_image': camera_image,
            'joint_positions': joint_positions,
            'target_in_view': np.array([target_info['in_view']], dtype=np.float32),
            'target_center_distance': np.array([target_info['center_distance']], dtype=np.float32),
            'target_occluded': np.array([target_info['occluded']], dtype=np.float32)
        }
        
        return observation
        
    def _get_dummy_observation(self):
        """Get dummy observation if robot failed to load"""
        return {
            'camera_image': np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8),
            'joint_positions': np.zeros(6, dtype=np.float32),
            'target_in_view': np.array([0.0], dtype=np.float32),
            'target_center_distance': np.array([1.0], dtype=np.float32),
            'target_occluded': np.array([1.0], dtype=np.float32)
        }
        
    def _get_camera_image(self):
        """Get camera image from end-effector"""
        if self.robot_id is None:
            return np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)
            
        # Get camera link state (last link or specific camera link)
        camera_link_idx = -1  # Use last link as camera
        
        # Try to find camera_link specifically
        num_joints = p.getNumJoints(self.robot_id)
        for i in range(num_joints):
            link_info = p.getJointInfo(self.robot_id, i)
            link_name = link_info[12].decode('utf-8')  # Link name
            if 'camera' in link_name.lower():
                camera_link_idx = i
                break
        
        # Get link state
        if camera_link_idx >= 0:
            link_state = p.getLinkState(self.robot_id, camera_link_idx)
        else:
            # Use end-effector (last joint)
            if self.joint_indices:
                link_state = p.getLinkState(self.robot_id, self.joint_indices[-1])
            else:
                # Fallback to base
                link_state = p.getBasePositionAndOrientation(self.robot_id)
                link_state = (link_state[0], link_state[1])
        
        camera_pos = link_state[0]
        camera_orn = link_state[1]
        
        # Get the end-effector orientation and compute forward direction
        rotation_matrix = p.getMatrixFromQuaternion(camera_orn)
        
        # Camera forward direction (along -X axis in end-effector frame, was backwards)
        forward_direction = [
            -rotation_matrix[0],  # -X component of X axis (reversed)
            -rotation_matrix[3],  # -Y component of X axis (reversed)
            -rotation_matrix[6]   # -Z component of X axis (reversed)
        ]
        
        # Position camera for optimal view (now that forward direction is correct)
        camera_offset = [0.1, 0.0, 0.15]  # Forward and up for good view
        camera_pos = [
            camera_pos[0] + camera_offset[0],
            camera_pos[1] + camera_offset[1],
            camera_pos[2] + camera_offset[2]
        ]
        
        # Create moderate downward look direction
        # Mix forward direction with downward component
        look_distance = 1.0
        downward_angle = 0.70  # Roughly 40 degrees in radians (increased from 30 degrees)
        
        # Forward and downward direction
        look_direction = [
            forward_direction[0] * np.cos(downward_angle),
            forward_direction[1] * np.cos(downward_angle),
            forward_direction[2] * np.cos(downward_angle) - np.sin(downward_angle)
        ]
        
        target_look_pos = [
            camera_pos[0] + look_direction[0] * look_distance,
            camera_pos[1] + look_direction[1] * look_distance,
            camera_pos[2] + look_direction[2] * look_distance
        ]
        
        # Use world Z-axis as up vector for stable orientation
        up_vector = [0, 0, 1]  # World up (Z-axis)
        
        # Create view matrix with simpler setup
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_pos,
            cameraTargetPosition=target_look_pos,
            cameraUpVector=up_vector
        )

        # Create projection matrix with wider FOV for better visibility
        proj_matrix = p.computeProjectionMatrixFOV(
           fov=75,  # Wider field of view
           aspect=self.camera_width / self.camera_height,
           nearVal=0.1,
           farVal=5.0  # Increased range for distant objects
       )
       
       # Render camera image
        try:
           (_, _, px, _, _) = p.getCameraImage(
               width=self.camera_width,
               height=self.camera_height,
               viewMatrix=view_matrix,
               projectionMatrix=proj_matrix,
               renderer=p.ER_BULLET_HARDWARE_OPENGL
           )
           
           # Convert to RGB 
           rgb_array = np.array(px, dtype=np.uint8)
           rgb_array = rgb_array[:, :, :3]  # Remove alpha channel
           
           # No flipping needed with correct up vector
           
        except Exception as e:
           print(f"⚠️  Camera rendering failed: {e}")
           # Return black image as fallback
           rgb_array = np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)
           
        return rgb_array
       
    def _get_joint_positions(self):
        """Get current joint positions"""
        if self.robot_id is None or not self.joint_indices:
            return np.zeros(self.num_joints, dtype=np.float32)
            
        joint_positions = []
        for joint_idx in self.joint_indices:
            joint_state = p.getJointState(self.robot_id, joint_idx)
            joint_positions.append(joint_state[0])  # Position
            
        # Pad with zeros if needed
        while len(joint_positions) < self.num_joints:
            joint_positions.append(0.0)
            
        return np.array(joint_positions[:self.num_joints], dtype=np.float32)
        
    def _get_target_info(self, camera_image):
        """Detect target in camera image and check for occlusion"""
        target_info = {
            'in_view': 0.0,
            'center_distance': 1.0,
            'pixel_position': None,
            'occluded': 0.0
        }
        
        if camera_image is None or camera_image.size == 0:
            return target_info
            
        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(camera_image, cv2.COLOR_RGB2HSV)
            
            # Red color detection (wider range for better detection)
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 100, 100])
            upper_red2 = np.array([180, 255, 255])
            
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = cv2.bitwise_or(mask1, mask2)
            
            # Morphological operations to clean up mask
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                
                if area > 100:  # Minimum area threshold
                    # Get center of mass
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        target_info['in_view'] = 1.0
                        target_info['pixel_position'] = (cx, cy)
                        
                        # Calculate distance from center
                        center_x, center_y = self.camera_center
                        distance = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
                        max_distance = np.sqrt(center_x**2 + center_y**2)
                        target_info['center_distance'] = min(distance / max_distance, 1.0)
                        
                        # Check for occlusion based on target area (smaller area = more occluded)
                        expected_area = 400  # Expected area when not occluded (25% occlusion setup)
                        occlusion_ratio = max(0, 1.0 - (area / expected_area))
                        target_info['occluded'] = min(occlusion_ratio, 1.0)
                        
        except Exception as e:
            print(f"⚠️  Target detection failed: {e}")
            
        return target_info
        
    def _calculate_reward(self, observation):
        """Calculate improved reward for current state with occlusion awareness"""
        reward = 0.0
        
        target_in_view = observation['target_in_view'][0]
        center_distance = observation['target_center_distance'][0]
        target_occluded = observation['target_occluded'][0]
        joint_positions = observation['joint_positions']
        
        # Base reward for having target in view
        if target_in_view > 0.5:
            reward += 3.0
            
            # Penalty for occlusion - encourage finding unoccluded view
            occlusion_penalty = target_occluded * 5.0
            reward -= occlusion_penalty
            
            # Bonus for unoccluded view
            if target_occluded < 0.3:
                reward += 5.0  # Clear view bonus
            
            # Smooth, continuous centering reward (exponential decay)
            centering_reward = np.exp(-2.0 * center_distance) * 10.0
            reward += centering_reward
            
            # Progressive precision bonuses
            if center_distance < 0.05:  # Very precise centering
                self.target_centered_steps += 1
                reward += 15.0
            elif center_distance < 0.1:  # Good centering
                reward += 8.0
            elif center_distance < 0.2:  # Moderate centering
                reward += 4.0
            
            # Bonus for sustained centering
            if self.target_centered_steps > 10:
                reward += 2.0  # Stability bonus
                
        else:
            # Progressive penalty for losing target
            reward -= 2.0
            # Additional penalty if target was recently visible
            if hasattr(self, 'prev_target_in_view') and self.prev_target_in_view:
                reward -= 3.0
            
        # Improved movement penalty (velocity-based)
        if len(joint_positions) > 0:
            # Penalty for extreme joint positions
            extreme_position_penalty = np.sum(np.abs(joint_positions) > 2.0) * 0.5
            reward -= extreme_position_penalty
            
        # Energy efficiency bonus (prefer smaller movements)
        if target_in_view > 0.5 and center_distance < 0.3:
            efficiency_bonus = 1.0 / (1.0 + np.sum(np.abs(joint_positions)) * 0.1)
            reward += efficiency_bonus
            
        # Track total error
        self.total_tracking_error += center_distance
        
        # Store previous state for next iteration
        self.prev_target_in_view = target_in_view > 0.5
        if len(joint_positions) > 0:
            self.prev_joint_positions = joint_positions.copy()
        
        return reward
        
    def _is_terminated(self, observation):
        """Check if episode should terminate"""
        # Success condition: target well-centered for extended period
        if self.target_centered_steps > 100:  # 10 seconds at 10 Hz
            return True
            
        return False
        
    def _get_info(self):
        """Get additional info"""
        return {
            'target_centered_steps': self.target_centered_steps,
            'average_tracking_error': self.total_tracking_error / max(self.current_step, 1),
            'current_step': self.current_step,
            'robot_loaded': self.robot_id is not None
        }
        
    def render(self):
        """Render environment"""
        if self.render_mode == "human":
            pass  # PyBullet GUI handles rendering
        return None
        
    def close(self):
        """Close environment"""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import cv2
from typing import Tuple, Dict, List, Optional
from pathlib import Path

class SO101IsolatedMultiRobotEnv(gym.Env):
    """
    Multi-robot SO101 environment with isolated workspaces
    Each robot has walls to prevent seeing other robots/targets
    """
    
    def __init__(self, n_robots: int = 4, render_mode="human", 
                 camera_width=128, camera_height=128):
        super().__init__()
        
        self.n_robots = n_robots
        self.render_mode = render_mode
        self.camera_width = camera_width
        self.camera_height = camera_height
        
        # Robot storage
        self.robot_ids = []
        self.target_ids = []
        self.wall_ids = []  # Wall IDs for isolation
        self.robot_joint_indices = []
        
        # Workspace parameters
        self.workspace_size = 2.0  # Size of each isolated workspace (increased)
        self.wall_height = 0.8
        self.wall_thickness = 0.05
        
        # Define action/observation spaces
        single_robot_action_dim = 4  # 4 DOF per robot
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, 
            shape=(n_robots * single_robot_action_dim,), 
            dtype=np.float32
        )
        
        # Each robot gets: camera_image + joint_positions + target_info
        self.observation_space = spaces.Dict({
            f'robot_{i}': spaces.Dict({
                'camera_image': spaces.Box(
                    low=0, high=255,
                    shape=(camera_height, camera_width, 3),
                    dtype=np.uint8
                ),
                'joint_positions': spaces.Box(
                    low=-np.pi, high=np.pi, shape=(4,), dtype=np.float32
                ),
                'target_in_view': spaces.Box(
                    low=0, high=1, shape=(1,), dtype=np.float32
                ),
                'target_center_distance': spaces.Box(
                    low=0, high=1, shape=(1,), dtype=np.float32
                )
            }) for i in range(n_robots)
        })
        
        # Episode tracking per robot
        self.episode_steps = [0] * n_robots
        self.max_episode_steps = 200
        
        # Physics setup
        self.physics_client = None
        self._setup_physics()
        
    def _setup_physics(self):
        """Initialize PyBullet physics with GPU acceleration"""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            
        if self.render_mode == "human":
            # Force NVIDIA GPU usage
            self.physics_client = p.connect(p.GUI, options="--opengl2 --gpu_id=0")
        else:
            self.physics_client = p.connect(p.DIRECT, options="--gpu_id=0")
            
        # Enable GPU acceleration
        try:
            p.configureDebugVisualizer(p.COV_ENABLE_GPU_RENDER_SEGMENTATION_MASK_PLUGIN, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
            print("✅ GPU rendering enabled")
        except:
            print("⚠️  GPU rendering not available, using CPU")
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1./240.)
        
        # Optimize physics for multiple robots
        p.setPhysicsEngineParameter(numSolverIterations=10)
        p.setPhysicsEngineParameter(numSubSteps=1)
        
        # Better camera for viewing all robots
        if self.render_mode == "human":
            p.resetDebugVisualizerCamera(
                cameraDistance=3.0,
                cameraYaw=45,
                cameraPitch=-30,
                cameraTargetPosition=[0, 0, 0]
            )
        
    def _create_environment(self):
        """Create multi-robot environment with isolation walls"""
        # Load ground plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Calculate robot positions and create isolation cells
        positions = self._calculate_robot_positions()
        
        print(f"🤖 Creating {self.n_robots} robots with isolated workspaces...")
        
        for i in range(self.n_robots):
            robot_pos = positions[i]
            print(f"  Creating robot {i} at position {robot_pos}...")
            
            try:
                # Create isolation walls for this robot
                self._create_isolation_walls(robot_pos, i)
                print(f"    ✅ Walls created for robot {i}")
                
                # Create robot
                robot_id = self._create_robot(robot_pos, i)
                if robot_id is None:
                    raise RuntimeError(f"Failed to create robot {i}")
                self.robot_ids.append(robot_id)
                print(f"    ✅ Robot {i} created with ID {robot_id}")
                
                # Setup joints for this robot
                joint_indices = self._setup_robot_joints(robot_id)
                self.robot_joint_indices.append(joint_indices)
                print(f"    ✅ Robot {i} joints: {joint_indices}")
                
                # Create target within the isolated workspace
                target_pos = self._get_target_position(robot_pos, i)
                target_id = self._create_target(target_pos, i)
                if target_id is None:
                    raise RuntimeError(f"Failed to create target for robot {i}")
                self.target_ids.append(target_id)
                print(f"    ✅ Target {i} created at {target_pos} with ID {target_id}")
                
            except Exception as e:
                print(f"    ❌ Error creating robot {i}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"✅ Created {len(self.robot_ids)} isolated robot workspaces")
        
    def _calculate_robot_positions(self) -> List[List[float]]:
        """Calculate robot base positions in 2x2 grid"""
        positions = []
        spacing = 2.5  # Distance between robot centers (increased for larger workspaces)
        
        if self.n_robots <= 4:
            # 2x2 grid layout
            for i in range(self.n_robots):
                x = -spacing/2 + (i % 2) * spacing
                y = -spacing/2 + (i // 2) * spacing
                positions.append([x, y, 0])
        else:
            # Larger grid for more robots
            grid_size = int(np.ceil(np.sqrt(self.n_robots)))
            for i in range(self.n_robots):
                x = -grid_size * spacing/2 + (i % grid_size) * spacing
                y = -grid_size * spacing/2 + (i // grid_size) * spacing
                positions.append([x, y, 0])
                
        return positions
        
    def _create_isolation_walls(self, robot_pos: List[float], robot_index: int):
        """Create walls around each robot's workspace"""
        x_center, y_center, z_center = robot_pos
        half_size = self.workspace_size / 2
        
        # Wall positions (4 walls around the workspace)
        wall_configs = [
            # [position, orientation, name]
            [[x_center + half_size, y_center, self.wall_height/2], [0, 0, 0, 1], "east"],
            [[x_center - half_size, y_center, self.wall_height/2], [0, 0, 0, 1], "west"],
            [[x_center, y_center + half_size, self.wall_height/2], [0, 0, 0.707, 0.707], "north"],
            [[x_center, y_center - half_size, self.wall_height/2], [0, 0, 0.707, 0.707], "south"],
        ]
        
        robot_walls = []
        
        for pos, orn, name in wall_configs:
            # Create wall visual
            visual_shape = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[self.wall_thickness/2, self.workspace_size/2, self.wall_height/2],
                rgbaColor=[0.7, 0.7, 0.7, 0.8]  # Gray, semi-transparent
            )
            
            # Create wall collision
            collision_shape = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[self.wall_thickness/2, self.workspace_size/2, self.wall_height/2]
            )
            
            wall_id = p.createMultiBody(
                baseMass=0,  # Static wall
                baseCollisionShapeIndex=collision_shape,
                baseVisualShapeIndex=visual_shape,
                basePosition=pos,
                baseOrientation=orn
            )
            
            robot_walls.append(wall_id)
            
        self.wall_ids.append(robot_walls)
        
        # Add floor marking for this workspace
        self._add_floor_marking(robot_pos, robot_index)
        
    def _add_floor_marking(self, robot_pos: List[float], robot_index: int):
        """Add colored floor marking to identify each robot's workspace"""
        colors = [
            [1, 0.8, 0.8, 0.3],  # Light red
            [0.8, 1, 0.8, 0.3],  # Light green
            [0.8, 0.8, 1, 0.3],  # Light blue
            [1, 1, 0.8, 0.3],    # Light yellow
            [1, 0.8, 1, 0.3],    # Light magenta
            [0.8, 1, 1, 0.3],    # Light cyan
        ]
        
        color = colors[robot_index % len(colors)]
        x_center, y_center, _ = robot_pos
        
        # Create floor marking
        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[self.workspace_size/2 - 0.05, self.workspace_size/2 - 0.05, 0.001],
            rgbaColor=color
        )
        
        floor_marking = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_shape,
            basePosition=[x_center, y_center, 0.001]
        )
        
    def _create_robot(self, position: List[float], robot_index: int) -> int:
        """Create a single robot at specified position"""
        urdf_path = self._get_urdf_path()
        
        if urdf_path and Path(urdf_path).exists():
            # Set search paths
            urdf_dir = Path(urdf_path).parent
            p.setAdditionalSearchPath(str(urdf_dir))
            p.setAdditionalSearchPath(str(urdf_dir / "assets"))
            
            robot_id = p.loadURDF(
                urdf_path,
                position,
                p.getQuaternionFromEuler([0, 0, 0]),
                useFixedBase=True,
                flags=p.URDF_USE_INERTIA_FROM_FILE
            )
            
            # Color code robots differently
            self._color_robot(robot_id, robot_index)
            
            return robot_id
        else:
            raise RuntimeError("SO101 URDF not found")
    
    def _color_robot(self, robot_id: int, robot_index: int):
        """Apply unique colors to distinguish robots"""
        colors = [
            [0.8, 0.2, 0.2, 1.0],  # Red
            [0.2, 0.8, 0.2, 1.0],  # Green
            [0.2, 0.2, 0.8, 1.0],  # Blue
            [0.8, 0.8, 0.2, 1.0],  # Yellow
            [0.8, 0.2, 0.8, 1.0],  # Magenta
            [0.2, 0.8, 0.8, 1.0],  # Cyan
        ]
        
        color = colors[robot_index % len(colors)]
        
        # Color all links
        num_links = p.getNumJoints(robot_id)
        p.changeVisualShape(robot_id, -1, rgbaColor=color)  # Base link
        for i in range(num_links):
            p.changeVisualShape(robot_id, i, rgbaColor=color)
    
    def _setup_robot_joints(self, robot_id: int) -> List[int]:
        """Setup joint control for a single robot (4 DOF only)"""
        joint_indices = []
        num_joints = p.getNumJoints(robot_id)
        
        for i in range(num_joints):
            info = p.getJointInfo(robot_id, i)
            joint_name = info[1].decode('utf-8')
            joint_type = info[2]
            
            # Only include first 4 arm joints (exclude gripper and wrist_roll)
            if (joint_type == p.JOINT_REVOLUTE and 
                'gripper' not in joint_name.lower() and 
                'wrist_roll' not in joint_name.lower()):
                joint_indices.append(i)
                
        return joint_indices[:4]  # Limit to 4 joints for active vision
    
    def _get_target_position(self, robot_pos: List[float], robot_index: int) -> List[float]:
        """Get random target position within robot's isolated workspace"""
        x_center, y_center, _ = robot_pos
        workspace_bound = self.workspace_size / 2 - 0.3  # Stay away from walls
        
        # Random position within the workspace
        x_offset = np.random.uniform(-workspace_bound, workspace_bound)
        y_offset = np.random.uniform(-workspace_bound, workspace_bound)
        
        target_pos = [
            x_center + x_offset,
            y_center + y_offset,
            0.1  # Height above ground
        ]
        
        return target_pos
    
    def _create_target(self, position: List[float], robot_index: int) -> int:
        """Create target object for robot"""
        # Different colored cubes for each robot
        colors = [
            [1, 0, 0, 1],    # Red
            [0, 1, 0, 1],    # Green  
            [0, 0, 1, 1],    # Blue
            [1, 1, 0, 1],    # Yellow
            [1, 0, 1, 1],    # Magenta
            [0, 1, 1, 1],    # Cyan
        ]
        
        color = colors[robot_index % len(colors)]
        
        # Create visual cube (easier to see than sphere)
        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.03, 0.03, 0.03],  # 6cm cube
            rgbaColor=color
        )
        
        # Create collision cube
        collision_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.03, 0.03, 0.03]
        )
        
        target_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position
        )
        
        return target_id
    
    def _get_urdf_path(self):
        """Get URDF path"""
        urdf_dir = Path("urdf")
        urdf_path = urdf_dir / "so101_new_calib.urdf"
        return str(urdf_path) if urdf_path.exists() else None
        
    def reset(self, seed=None, options=None):
        """Reset all robots and targets"""
        super().reset(seed=seed)
        
        # Clear existing objects
        if self.robot_ids:
            for robot_id in self.robot_ids:
                p.removeBody(robot_id)
            for target_id in self.target_ids:
                p.removeBody(target_id)
            for wall_group in self.wall_ids:
                for wall_id in wall_group:
                    p.removeBody(wall_id)
        
        self.robot_ids.clear()
        self.target_ids.clear()
        self.wall_ids.clear()
        self.robot_joint_indices.clear()
        
        # Recreate environment
        self._create_environment()
        
        # Reset episode counters
        self.episode_steps = [0] * self.n_robots
        
        # Get initial observations
        observations = self._get_all_observations()
        info = self._get_all_info()
        
        return observations, info
    
    def step(self, actions: np.ndarray):
        """Step all robots simultaneously"""
        # Split actions for each robot (4 DOF each)
        actions_per_robot = actions.reshape(self.n_robots, 4)
        
        # Apply actions to each robot
        for i, (robot_id, robot_actions) in enumerate(zip(self.robot_ids, actions_per_robot)):
            joint_indices = self.robot_joint_indices[i]
            
            # Apply joint control
            for j, action in enumerate(robot_actions):
                if j < len(joint_indices):
                    joint_idx = joint_indices[j]
                    current_pos = p.getJointState(robot_id, joint_idx)[0]
                    target_pos = current_pos + action * 0.05  # Smaller action scale
                    
                    p.setJointMotorControl2(
                        robot_id, joint_idx,
                        p.POSITION_CONTROL,
                        targetPosition=target_pos,
                        maxVelocity=1.0
                    )
        
        # Step simulation
        p.stepSimulation()
        
        # Get observations and rewards for all robots
        observations = self._get_all_observations()
        rewards = self._calculate_all_rewards(observations)
        
        # Check termination conditions
        terminated = [False] * self.n_robots
        truncated = [False] * self.n_robots
        
        for i in range(self.n_robots):
            self.episode_steps[i] += 1
            if self.episode_steps[i] >= self.max_episode_steps:
                truncated[i] = True
        
        info = self._get_all_info()
        
        return observations, rewards, terminated, truncated, info
    
    def _get_all_observations(self) -> Dict:
        """Get observations for all robots"""
        observations = {}
        
        for i in range(self.n_robots):
            obs = self._get_robot_observation(i)
            observations[f'robot_{i}'] = obs
            
        return observations
    
    def _get_robot_observation(self, robot_index: int) -> Dict:
        """Get observation for single robot"""
        robot_id = self.robot_ids[robot_index]
        target_id = self.target_ids[robot_index]
        joint_indices = self.robot_joint_indices[robot_index]
        
        # Get camera image
        camera_image = self._get_camera_image(robot_id)
        
        # Get joint positions (4 DOF only)
        joint_positions = []
        for joint_idx in joint_indices:
            joint_state = p.getJointState(robot_id, joint_idx)
            joint_positions.append(joint_state[0])
        
        # Pad to exactly 4 joints if needed
        while len(joint_positions) < 4:
            joint_positions.append(0.0)
        joint_positions = np.array(joint_positions[:4], dtype=np.float32)
        
        # Get target detection info
        target_in_view, center_distance = self._detect_target_in_camera(
            camera_image, target_id, robot_id
        )
        
        return {
            'camera_image': camera_image,
            'joint_positions': joint_positions,
            'target_in_view': np.array([target_in_view], dtype=np.float32),
            'target_center_distance': np.array([center_distance], dtype=np.float32)
        }
    
    def _get_camera_image(self, robot_id: int) -> np.ndarray:
        """Get camera image for specific robot"""
        # Find camera link
        camera_link_idx = -1
        num_joints = p.getNumJoints(robot_id)
        
        for i in range(num_joints):
            info = p.getJointInfo(robot_id, i)
            link_name = info[12].decode('utf-8')
            if 'camera' in link_name.lower():
                camera_link_idx = i
                break
        
        if camera_link_idx >= 0:
            link_state = p.getLinkState(robot_id, camera_link_idx)
        else:
            # Use end-effector as fallback
            link_state = p.getLinkState(robot_id, num_joints-1)
        
        camera_pos = link_state[0]
        camera_orn = link_state[1]
        
        # Calculate camera view
        rotation_matrix = p.getMatrixFromQuaternion(camera_orn)
        forward_direction = [
            rotation_matrix[0],
            rotation_matrix[3], 
            rotation_matrix[6]
        ]
        
        # Camera looks downward at workspace
        target_pos = [
            camera_pos[0] + forward_direction[0] * 0.3,
            camera_pos[1] + forward_direction[1] * 0.3,
            max(camera_pos[2] - 0.2, 0.05)  # Look down but not below ground
        ]
        
        up_vector = [0, 0, 1]
        
        # Get camera image
        view_matrix = p.computeViewMatrix(camera_pos, target_pos, up_vector)
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=1.0, nearVal=0.1, farVal=2.0
        )
        
        width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
            self.camera_width, self.camera_height, view_matrix, proj_matrix
        )
        
        # Convert to numpy array
        rgb_array = np.array(rgb_img).reshape(height, width, 4)[:, :, :3]
        return rgb_array.astype(np.uint8)
    
    def _detect_target_in_camera(self, camera_image: np.ndarray, 
                                 target_id: int, robot_id: int) -> Tuple[float, float]:
        """Detect if target is visible and centered in camera"""
        # Simple color-based detection for the target cube
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(camera_image, cv2.COLOR_RGB2HSV)
        
        # Detect red regions (assuming red targets for now)
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
            # Find largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            if area > 50:  # Minimum area threshold
                # Calculate center
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Calculate distance from image center
                    img_center_x = camera_image.shape[1] // 2
                    img_center_y = camera_image.shape[0] // 2
                    
                    distance = np.sqrt((cx - img_center_x)**2 + (cy - img_center_y)**2)
                    max_distance = np.sqrt(img_center_x**2 + img_center_y**2)
                    
                    normalized_distance = min(distance / max_distance, 1.0)
                    
                    return 1.0, normalized_distance
        
        return 0.0, 1.0
    
    def _calculate_all_rewards(self, observations: Dict) -> np.ndarray:
        """Calculate rewards for all robots"""
        rewards = []
        
        for i in range(self.n_robots):
            obs = observations[f'robot_{i}']
            reward = self._calculate_single_reward(obs)
            rewards.append(reward)
            
        return np.array(rewards, dtype=np.float32)
    
    def _calculate_single_reward(self, observation: Dict) -> float:
        """Calculate reward for single robot"""
        reward = 0.0
        
        target_in_view = observation['target_in_view'][0]
        center_distance = observation['target_center_distance'][0]
        
        # Base reward for having target in view
        if target_in_view > 0.5:
            reward += 3.0
            
            # Smooth centering reward
            centering_reward = np.exp(-2.0 * center_distance) * 10.0
            reward += centering_reward
            
            # Precision bonuses
            if center_distance < 0.05:
                reward += 15.0
            elif center_distance < 0.1:
                reward += 8.0
            elif center_distance < 0.2:
                reward += 4.0
        else:
            # Penalty for losing target
            reward -= 2.0
            
        return reward
    
    def _get_all_info(self) -> Dict:
        """Get info for all robots"""
        info = {}
        for i in range(self.n_robots):
            info[f'robot_{i}'] = {
                'episode_step': self.episode_steps[i],
                'robot_id': self.robot_ids[i] if i < len(self.robot_ids) else None,
                'target_id': self.target_ids[i] if i < len(self.target_ids) else None,
                'workspace_isolated': True
            }
        return info
    
    def close(self):
        """Clean up environment"""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None
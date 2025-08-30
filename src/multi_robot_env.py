import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import cv2
from typing import Tuple, Dict, List, Optional
from pathlib import Path

class SO101MultiRobotEnv(gym.Env):
    """
    Multi-robot SO101 environment for parallel training
    Multiple robots and targets in single PyBullet environment
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
        self.robot_joint_indices = []
        
        # Workspace parameters
        self.workspace_radius = 0.8
        self.robot_spacing = 1.5  # Distance between robots
        
        # Define action/observation spaces for all robots
        single_robot_action_dim = 5  # 5 joints per robot
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
                    low=-np.pi, high=np.pi, shape=(5,), dtype=np.float32
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
        """Initialize PyBullet physics"""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            
        if self.render_mode == "human":
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
            
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1./240.)
        
    def _create_environment(self):
        """Create multi-robot environment"""
        # Load ground plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Calculate robot positions in a grid
        positions = self._calculate_robot_positions()
        
        print(f"🤖 Creating {self.n_robots} robots...")
        
        for i in range(self.n_robots):
            robot_pos = positions[i]
            
            # Create robot
            robot_id = self._create_robot(robot_pos, i)
            self.robot_ids.append(robot_id)
            
            # Setup joints for this robot
            joint_indices = self._setup_robot_joints(robot_id)
            self.robot_joint_indices.append(joint_indices)
            
            # Create target for this robot
            target_pos = [robot_pos[0] + 0.5, robot_pos[1], 0.1]
            target_id = self._create_target(target_pos, i)
            self.target_ids.append(target_id)
            
            print(f"  Robot {i}: position {robot_pos}, {len(joint_indices)} joints")
        
        print(f"✅ Created {len(self.robot_ids)} robots with targets")
        
    def _calculate_robot_positions(self) -> List[List[float]]:
        """Calculate robot base positions in grid layout"""
        positions = []
        
        if self.n_robots == 1:
            positions = [[0, 0, 0]]
        elif self.n_robots <= 4:
            # 2x2 grid
            for i in range(self.n_robots):
                x = -self.robot_spacing/2 + (i % 2) * self.robot_spacing
                y = -self.robot_spacing/2 + (i // 2) * self.robot_spacing
                positions.append([x, y, 0])
        else:
            # Larger grid for more robots
            grid_size = int(np.ceil(np.sqrt(self.n_robots)))
            for i in range(self.n_robots):
                x = -grid_size * self.robot_spacing/2 + (i % grid_size) * self.robot_spacing
                y = -grid_size * self.robot_spacing/2 + (i // grid_size) * self.robot_spacing
                positions.append([x, y, 0])
                
        return positions
        
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
        """Setup joint control for a single robot"""
        joint_indices = []
        num_joints = p.getNumJoints(robot_id)
        
        for i in range(num_joints):
            info = p.getJointInfo(robot_id, i)
            joint_name = info[1].decode('utf-8')
            joint_type = info[2]
            
            # Exclude gripper joints, include arm joints only
            if joint_type == p.JOINT_REVOLUTE and 'gripper' not in joint_name.lower():
                joint_indices.append(i)
                
        return joint_indices[:5]  # Limit to 5 joints
    
    def _create_target(self, position: List[float], robot_index: int) -> int:
        """Create target object for robot"""
        # Different colored spheres for each robot
        colors = [
            [1, 0, 0, 1],    # Red
            [0, 1, 0, 1],    # Green  
            [0, 0, 1, 1],    # Blue
            [1, 1, 0, 1],    # Yellow
            [1, 0, 1, 1],    # Magenta
            [0, 1, 1, 1],    # Cyan
        ]
        
        color = colors[robot_index % len(colors)]
        
        # Create visual sphere
        visual_shape = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.03,
            rgbaColor=color
        )
        
        # Create collision sphere  
        collision_shape = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius=0.03
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
        
        self.robot_ids.clear()
        self.target_ids.clear()
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
        # Split actions for each robot
        actions_per_robot = actions.reshape(self.n_robots, 5)
        
        # Apply actions to each robot
        for i, (robot_id, robot_actions) in enumerate(zip(self.robot_ids, actions_per_robot)):
            joint_indices = self.robot_joint_indices[i]
            
            # Apply joint control
            for j, action in enumerate(robot_actions):
                if j < len(joint_indices):
                    joint_idx = joint_indices[j]
                    current_pos = p.getJointState(robot_id, joint_idx)[0]
                    target_pos = current_pos + action * 0.1  # Scale action
                    
                    p.setJointMotorControl2(
                        robot_id, joint_idx,
                        p.POSITION_CONTROL,
                        targetPosition=target_pos,
                        maxVelocity=2.0
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
        
        # Get joint positions
        joint_positions = []
        for joint_idx in joint_indices:
            joint_state = p.getJointState(robot_id, joint_idx)
            joint_positions.append(joint_state[0])
        joint_positions = np.array(joint_positions, dtype=np.float32)
        
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
        
        # Camera looks downward
        target_pos = [
            camera_pos[0] + forward_direction[0] * 0.5,
            camera_pos[1] + forward_direction[1] * 0.5,
            camera_pos[2] - 0.3  # Look down
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
        # Get target position in world
        target_pos, _ = p.getBasePositionAndOrientation(target_id)
        
        # Get camera transform
        camera_link_idx = p.getNumJoints(robot_id) - 1  # Last link
        link_state = p.getLinkState(robot_id, camera_link_idx)
        camera_pos = link_state[0]
        
        # Simple distance-based visibility check
        distance_to_target = np.linalg.norm(np.array(target_pos) - np.array(camera_pos))
        
        if distance_to_target < 1.0:  # Within camera range
            target_in_view = 1.0
            # Approximate center distance (could be improved with actual projection)
            center_distance = min(distance_to_target / 1.0, 1.0)
        else:
            target_in_view = 0.0
            center_distance = 1.0
            
        return target_in_view, center_distance
    
    def _calculate_all_rewards(self, observations: Dict) -> np.ndarray:
        """Calculate rewards for all robots"""
        rewards = []
        
        for i in range(self.n_robots):
            obs = observations[f'robot_{i}']
            reward = self._calculate_single_reward(obs)
            rewards.append(reward)
            
        return np.array(rewards, dtype=np.float32)
    
    def _calculate_single_reward(self, observation: Dict) -> float:
        """Calculate reward for single robot (same as original)"""
        reward = 0.0
        
        target_in_view = observation['target_in_view'][0]
        center_distance = observation['target_center_distance'][0]
        
        if target_in_view > 0.5:
            reward += 3.0
            centering_reward = np.exp(-2.0 * center_distance) * 10.0
            reward += centering_reward
            
            if center_distance < 0.05:
                reward += 15.0
            elif center_distance < 0.1:
                reward += 8.0
            elif center_distance < 0.2:
                reward += 4.0
        else:
            reward -= 2.0
            
        return reward
    
    def _get_all_info(self) -> Dict:
        """Get info for all robots"""
        info = {}
        for i in range(self.n_robots):
            info[f'robot_{i}'] = {
                'episode_step': self.episode_steps[i],
                'robot_id': self.robot_ids[i] if i < len(self.robot_ids) else None,
                'target_id': self.target_ids[i] if i < len(self.target_ids) else None
            }
        return info
    
    def _get_urdf_path(self):
        """Get URDF path"""
        urdf_dir = Path("urdf")
        return str(urdf_dir / "so101_new_calib.urdf")
    
    def close(self):
        """Clean up environment"""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None
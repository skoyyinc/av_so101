"""
RL Environment wrapper for search behavior optimization

This module wraps the SO101CameraTrackingEnv to focus specifically on 
search episodes when the target object is lost from view.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Any, Optional, List
import time

from ..so_arm_gym_env import SO101CameraTrackingEnv
from .utils import (
    normalize_joint_positions, 
    compute_workspace_distance,
    check_joint_limits,
    compute_exploration_bonus
)


class SearchRLEnv(gym.Env):
    """
    Gymnasium environment wrapper for SO101 search behavior training
    
    This environment focuses specifically on search episodes - when the target
    is lost and the robot must find it again using RL-optimized search patterns.
    """
    
    def __init__(self, 
                 base_env: Optional[SO101CameraTrackingEnv] = None,
                 max_search_steps: int = 300,  # 30 seconds at 10Hz
                 reward_config: Optional[Dict] = None,
                 max_joint_velocities: Optional[List[float]] = None,
                 static_target: bool = True,
                 target_position: Optional[List[float]] = None,
                 min_visual_servoing_steps: int = 100,  # 10 seconds at 10Hz
                 centering_threshold: float = 0.1):
        """
        Initialize search RL environment
        
        Args:
            base_env: Base SO101 environment (creates new if None)
            max_search_steps: Maximum steps per search episode
            reward_config: Custom reward configuration
            max_joint_velocities: Max velocity for each joint [rad/s] (uses defaults if None)
        """
        super().__init__()
        
        # Create or use provided base environment
        if base_env is None:
            self.base_env = SO101CameraTrackingEnv(render_mode="rgb_array")
        else:
            self.base_env = base_env
            
        # Episode configuration
        self.max_search_steps = max_search_steps
        self.current_step = 0
        self.search_start_time = None
        
        # Robot configuration (from base environment)
        self.num_joints = 6
        self.joint_limits = [
            (-np.pi, np.pi),      # Base rotation
            (-np.pi/2, np.pi/2),  # Shoulder
            (-np.pi/2, np.pi/2),  # Elbow  
            (-np.pi/2, np.pi/2),  # Wrist 1
            (-np.pi/2, np.pi/2),  # Wrist 2
            (-np.pi, np.pi)       # Wrist rotation
        ]
        
        # Joint velocity limits - doubled for very aggressive search behavior
        if max_joint_velocities is None:
            self.max_joint_velocities = [
                8.0,  # Base rotation - extremely fast for wide search sweeps
                7.0,  # Shoulder - doubled for large arm movements
                7.0,  # Elbow - doubled for large arm movements
                5.6,  # Wrist 1 - doubled but controlled
                5.6,  # Wrist 2 - doubled but controlled
                8.0   # Wrist rotation - extremely fast for orientation changes
            ]
        else:
            if len(max_joint_velocities) != 6:
                raise ValueError("max_joint_velocities must have 6 values for 6-DOF robot")
            self.max_joint_velocities = max_joint_velocities
        
        # Minimum velocity limits - ensure robot always moves with significant speed
        self.min_joint_velocities = [
            1.0,  # Base rotation - minimum pan speed
            0.8,  # Shoulder - minimum arm movement
            0.8,  # Elbow - minimum arm movement  
            0.5,  # Wrist 1 - minimum wrist movement
            0.5,  # Wrist 2 - minimum wrist movement
            1.0   # Wrist rotation - minimum orientation change
        ]
        
        # Target positioning configuration
        self.static_target = static_target
        if target_position is not None:
            self.custom_target_position = np.array(target_position)
        else:
            # Default static position: left side, further away
            angle = np.radians(50)   # 75° from front (proper left side)
            distance = 1.5      # 0.75m from robot
            height = 0.35            # 35cm height
            self.custom_target_position = np.array([
                distance * np.cos(angle),
                distance * np.sin(angle),
                height
            ])
        
        # State tracking
        self.last_target_position = self.custom_target_position.copy()
        self.visited_positions = []
        self.previous_joint_positions = None  # For tracking movement
        self.pan_direction_history = []  # Track pan direction for consistency bonus
        self.episode_data = {}
        
        # Visual servoing phase tracking
        self.min_visual_servoing_steps = min_visual_servoing_steps
        self.centering_threshold = centering_threshold
        self.visual_servoing_start_step = None
        self.visual_servoing_steps = 0
        self.target_centered = False
        
        # Enhanced reward configuration for dynamic movement
        default_rewards = {
            'target_found': 100.0,
            'positioning_bonus_scale': 50.0, 
            'timestep_penalty': -1.0,  # Base penalty
            'active_search_penalty': -0.3,  # Reduced penalty for active movement
            'idle_penalty': -2.5,  # Higher penalty for minimal movement
            'joint_violation_penalty': -8.0,  # Slightly reduced to not over-penalize speed
            'exploration_bonus': 1.0,  # Increased exploration reward
            'stuck_penalty': -8.0,  # Increased stuck penalty
            'movement_magnitude_bonus': 3.0,  # NEW: Reward for larger movements
            'shoulder_pan_bonus': 2.0,  # Increased reward for base rotation
            'directional_consistency_bonus': 1.0,  # Increased consistency bonus
            'exploration_speed_bonus': 2.0,  # NEW: Bonus for fast exploration
            'workspace_coverage_bonus': 1.5  # NEW: Bonus for covering more area
        }
        self.reward_config = reward_config or default_rewards
        
        # Define spaces
        self._define_spaces()
        
        print("🔍 SearchRLEnv initialized:")
        print(f"   Max search steps: {self.max_search_steps}")
        print(f"   State space: {self.observation_space.shape}")
        print(f"   Action space: {self.action_space.shape}")
        print(f"   Max joint velocities: {self.max_joint_velocities} rad/s")
        print(f"   Min joint velocities: {self.min_joint_velocities} rad/s")
        if self.static_target:
            x, y, z = self.custom_target_position
            angle_deg = np.degrees(np.arctan2(y, x))
            distance = np.linalg.norm([x, y])
            print(f"   Static target: {angle_deg:.1f}°, {distance:.2f}m, height {z:.2f}m")
    
    def _define_spaces(self):
        """Define observation and action spaces"""
        
        # State space: [joint_pos(6), time_lost(1), last_target_pos(3), workspace_dist(1)] = 11D
        state_low = np.array([
            -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,  # Normalized joint positions [-1, 1]
            0.0,                                   # Time since lost [0, 1] 
            0.0, -1.0, 0.0,                       # Last target position [m] (workspace bounds)
            0.0                                    # Workspace distance [0, 1]
        ], dtype=np.float32)
        
        state_high = np.array([
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0,        # Normalized joint positions
            1.0,                                   # Time since lost (normalized)
            1.0, 1.0, 1.0,                        # Last target position 
            1.0                                    # Workspace distance
        ], dtype=np.float32)
        
        self.observation_space = spaces.Box(low=state_low, high=state_high, dtype=np.float32)
        
        # Action space: joint velocities [-1, 1] for all 6 joints
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_joints,), dtype=np.float32
        )
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to start of search episode
        
        Returns:
            observation: Initial state for search episode
            info: Episode information
        """
        super().reset(seed=seed)
        
        # Reset base environment
        base_obs, base_info = self.base_env.reset()
        
        # Force target to be lost (simulate search scenario)
        self._setup_search_scenario()
        
        # Reset episode tracking
        self.current_step = 0
        self.search_start_time = time.time()
        self.visited_positions = []
        self.previous_joint_positions = None
        
        # Reset visual servoing phase tracking
        self.visual_servoing_start_step = None
        self.visual_servoing_steps = 0
        self.target_centered = False
        self.pan_direction_history = []
        
        # Initialize episode data for metrics
        self.episode_data = {
            'start_joint_positions': self.base_env._get_joint_positions().copy(),
            'target_position': self.base_env.target_position.copy(),
            'search_start_time': self.search_start_time,
            'joint_violations': 0,
            'collisions': 0,
            'exploration_cells_visited': set(),
            'stuck_counter': 0
        }
        
        # Get initial RL observation
        observation = self._get_rl_observation()
        info = self._get_episode_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the search environment
        
        Args:
            action: Normalized joint velocity commands [-1, 1]
            
        Returns:
            observation: Next state
            reward: Step reward
            terminated: Episode terminated (target found)
            truncated: Episode truncated (timeout/failure)
            info: Step information
        """
        self.current_step += 1
        
        # Denormalize action - use different scaling for visual servoing vs search
        if self.visual_servoing_start_step is not None:  # Visual servoing mode
            # During visual servoing, use gentler max velocities for smooth tracking
            visual_max_velocities = np.array([1.3, 1.3, 1.3, 1.3, 1.3, 1.3])  # Consistent with visual policy
            denormalized_action = action * visual_max_velocities
        else:  # Search mode
            # During search, use aggressive max velocities for fast exploration  
            denormalized_action = action * np.array(self.max_joint_velocities)
        
        # Apply minimum velocity enforcement ONLY during search phase (not visual servoing)
        # During visual servoing, allow small precise movements for smooth centering
        if self.visual_servoing_start_step is None:  # Still in search mode
            for i in range(len(denormalized_action)):
                if abs(denormalized_action[i]) < self.min_joint_velocities[i]:
                    # If action is too small, boost it to minimum velocity while preserving direction
                    sign = np.sign(denormalized_action[i]) if denormalized_action[i] != 0 else np.random.choice([-1, 1])
                    denormalized_action[i] = sign * self.min_joint_velocities[i]
        # During visual servoing phase, use the original action as-is for smooth tracking
        
        base_obs, base_reward, base_terminated, base_truncated, base_info = self.base_env.step(denormalized_action)
        
        # Check if target was found
        target_found = base_obs['target_in_view'][0] > 0.5
        center_distance = abs(base_obs['target_center_distance'][0])
        
        # Track visual servoing phase
        if target_found:
            if self.visual_servoing_start_step is None:
                self.visual_servoing_start_step = self.current_step
                print(f"🎯 Target found at step {self.current_step}! Starting visual servoing phase...")
                print(f"   Switching to gentle action scaling (1.3 rad/s max)")
            
            self.visual_servoing_steps = self.current_step - self.visual_servoing_start_step
            
            # Check if target is centered
            self.target_centered = center_distance < self.centering_threshold
            
            # Progress logging every 50 steps during visual servoing
            if self.visual_servoing_steps > 0 and self.visual_servoing_steps % 50 == 0:
                print(f"   📏 Visual servoing step {self.visual_servoing_steps}: center_distance={center_distance:.3f}, centered={self.target_centered}")
            
        else:
            # Target lost, reset visual servoing tracking
            if self.visual_servoing_start_step is not None:
                print("❌ Target lost during visual servoing! Returning to search mode...")
                print(f"   Switching back to aggressive action scaling (8.0+ rad/s max)")
            self.visual_servoing_start_step = None
            self.visual_servoing_steps = 0
            self.target_centered = False
        
        # Update episode tracking
        current_joint_pos = base_obs['joint_positions']
        self._update_episode_tracking(current_joint_pos, action, target_found)
        
        # Compute RL-specific reward
        reward = self._compute_reward(base_obs, action, target_found)
        
        # Check termination conditions - only terminate if target is centered AND minimum time elapsed
        terminated = (
            target_found and 
            self.target_centered and 
            self.visual_servoing_steps >= self.min_visual_servoing_steps
        )
        
        truncated = (
            self.current_step >= self.max_search_steps or
            self.episode_data['joint_violations'] > 3 or
            self.episode_data['stuck_counter'] > 50
        )
        
        # Get RL observation and info
        observation = self._get_rl_observation()
        info = self._get_episode_info()
        
        # Add search-specific info
        info.update({
            'target_found': target_found,
            'target_centered': self.target_centered,
            'visual_servoing_steps': self.visual_servoing_steps,
            'center_distance': center_distance if target_found else -1.0,
            'search_time': time.time() - self.search_start_time,
            'steps_taken': self.current_step,
            'outcome': 'success' if terminated else ('timeout' if truncated else 'ongoing'),
            'phase': 'visual_servoing' if target_found else 'search'
        })
        
        return observation, reward, terminated, truncated, info
    
    def get_hybrid_observation(self) -> Tuple[np.ndarray, Dict]:
        """
        Get both RL state vector and base environment observation
        
        This is needed for hybrid policies that need both the RL state
        and access to camera image for target visibility detection.
        
        Returns:
            rl_state: 11D state vector for RL policy
            base_obs: Full base environment observation dict
        """
        rl_state = self._get_rl_observation()
        base_obs = self.base_env._get_observation()
        return rl_state, base_obs
    
    def _setup_search_scenario(self):
        """
        Setup the environment for a search scenario
        Uses static or dynamic target placement based on configuration
        """
        if self.static_target:
            # Use predefined static target position
            self.base_env.target_position = self.custom_target_position.copy()
            self.last_target_position = self.base_env.target_position.copy()
            
            # Debug output for target placement validation
            x, y, z = self.base_env.target_position
            target_angle_deg = np.degrees(np.arctan2(y, x))
            target_distance = np.linalg.norm([x, y])
            
            print(f"🎯 Static target: angle={target_angle_deg:.1f}°, distance={target_distance:.2f}m, height={z:.2f}m")
            print(f"   Position: x={x:.3f}, y={y:.3f}, z={z:.3f}")
            
        else:
            # Original dynamic placement logic (kept for future use)
            print("⚠️  Dynamic target placement not implemented in simplified version")
            # Fallback to static position
            self.base_env.target_position = self.custom_target_position.copy()
            self.last_target_position = self.base_env.target_position.copy()
        
        # Update target object position in simulation
        if self.base_env.target_object_id is not None:
            import pybullet as p
            p.resetBasePositionAndOrientation(
                self.base_env.target_object_id,
                self.base_env.target_position,
                [0, 0, 0, 1]
            )
    
    def _get_rl_observation(self) -> np.ndarray:
        """
        Convert base environment observation to RL state vector
        
        Returns:
            11D state vector for RL policy
        """
        # Get base environment observation
        base_obs = self.base_env._get_observation()
        
        # Extract joint positions and normalize
        joint_positions = base_obs['joint_positions']
        normalized_joints = normalize_joint_positions(joint_positions, self.joint_limits)
        
        # Time since search started (normalized to [0, 1])
        time_lost = min(self.current_step / self.max_search_steps, 1.0)
        
        # Last known target position (normalized to workspace bounds)
        target_pos_norm = np.array([
            (self.last_target_position[0] - 0.2) / 0.8,  # X: [0.2, 1.0] -> [0, 1]
            (self.last_target_position[1] + 0.5) / 1.0,  # Y: [-0.5, 0.5] -> [0, 1]  
            (self.last_target_position[2] - 0.1) / 0.5   # Z: [0.1, 0.6] -> [0, 1]
        ])
        target_pos_norm = np.clip(target_pos_norm, 0.0, 1.0)
        
        # Current workspace distance (normalized)
        current_ee_pos = self._estimate_end_effector_position(joint_positions)
        workspace_dist = compute_workspace_distance(current_ee_pos)
        
        # Combine into state vector
        state = np.concatenate([
            normalized_joints,     # 6D
            [time_lost],          # 1D
            target_pos_norm,      # 3D  
            [workspace_dist]      # 1D
        ]).astype(np.float32)
        
        return state
    
    def _estimate_end_effector_position(self, joint_positions: np.ndarray) -> np.ndarray:
        """
        Rough estimate of end-effector position for workspace calculations
        
        In a full implementation, this would use proper forward kinematics.
        For now, using a simplified approximation.
        """
        # Simplified FK approximation
        # This is just for workspace distance calculation
        base_angle = joint_positions[0]
        shoulder_angle = joint_positions[1] 
        elbow_angle = joint_positions[2]
        
        # Rough arm lengths (approximate SO101 dimensions)
        l1 = 0.15  # Shoulder to elbow
        l2 = 0.15  # Elbow to wrist
        
        # 2D projection in base frame
        elbow_x = l1 * np.cos(shoulder_angle)
        elbow_z = l1 * np.sin(shoulder_angle)
        
        ee_x = elbow_x + l2 * np.cos(shoulder_angle + elbow_angle)
        ee_z = elbow_z + l2 * np.sin(shoulder_angle + elbow_angle) + 0.1  # Base height
        
        # Rotate by base angle
        ee_world_x = ee_x * np.cos(base_angle)
        ee_world_y = ee_x * np.sin(base_angle)
        ee_world_z = ee_z
        
        return np.array([ee_world_x, ee_world_y, ee_world_z])
    
    def _compute_reward(self, base_obs: Dict, action: np.ndarray, target_found: bool) -> float:
        """
        Compute RL reward for search behavior
        
        Args:
            base_obs: Base environment observation
            action: Action taken this step
            target_found: Whether target was found
            
        Returns:
            Reward value
        """
        reward = 0.0
        
        # Primary reward: finding the target
        if target_found:
            reward += self.reward_config['target_found']
            
            # Bonus for good positioning (helps visual servoing)
            center_distance = base_obs['target_center_distance'][0]
            positioning_bonus = self.reward_config['positioning_bonus_scale'] * (1.0 - center_distance)
            reward += positioning_bonus
        
        # Dynamic time penalty based on movement activity (adjusted for higher velocities)
        action_magnitude = np.linalg.norm(action)
        if action_magnitude > 0.3:  # Lowered threshold since minimum velocities enforce movement
            reward += self.reward_config['active_search_penalty']  # Reduced penalty for active search
        else:
            reward += self.reward_config['idle_penalty']  # Higher penalty for idle behavior
        
        # Movement magnitude bonus (encourage larger actions)
        movement_bonus = self.reward_config['movement_magnitude_bonus'] * min(action_magnitude / 0.8, 1.0)
        reward += movement_bonus
        
        # Safety penalties
        joint_positions = base_obs['joint_positions']
        if not check_joint_limits(joint_positions, self.joint_limits):
            reward += self.reward_config['joint_violation_penalty']
            self.episode_data['joint_violations'] += 1
        
        # Enhanced exploration rewards
        current_ee_pos = self._estimate_end_effector_position(joint_positions)
        exploration_bonus = compute_exploration_bonus(
            self.visited_positions, current_ee_pos, grid_size=0.1
        )
        if exploration_bonus > 0:
            reward += self.reward_config['exploration_bonus']
            # Add to visited positions
            self.visited_positions.append(current_ee_pos.copy())
        
        # Exploration speed bonus (reward for fast workspace coverage)
        if len(self.visited_positions) >= 2:
            movement_distance = np.linalg.norm(
                self.visited_positions[-1] - self.visited_positions[-2]
            )
            # Normalize distance and give bonus for significant movement (adjusted for higher speeds)
            if movement_distance > 0.03:  # > 3cm movement (higher threshold for faster movements)
                speed_bonus = self.reward_config['exploration_speed_bonus'] * min(movement_distance / 0.08, 1.0)
                reward += speed_bonus
        
        # Workspace coverage bonus (reward for exploring diverse areas)
        if len(self.visited_positions) >= 5:
            # Calculate spatial diversity of recent positions
            recent_positions = np.array(self.visited_positions[-5:])
            position_variance = np.var(recent_positions, axis=0).sum()
            coverage_bonus = self.reward_config['workspace_coverage_bonus'] * min(position_variance / 0.1, 1.0)
            reward += coverage_bonus
        
        # Enhanced shoulder pan movement bonus (encourage base rotation for search)
        if self.previous_joint_positions is not None:
            pan_delta = joint_positions[0] - self.previous_joint_positions[0]
            shoulder_pan_movement = abs(pan_delta)
            
            # Give bonus for significant shoulder pan movement (search behavior)
            if shoulder_pan_movement > 0.05:  # > ~3 degrees movement
                # Base movement bonus (increased)
                pan_bonus = self.reward_config['shoulder_pan_bonus'] * min(shoulder_pan_movement / 0.15, 1.0)
                reward += pan_bonus
                
                # Direction consistency bonus (reward for sweeping in same direction)
                pan_direction = np.sign(pan_delta)  # +1 for right, -1 for left
                self.pan_direction_history.append(pan_direction)
                
                # Check for consistent direction (last 3-5 movements)
                if len(self.pan_direction_history) >= 3:
                    recent_directions = self.pan_direction_history[-5:]  # Last 5 movements
                    # Count consecutive movements in same direction
                    consecutive_count = 1
                    for i in range(len(recent_directions)-2, -1, -1):
                        if recent_directions[i] == recent_directions[-1]:
                            consecutive_count += 1
                        else:
                            break
                    
                    # Give increasing bonus for consistent sweeping motion
                    if consecutive_count >= 2:
                        consistency_bonus = self.reward_config['directional_consistency_bonus'] * min(consecutive_count / 3.0, 1.0)
                        reward += consistency_bonus
        
        # Update previous joint positions for next step
        self.previous_joint_positions = joint_positions.copy()
        
        # Enhanced stuck penalty (encourage consistent movement)
        if len(self.visited_positions) >= 2:
            recent_movement = np.linalg.norm(
                self.visited_positions[-1] - self.visited_positions[-2]
            )
            # More sensitive stuck detection for faster response (adjusted for higher speeds)
            if recent_movement < 0.025:  # Less than 2.5cm movement (higher threshold for faster movements)
                self.episode_data['stuck_counter'] += 1
                # Escalating stuck penalty
                if self.episode_data['stuck_counter'] > 5:  # Faster trigger
                    stuck_multiplier = min(self.episode_data['stuck_counter'] / 10.0, 2.0)
                    reward += self.reward_config['stuck_penalty'] * stuck_multiplier
            else:
                self.episode_data['stuck_counter'] = 0
        
        return reward
    
    def _update_episode_tracking(self, joint_positions: np.ndarray, 
                               action: np.ndarray, target_found: bool):
        """Update episode tracking data for metrics"""
        
        # Track exploration
        current_ee_pos = self._estimate_end_effector_position(joint_positions)
        grid_pos = tuple(np.round(current_ee_pos / 0.1).astype(int))
        self.episode_data['exploration_cells_visited'].add(grid_pos)
    
    def _get_episode_info(self) -> Dict[str, Any]:
        """Get episode information for logging and analysis"""
        
        info = {
            'episode_step': self.current_step,
            'max_steps': self.max_search_steps,
            'joint_violations': self.episode_data['joint_violations'],
            'exploration_coverage': len(self.episode_data['exploration_cells_visited']),
            'stuck_counter': self.episode_data['stuck_counter']
        }
        
        return info
    
    def render(self, mode: str = 'rgb_array'):
        """Render the environment (delegates to base environment)"""
        return self.base_env.render()
    
    def close(self):
        """Close the environment"""
        if hasattr(self.base_env, 'close'):
            self.base_env.close()
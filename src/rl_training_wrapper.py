import gymnasium as gym
import numpy as np
from typing import Dict, Any, Optional, Tuple
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.type_aliases import GymStepReturn
from src.so_arm_gym_env import SO101CameraTrackingEnv

class SO101RLWrapper(gym.Wrapper):
    """
    RL training wrapper for SO101 active vision
    Optimizes environment for SAC training
    """
    
    def __init__(self, env: SO101CameraTrackingEnv, 
                 max_episode_steps: int = 200,
                 success_threshold: float = 0.05,
                 success_steps_required: int = 20):
        super().__init__(env)
        
        self.max_episode_steps = max_episode_steps
        self.success_threshold = success_threshold
        self.success_steps_required = success_steps_required
        
        # Episode tracking
        self.step_count = 0
        self.success_count = 0
        self.cumulative_reward = 0.0
        
        # Performance metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rate = 0.0
        
    def reset(self, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset with RL-appropriate episode management"""
        obs, info = self.env.reset(**kwargs)
        
        # Reset episode tracking
        self.step_count = 0
        self.success_count = 0
        self.cumulative_reward = 0.0
        
        # Add RL-specific info
        info.update({
            'episode_step': self.step_count,
            'success_rate': self.success_rate,
            'is_success': False
        })
        
        return obs, info
    
    def step(self, action: np.ndarray) -> GymStepReturn:
        """Step with RL-appropriate termination and info"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.step_count += 1
        self.cumulative_reward += reward
        
        # Success detection
        target_in_view = obs['target_in_view'][0] > 0.5
        well_centered = obs['target_center_distance'][0] < self.success_threshold
        
        if target_in_view and well_centered:
            self.success_count += 1
        else:
            self.success_count = 0
        
        # Episode termination conditions
        is_success = self.success_count >= self.success_steps_required
        
        if is_success:
            terminated = True
            reward += 50.0  # Success bonus
            print(f"🎯 Episode success! Target centered for {self.success_count} steps")
        
        # Time limit
        if self.step_count >= self.max_episode_steps:
            truncated = True
        
        # Enhanced info for RL
        info.update({
            'episode_step': self.step_count,
            'success_count': self.success_count,
            'is_success': is_success,
            'cumulative_reward': self.cumulative_reward,
            'target_centered': well_centered,
            'target_visible': target_in_view
        })
        
        # Track episode statistics
        if terminated or truncated:
            self.episode_rewards.append(self.cumulative_reward)
            self.episode_lengths.append(self.step_count)
            
            # Update success rate (last 100 episodes)
            recent_episodes = self.episode_rewards[-100:]
            recent_successes = sum(1 for r in recent_episodes if r > 100)  # Arbitrary success threshold
            self.success_rate = recent_successes / len(recent_episodes) if recent_episodes else 0.0
        
        return obs, reward, terminated, truncated, info

class SO101ImageWrapper(gym.ObservationWrapper):
    """
    Wrapper to handle image observations for CNN-based SAC
    """
    
    def __init__(self, env):
        super().__init__(env)
        
        # Modify observation space to prioritize image
        original_space = env.observation_space
        
        # Get actual joint dimensions from environment
        env.reset()  # Initialize to get actual joint count
        dummy_obs, _ = env.reset()
        actual_joint_count = len(dummy_obs['joint_positions'])
        
        self.observation_space = gym.spaces.Dict({
            'image': original_space['camera_image'],
            'state': gym.spaces.Box(
                low=np.concatenate([
                    -np.pi * np.ones(actual_joint_count),  # Joint positions
                    np.zeros(1),  # target_in_view
                    np.zeros(1)   # target_center_distance
                ]),
                high=np.concatenate([
                    np.pi * np.ones(actual_joint_count),   # Joint positions
                    np.ones(1),   # target_in_view
                    np.ones(1)    # target_center_distance
                ]),
                dtype=np.float32
            )
        })
    
    def observation(self, obs):
        """Reformat observation for RL training"""
        return {
            'image': obs['camera_image'],
            'state': np.concatenate([
                obs['joint_positions'],
                obs['target_in_view'],
                obs['target_center_distance']
            ])
        }

def create_rl_env(render_mode: str = "rgb_array", **kwargs) -> gym.Env:
    """
    Create environment optimized for RL training
    """
    # Base environment
    base_env = SO101CameraTrackingEnv(
        render_mode=render_mode,
        camera_width=128,  # Smaller for faster training
        camera_height=128,
        **kwargs
    )
    
    # Apply wrappers
    env = SO101RLWrapper(base_env)
    env = SO101ImageWrapper(env)
    
    return env

def create_eval_env(render_mode: str = "human", **kwargs) -> gym.Env:
    """
    Create environment for evaluation
    """
    base_env = SO101CameraTrackingEnv(
        render_mode=render_mode,
        camera_width=640,  # Higher resolution for evaluation
        camera_height=480,
        **kwargs
    )
    
    env = SO101RLWrapper(base_env, max_episode_steps=300)  # Longer episodes for eval
    env = SO101ImageWrapper(env)
    
    return env
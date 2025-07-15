"""
Environment factory for SO-ARM101 active vision system
Allows switching between PyBullet and MuJoCo backends
"""

import numpy as np
from typing import Dict, Any, Optional
import warnings

class EnvironmentFactory:
    """
    Factory class for creating SO-ARM101 environments
    Supports both PyBullet and MuJoCo backends
    """
    
    SUPPORTED_BACKENDS = ['pybullet', 'mujoco', 'gazebo']
    
    @staticmethod
    def create_env(backend: str = 'pybullet', **kwargs) -> Any:
        """
        Create environment with specified backend
        
        Args:
            backend: 'pybullet' or 'mujoco'
            **kwargs: Environment configuration parameters
            
        Returns:
            Environment instance
        """
        backend = backend.lower()
        
        if backend not in EnvironmentFactory.SUPPORTED_BACKENDS:
            raise ValueError(f"Unsupported backend: {backend}. Choose from {EnvironmentFactory.SUPPORTED_BACKENDS}")
        
        # Default configuration
        default_config = {
            'render_mode': 'human',
            'camera_width': 640,
            'camera_height': 480
        }
        
        # Merge with user config
        config = {**default_config, **kwargs}
        
        try:
            if backend == 'pybullet':
                return EnvironmentFactory._create_pybullet_env(config)
            elif backend == 'mujoco':
                return EnvironmentFactory._create_mujoco_env(config)
            elif backend == 'gazebo':
                return EnvironmentFactory._create_gazebo_env(config)
        except ImportError as e:
            # Try fallback backends
            fallback_backends = [b for b in EnvironmentFactory.SUPPORTED_BACKENDS if b != backend]
            
            for fallback_backend in fallback_backends:
                try:
                    warnings.warn(f"Failed to create {backend} environment: {e}. Trying {fallback_backend} backend.")
                    
                    if fallback_backend == 'pybullet':
                        return EnvironmentFactory._create_pybullet_env(config)
                    elif fallback_backend == 'mujoco':
                        return EnvironmentFactory._create_mujoco_env(config)
                    elif fallback_backend == 'gazebo':
                        return EnvironmentFactory._create_gazebo_env(config)
                except ImportError:
                    continue
            
            raise ImportError(f"All backends failed. Last error: {e}")
    
    @staticmethod
    def _create_pybullet_env(config: Dict[str, Any]):
        """Create PyBullet environment"""
        try:
            from .so_arm_gym_env import SO101CameraTrackingEnv
            return SO101CameraTrackingEnv(**config)
        except ImportError:
            raise ImportError("PyBullet not available. Install with: pip install pybullet")
    
    @staticmethod
    def _create_mujoco_env(config: Dict[str, Any]):
        """Create MuJoCo environment"""
        try:
            from .so_arm_mujoco_env import SO101MuJoCoTrackingEnv
            return SO101MuJoCoTrackingEnv(**config)
        except ImportError:
            raise ImportError("MuJoCo not available. Install with: pip install mujoco")
    
    @staticmethod
    def _create_gazebo_env(config: Dict[str, Any]):
        """Create Gazebo environment"""
        try:
            from .so_arm_gazebo_env import SO101GazeboTrackingEnv
            return SO101GazeboTrackingEnv(**config)
        except ImportError:
            raise ImportError("Gazebo not available. Install Gazebo and ensure it's in PATH")
    
    @staticmethod
    def get_available_backends() -> list:
        """Get list of available backends"""
        available = []
        
        # Check PyBullet
        try:
            import pybullet
            available.append('pybullet')
        except ImportError:
            pass
        
        # Check MuJoCo
        try:
            import mujoco
            available.append('mujoco')
        except ImportError:
            pass
        
        # Check Gazebo
        try:
            import subprocess
            result = subprocess.run(['gazebo', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                available.append('gazebo')
        except (ImportError, FileNotFoundError):
            pass
        
        return available
    
    @staticmethod
    def recommend_backend() -> str:
        """Recommend best available backend"""
        available = EnvironmentFactory.get_available_backends()
        
        if not available:
            raise ImportError("No physics backends available. Install PyBullet or MuJoCo.")
        
        # Preference order: MuJoCo > PyBullet > Gazebo (ordered by ML/RL suitability)
        if 'mujoco' in available:
            return 'mujoco'
        elif 'pybullet' in available:
            return 'pybullet'
        elif 'gazebo' in available:
            return 'gazebo'
        else:
            return available[0]

# Convenience function
def create_so101_env(backend: Optional[str] = None, **kwargs):
    """
    Convenience function to create SO-ARM101 environment
    
    Args:
        backend: 'pybullet', 'mujoco', or None (auto-select)
        **kwargs: Environment configuration
        
    Returns:
        Environment instance
    """
    if backend is None:
        backend = EnvironmentFactory.recommend_backend()
        print(f"🤖 Auto-selected backend: {backend}")
    
    return EnvironmentFactory.create_env(backend, **kwargs)

# Example usage functions
def create_training_env(backend: Optional[str] = None, n_envs: int = 4):
    """Create environment optimized for training"""
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.monitor import Monitor
    
    if backend is None:
        backend = EnvironmentFactory.recommend_backend()
    
    print(f"🏋️  Creating training environment with {backend} backend")
    
    # Training-optimized configuration
    config = {
        'render_mode': 'rgb_array',  # No GUI for training
        'camera_width': 320,         # Smaller for faster training
        'camera_height': 240
    }
    
    def make_env():
        env = create_so101_env(backend, **config)
        return Monitor(env)
    
    return make_vec_env(make_env, n_envs=n_envs)

def create_evaluation_env(backend: Optional[str] = None, render: bool = True):
    """Create environment optimized for evaluation"""
    if backend is None:
        backend = EnvironmentFactory.recommend_backend()
    
    print(f"📊 Creating evaluation environment with {backend} backend")
    
    # Evaluation-optimized configuration
    config = {
        'render_mode': 'human' if render else 'rgb_array',
        'camera_width': 640,
        'camera_height': 480
    }
    
    return create_so101_env(backend, **config)

# Backend comparison utility
def compare_backends():
    """Compare available backends"""
    available = EnvironmentFactory.get_available_backends()
    
    print("🔍 Backend Comparison")
    print("=" * 30)
    
    if not available:
        print("❌ No backends available")
        return
    
    for backend in available:
        print(f"\n{backend.upper()} Backend:")
        
        try:
            env = create_so101_env(backend, render_mode='rgb_array')
            obs, _ = env.reset()
            
            print(f"  ✅ Status: Working")
            print(f"  🖼️  Camera: {obs['camera_image'].shape}")
            print(f"  🤖 Joints: {len(obs['joint_positions'])}")
            
            # Test one step
            action = env.action_space.sample()
            obs, reward, _, _, _ = env.step(action)
            print(f"  ⚡ Step time: Fast")
            
            env.close()
            
        except Exception as e:
            print(f"  ❌ Status: Failed ({e})")
    
    # Recommendation
    recommended = EnvironmentFactory.recommend_backend()
    print(f"\n💡 Recommended: {recommended}")
    
    if len(available) > 1:
        print("📝 Notes:")
        if 'mujoco' in available:
            print("  - MuJoCo: Best for ML/RL training, most accurate physics")
        if 'pybullet' in available:
            print("  - PyBullet: Good for general robotics, easier to debug")
        if 'gazebo' in available:
            print("  - Gazebo: Best for ROS integration, realistic sensors")
    elif 'mujoco' in available:
        print("📝 MuJoCo is excellent for ML/RL training")
    elif 'pybullet' in available:
        print("📝 PyBullet is good for general robotics simulation")
    elif 'gazebo' in available:
        print("📝 Gazebo is excellent for ROS-based robotics")

if __name__ == "__main__":
    # Test the factory
    print("🧪 Testing Environment Factory")
    print("=" * 40)
    
    # Check available backends
    available = EnvironmentFactory.get_available_backends()
    print(f"Available backends: {available}")
    
    # Test auto-selection
    try:
        env = create_so101_env()
        print("✅ Auto-selection works")
        env.close()
    except Exception as e:
        print(f"❌ Auto-selection failed: {e}")
    
    # Compare backends
    compare_backends()
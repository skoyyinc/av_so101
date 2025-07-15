#!/usr/bin/env python3
"""
Setup and test SO-ARM101 object tracking system
"""

import subprocess
import sys
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    
    requirements = [
        "gymnasium[mujoco]",
        "pybullet", 
        "opencv-python",
        "numpy",
        "matplotlib",
        "requests"
    ]
    
    for req in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", req])
            print(f"✅ Installed {req}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {req}: {e}")

def setup_urdf():
    """Setup SO-ARM101 URDF"""
    print("\n🔧 Setting up SO-ARM101 URDF...")
    
    try:
        from src.setup_so101_urdf import download_so101_urdf
        download_so101_urdf()
    except Exception as e:
        print(f"❌ URDF setup failed: {e}")
        print("⚠️  Will use fallback robot model")

def test_system():
    """Test the complete system"""
    print("\n🧪 Testing object tracking system...")
    
    try:
        from src.so_arm_gym_env import SO101CameraTrackingEnv
        from src.tracking_policy import ImprovedTrackingPolicy
        
        # Quick functionality test
        print("1. Creating environment...")
        env = SO101CameraTrackingEnv(render_mode="rgb_array")
        
        print("2. Resetting environment...")
        observation, info = env.reset()
        
        print("3. Creating tracking policy...")
        policy = ImprovedTrackingPolicy({
            'p_gain_fast': 0.8,
            'p_gain_slow': 0.3,
            'max_velocity': 1.0
        })
        
        print("4. Testing tracking for 50 steps...")
        total_reward = 0
        target_found_steps = 0
        
        for step in range(50):
            action = policy.predict(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if observation['target_in_view'][0] > 0.5:
                target_found_steps += 1
                
            if step % 10 == 0:
                target_visible = "YES" if observation['target_in_view'][0] > 0.5 else "NO"
                error = observation['target_center_distance'][0]
                print(f"   Step {step}: Target visible: {target_visible}, Error: {error:.3f}")
            
            if terminated or truncated:
                break
                
        env.close()
        
        print(f"\n📊 Test Results:")
        print(f"   Total reward: {total_reward:.1f}")
        print(f"   Target visible: {target_found_steps}/50 steps ({target_found_steps/50:.1%})")
        print(f"   Robot loaded: {info.get('robot_loaded', False)}")
        
        if target_found_steps > 10 and total_reward > 0:
            print("✅ System test PASSED!")
            return True
        else:
            print("⚠️  System test shows issues but basic functionality works")
            return False
            
    except Exception as e:
        print(f"❌ System test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main setup and test function"""
    print("🤖 SO-ARM101 Object Tracking Setup")
    print("=" * 50)
    
    # Step 1: Install requirements
    install_requirements()
    
    # Step 2: Setup URDF
    setup_urdf()
    
    # Step 3: Test system
    success = test_system()
    
    if success:
        print(f"\n🎉 Setup complete! You can now run:")
        print(f"   python demo_object_tracking.py")
    else:
        print(f"\n⚠️  Setup completed with issues. Try running the demo anyway:")
        print(f"   python demo_object_tracking.py")

if __name__ == "__main__":
    main()
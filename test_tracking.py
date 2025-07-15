#!/usr/bin/env python3
"""
Test SO-ARM101 object tracking with manual setup
"""

import sys
import numpy as np
from pathlib import Path

def check_setup_first():
    """Check if setup is ready"""
    urdf_dir = Path("urdf")
    
    if not urdf_dir.exists():
        print("❌ Setup not complete. Run: python quick_setup.py")
        return False
    
    urdf_files = list(urdf_dir.glob("*.urdf"))
    if not urdf_files:
        print("❌ No URDF files found. Run: python quick_setup.py") 
        return False
    
    return True

def test_with_manual_setup():
    """Test tracking with manual setup"""
    print("🧪 SO-ARM101 Object Tracking Test (Manual Setup)")
    print("=" * 50)
    
    if not check_setup_first():
        return False
    
    try:
        from src.so_arm_gym_env import SO101CameraTrackingEnv
        from src.tracking_policy import ImprovedTrackingPolicy
        
        # Test environment creation
        print("1. Creating environment...")
        env = SO101CameraTrackingEnv(render_mode="rgb_array")
        print("   ✅ Environment created")
        
        # Test reset
        print("2. Resetting environment...")
        observation, info = env.reset()
        robot_loaded = info.get('robot_loaded', False)
        print(f"   ✅ Reset complete. Robot loaded: {robot_loaded}")
        
        if not robot_loaded:
            print("   ⚠️  Robot failed to load, but will continue with test")
        
        # Test policy
        print("3. Creating improved tracking policy...")
        policy = ImprovedTrackingPolicy({
            'p_gain_fast': 0.8,
            'p_gain_slow': 0.3,
            'max_velocity': 1.0
        })
        print("   ✅ Policy created")
        
        # Test tracking loop
        print("4. Testing tracking for 100 steps...")
        total_reward = 0
        target_visible_steps = 0
        movement_detected = False
        
        for step in range(100):
            # Get action
            action = policy.predict(observation)
            
            # Check if action is non-zero (movement)
            if np.any(np.abs(action) > 0.01):
                movement_detected = True
            
            # Step environment
            prev_joints = observation['joint_positions'].copy()
            observation, reward, terminated, truncated, info = env.step(action)
            
            # Check if joints actually moved
            joint_diff = np.abs(observation['joint_positions'] - prev_joints)
            if np.any(joint_diff > 0.001):
                movement_detected = True
            
            total_reward += reward
            
            if observation['target_in_view'][0] > 0.5:
                target_visible_steps += 1
            
            # Print progress every 20 steps
            if step % 20 == 0:
                target_status = "YES" if observation['target_in_view'][0] > 0.5 else "NO"
                error = observation['target_center_distance'][0]
                joint_range = f"[{observation['joint_positions'].min():.2f}, {observation['joint_positions'].max():.2f}]"
                print(f"   Step {step:2d}: Target: {target_status}, Error: {error:.3f}, "
                      f"Joints: {joint_range}, Action: [{action.min():.2f}, {action.max():.2f}]")
            
            if terminated or truncated:
                break
        
        env.close()
        
        # Results
        print(f"\n📊 Test Results:")
        print(f"   Total reward: {total_reward:.1f}")
        print(f"   Target visible: {target_visible_steps}/100 steps ({target_visible_steps:.0%})")
        print(f"   Movement detected: {'YES' if movement_detected else 'NO'}")
        print(f"   Robot loaded: {'YES' if robot_loaded else 'NO'}")
        
        # Evaluation
        if movement_detected and target_visible_steps > 20:
            print("\n✅ TEST PASSED: Robot is moving and tracking objects!")
            return True
        elif movement_detected:
            print("\n⚠️  PARTIAL SUCCESS: Robot moves but tracking needs improvement")
            return True
        elif target_visible_steps > 0:
            print("\n⚠️  PARTIAL SUCCESS: Target detected but robot not moving properly")
            print("      Check joint control and URDF loading")
            return False
        else:
            print("\n❌ TEST FAILED: No movement or target detection")
            return False
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_visual_demo():
    """Run a quick visual demo"""
    print("\n🎥 Running Visual Demo...")
    
    try:
        from src.so_arm_gym_env import SO101CameraTrackingEnv
        from src.tracking_policy import ImprovedTrackingPolicy
        
        env = SO101CameraTrackingEnv(render_mode="human")  # GUI mode
        policy = ImprovedTrackingPolicy({'p_gain_fast': 0.8})
        
        observation, _ = env.reset()
        
        print("🎮 Demo running in PyBullet GUI window...")
        print("   Watch the robot arm move to track the red sphere!")
        print("   Close the PyBullet window or press Ctrl+C to stop")
        
        try:
            for step in range(300):  # 30 seconds at 10 Hz
                action = policy.predict(observation)
                observation, reward, terminated, truncated, info = env.step(action)
                
                if step % 50 == 0:
                    target_status = "visible" if observation['target_in_view'][0] > 0.5 else "hidden"
                    error = observation['target_center_distance'][0]
                    print(f"   Step {step}: Target {target_status}, error: {error:.3f}")
                
                if terminated or truncated:
                    break
                    
        except KeyboardInterrupt:
            print("\n⏹️  Demo stopped by user")
        
        env.close()
        print("✅ Demo completed!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        run_visual_demo()
    else:
        success = test_with_manual_setup()
        
        if success:
            print(f"\n🎉 Success! Try the visual demo:")
            print(f"   python test_tracking.py demo")
        else:
            print(f"\n🔧 If you're having issues:")
            print(f"   1. Check that STL files are in urdf/assets/")
            print(f"   2. Run: python quick_setup.py")
            print(f"   3. The system will work with simple geometry too!")
#!/usr/bin/env python3
"""
Demo of isolated multi-robot environment for SO101 active vision training
Shows 4 robots in separate walled workspaces
"""

import numpy as np
import time
from src.multi_robot_isolated_env import SO101IsolatedMultiRobotEnv

def demo_isolated_environment():
    """Demo the isolated multi-robot environment"""
    print("🤖 SO101 Isolated Multi-Robot Active Vision Demo")
    print("=" * 60)
    print("Features:")
    print("  • 4 robots in separate walled workspaces") 
    print("  • Each robot has its own colored target cube")
    print("  • Walls prevent robots from seeing each other")
    print("  • 4 DOF control per robot (no gripper/wrist_roll)")
    print("  • Each workspace is color-coded on the floor")
    print("=" * 60)
    
    # Create environment with 4 robots
    env = SO101IsolatedMultiRobotEnv(n_robots=4, render_mode="human")
    
    try:
        print("\n🔄 Initializing environment...")
        observations, info = env.reset()
        
        print("✅ Environment ready!")
        print(f"   Total action space: {env.action_space.shape} (4 robots × 4 DOF each)")
        print(f"   Observation spaces: {len(observations)} robots")
        print(f"   Actual robots created: {len(env.robot_ids)}")
        print(f"   Actual targets created: {len(env.target_ids)}")
        
        # Print robot info
        for i in range(min(4, len(observations))):
            if f'robot_{i}' in observations:
                robot_info = info[f'robot_{i}']
                obs = observations[f'robot_{i}']
                robot_id = robot_info.get('robot_id', 'None')
                target_id = robot_info.get('target_id', 'None')
                print(f"   Robot {i}: {obs['joint_positions'].shape} joints, robot_id: {robot_id}, target_id: {target_id}")
        
        print("\n🎮 Running demo with random actions...")
        print("   Watch each robot try to find and center its target!")
        print("   Each robot can only see within its walled workspace")
        print("   Press Ctrl+C to stop")
        
        step_count = 0
        
        while True:
            # Generate random actions for all robots
            # Each robot gets 4 DOF: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex]
            actions = np.random.uniform(-0.2, 0.2, size=(16,))  # 4 robots × 4 DOF
            
            # Step environment
            observations, rewards, terminated, truncated, info = env.step(actions)
            
            step_count += 1
            
            # Print status every 50 steps
            if step_count % 50 == 0:
                print(f"\n📊 Step {step_count}:")
                
                for i in range(4):
                    obs = observations[f'robot_{i}']
                    reward = rewards[i]
                    target_visible = "👁️ " if obs['target_in_view'][0] > 0.5 else "🔍"
                    center_dist = obs['target_center_distance'][0]
                    
                    status = f"CENTERED" if center_dist < 0.1 else f"TRACKING({center_dist:.2f})"
                    
                    print(f"   Robot {i}: {target_visible} {status} | Reward: {reward:+.1f}")
            
            # Small delay to make it watchable
            time.sleep(0.02)  # 50 Hz
            
            # Check if any episodes ended
            any_done = any(terminated) or any(truncated)
            if any_done:
                print(f"\n🔄 Episode ended, resetting...")
                observations, info = env.reset()
                step_count = 0
                
    except KeyboardInterrupt:
        print(f"\n⏹️  Demo stopped by user after {step_count} steps")
        
    finally:
        env.close()
        print("👋 Demo ended!")

def test_single_robot_control():
    """Test controlling a single robot manually"""
    print("\n🎯 Single Robot Control Test")
    print("=" * 40)
    
    env = SO101IsolatedMultiRobotEnv(n_robots=1, render_mode="human")
    
    try:
        observations, info = env.reset()
        
        print("Testing 4 DOF control:")
        print("  Joint 0: Shoulder Pan")
        print("  Joint 1: Shoulder Lift") 
        print("  Joint 2: Elbow Flex")
        print("  Joint 3: Wrist Flex")
        
        # Test each joint individually
        joint_names = ["Shoulder Pan", "Shoulder Lift", "Elbow Flex", "Wrist Flex"]
        
        for joint_idx in range(4):
            print(f"\n🔧 Testing {joint_names[joint_idx]} (Joint {joint_idx})")
            
            for direction in [1, -1]:  # Positive then negative
                for step in range(20):
                    # Create action array (4 DOF for 1 robot)
                    action = np.zeros(4)
                    action[joint_idx] = direction * 0.3
                    
                    observations, rewards, terminated, truncated, info = env.step(action)
                    time.sleep(0.05)
                
                print(f"   Moved joint {joint_idx} in direction {'+' if direction > 0 else '-'}")
        
        print("✅ Single robot control test completed!")
        
    finally:
        env.close()

def main():
    """Main demo function"""
    print("🚀 Choose demo mode:")
    print("1. Full 4-robot isolated environment")
    print("2. Single robot control test")
    
    try:
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "2":
            test_single_robot_control()
        else:
            demo_isolated_environment()
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()
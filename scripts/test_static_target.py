#!/usr/bin/env python3
"""
Test script for static target placement validation

This script tests the simplified static target positioning to ensure
it's working correctly before training.
"""

import sys
import os
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.rl import SearchRLEnv
from src.so_arm_gym_env import SO101CameraTrackingEnv


def test_static_target_placement():
    """Test static target placement functionality"""
    print("🧪 Testing static target placement...")
    
    try:
        # Test 1: Default static position
        print("\n📍 Test 1: Default static position")
        base_env = SO101CameraTrackingEnv(render_mode="rgb_array")
        search_env = SearchRLEnv(
            base_env=base_env,
            static_target=True  # Default position
        )
        
        # Reset to trigger target placement
        obs, info = search_env.reset()
        target_pos = search_env.base_env.target_position
        
        # Verify position
        x, y, z = target_pos
        angle = np.degrees(np.arctan2(y, x))
        distance = np.linalg.norm([x, y])
        
        print(f"   Target position: [{x:.3f}, {y:.3f}, {z:.3f}]")
        print(f"   Angle: {angle:.1f}° (expected ~75°)")
        print(f"   Distance: {distance:.3f}m (expected ~0.75m)")
        print(f"   Height: {z:.3f}m (expected ~0.35m)")
        
        # Validation
        if 70 <= angle <= 80 and 0.7 <= distance <= 0.8 and 0.3 <= z <= 0.4:
            print("   ✅ Default position is correct")
        else:
            print("   ❌ Default position is incorrect")
        
        search_env.close()
        
        # Test 2: Custom static position
        print("\n📍 Test 2: Custom static position")
        custom_pos = [0.5, 0.5, 0.4]  # Custom position: right side, closer
        
        base_env = SO101CameraTrackingEnv(render_mode="rgb_array")
        search_env = SearchRLEnv(
            base_env=base_env,
            static_target=True,
            target_position=custom_pos
        )
        
        # Reset to trigger target placement
        obs, info = search_env.reset()
        target_pos = search_env.base_env.target_position
        
        # Verify custom position
        print(f"   Expected: {custom_pos}")
        print(f"   Actual: {target_pos.tolist()}")
        
        if np.allclose(target_pos, custom_pos, atol=0.001):
            print("   ✅ Custom position is correct")
        else:
            print("   ❌ Custom position is incorrect")
        
        search_env.close()
        
        # Test 3: Multiple resets (should be consistent)
        print("\n📍 Test 3: Consistency across resets")
        base_env = SO101CameraTrackingEnv(render_mode="rgb_array")
        search_env = SearchRLEnv(
            base_env=base_env,
            static_target=True
        )
        
        positions = []
        for i in range(5):
            obs, info = search_env.reset()
            positions.append(search_env.base_env.target_position.copy())
        
        # Check consistency
        all_same = all(np.allclose(pos, positions[0], atol=0.001) for pos in positions)
        
        print(f"   Reset 1: {positions[0]}")
        print(f"   Reset 5: {positions[-1]}")
        
        if all_same:
            print("   ✅ Static position is consistent across resets")
        else:
            print("   ❌ Static position varies between resets")
        
        search_env.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_visual_verification():
    """Visual test to see static target placement"""
    print("\n🎬 Visual verification test...")
    print("This test opens PyBullet GUI to visually verify target placement.")
    
    try:
        # Create environment with GUI
        base_env = SO101CameraTrackingEnv(render_mode="human")
        search_env = SearchRLEnv(
            base_env=base_env,
            static_target=True
        )
        
        print("📺 PyBullet GUI opened. Look for the red target cube.")
        print("   Expected: Left side of robot, further away")
        print("   Press Enter to continue or Ctrl+C to skip...")
        
        try:
            input()  # Wait for user
            
            # Reset a few times to show it stays static
            for i in range(3):
                print(f"   Reset {i+1}/3...")
                obs, info = search_env.reset()
                target_pos = search_env.base_env.target_position
                print(f"   Target: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
                
                # Brief pause to see the position
                import time
                time.sleep(2)
            
            print("✅ Visual verification completed")
            
        except KeyboardInterrupt:
            print("⏭️  Visual test skipped by user")
        
        search_env.close()
        return True
        
    except Exception as e:
        print(f"❌ Visual test failed: {e}")
        return False


def main():
    """Run static target placement tests"""
    print("🎯 Static Target Placement Test")
    print("=" * 50)
    print("Testing simplified static target positioning for RL training")
    
    success_count = 0
    total_tests = 2
    
    # Run placement tests
    if test_static_target_placement():
        success_count += 1
    
    # Optional visual test
    print("\n" + "=" * 30)
    visual_test = input("Run visual verification test? (y/N): ").lower().strip()
    if visual_test == 'y':
        if test_visual_verification():
            success_count += 1
        total_tests += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"🏁 Test Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("✅ All tests passed! Static target placement is working correctly.")
        print("🚀 Ready for simplified RL training with static target.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

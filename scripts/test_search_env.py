#!/usr/bin/env python3
"""
Test script for SearchRLEnv - Phase 1 validation

This script validates that the RL environment wrapper works correctly
without breaking the existing system.
"""

import sys
import os
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.rl import SearchRLEnv, ComprehensiveMetricsTracker


def test_search_env_basic():
    """Test basic SearchRLEnv functionality"""
    print("🧪 Testing SearchRLEnv basic functionality...")
    
    try:
        # Create environment
        env = SearchRLEnv()
        print(f"✅ Environment created successfully")
        print(f"   State space: {env.observation_space.shape}")
        print(f"   Action space: {env.action_space.shape}")
        
        # Test reset
        obs, info = env.reset()
        print(f"✅ Reset successful, observation shape: {obs.shape}")
        print(f"   Info keys: {list(info.keys())}")
        
        # Test a few steps
        for step in range(5):
            action = env.action_space.sample()  # Random action
            obs, reward, terminated, truncated, info = env.step(action)
            
            print(f"   Step {step+1}: reward={reward:.2f}, terminated={terminated}, truncated={truncated}")
            
            if terminated or truncated:
                break
        
        env.close()
        print("✅ Basic functionality test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False


def test_metrics_tracker():
    """Test ComprehensiveMetricsTracker"""
    print("\n🧪 Testing ComprehensiveMetricsTracker...")
    
    try:
        # Create metrics tracker
        tracker = ComprehensiveMetricsTracker(log_dir="test_logs")
        print("✅ MetricsTracker created successfully")
        
        # Simulate some episode data
        for episode in range(3):
            episode_data = {
                'total_reward': np.random.uniform(-10, 50),
                'steps': np.random.randint(10, 100),
                'outcome': np.random.choice(['success', 'timeout', 'stuck']),
                'search_time': np.random.uniform(5, 30),
                'workspace_coverage': np.random.uniform(0.1, 0.8),
                'joint_violations': np.random.randint(0, 3),
                'collisions': 0,
                'difficulty_level': 1,
                'target_distance': np.random.uniform(0.3, 0.8)
            }
            
            tracker.update_episode(episode_data)
        
        # Get summary
        summary = tracker.get_performance_summary()
        print(f"✅ Summary generated: {len(summary)} metrics")
        
        # Save report
        tracker.save_summary_report()
        print("✅ Summary report saved")
        
        print("✅ MetricsTracker test passed!")
        return True
        
    except Exception as e:
        print(f"❌ MetricsTracker test failed: {e}")
        return False


def test_integration():
    """Test SearchRLEnv with MetricsTracker integration"""
    print("\n🧪 Testing SearchRLEnv + MetricsTracker integration...")
    
    try:
        # Create environment and tracker
        env = SearchRLEnv(max_search_steps=50)  # Short episodes for testing
        tracker = ComprehensiveMetricsTracker(log_dir="integration_test_logs")
        
        # Run a few complete episodes
        for episode in range(3):
            print(f"   Running episode {episode+1}...")
            
            obs, info = env.reset()
            episode_reward = 0.0
            episode_steps = 0
            
            while True:
                # Random action
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                
                episode_reward += reward
                episode_steps += 1
                
                if terminated or truncated:
                    break
            
            # Record episode in tracker
            episode_data = {
                'total_reward': episode_reward,
                'steps': episode_steps,
                'outcome': info.get('outcome', 'unknown'),
                'search_time': info.get('search_time', 0.0),
                'workspace_coverage': info.get('exploration_coverage', 0) / 10.0,  # Normalize
                'joint_violations': info.get('joint_violations', 0),
                'collisions': 0,
                'difficulty_level': 1,
                'target_distance': 0.5
            }
            
            tracker.update_episode(episode_data)
            print(f"      Episode {episode+1}: {episode_data['outcome']}, "
                  f"reward={episode_reward:.1f}, steps={episode_steps}")
        
        env.close()
        
        # Final summary
        summary = tracker.get_performance_summary()
        print(f"✅ Integration test passed!")
        print(f"   Final success rate: {summary['overall_success_rate']:.3f}")
        print(f"   Mean reward: {summary['mean_episode_reward']:.1f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False


def main():
    """Run all Phase 1 tests"""
    print("🚀 Running Phase 1 Validation Tests")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    # Run tests
    if test_search_env_basic():
        tests_passed += 1
    
    if test_metrics_tracker():
        tests_passed += 1
        
    if test_integration():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print(f"🏁 Phase 1 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("✅ All tests passed! Phase 1 implementation is ready.")
        print("🎯 Ready to proceed to Phase 2 (PPO Implementation)")
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
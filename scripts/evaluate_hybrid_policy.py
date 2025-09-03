#!/usr/bin/env python3
"""
Evaluation script for HybridTrackingPolicy

This script evaluates the hybrid policy that automatically switches
between RL search and visual servoing based on target visibility.
"""

import sys
import os
import argparse
import time
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.hybrid_tracking_policy import HybridTrackingPolicy
from src.so_arm_gym_env import SO101CameraTrackingEnv


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate HybridTrackingPolicy')
    
    parser.add_argument('--rl-model', type=str, default='models/search_ppo_model.zip',
                       help='Path to trained RL search model')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of evaluation episodes (default: 10)')
    parser.add_argument('--max-steps', type=int, default=1000,
                       help='Maximum steps per episode (default: 1000)')
    parser.add_argument('--render', action='store_true',
                       help='Render episodes during evaluation')
    parser.add_argument('--deterministic', action='store_true', default=True,
                       help='Use deterministic RL policy (default: True)')
    
    return parser.parse_args()


def evaluate_hybrid_policy(args):
    """Evaluate the hybrid tracking policy"""
    
    print("🔀 Evaluating Hybrid Tracking Policy")
    print("=" * 50)
    
    # Create environment - use SearchRLEnv for consistent target placement!
    render_mode = "human" if args.render else "rgb_array"
    from src.rl import SearchRLEnv
    base_env = SO101CameraTrackingEnv(render_mode=render_mode)
    
    # Wrap in SearchRLEnv with static target (same as visual_test_search_env)
    env = SearchRLEnv(
        base_env=base_env,
        static_target=True,  # Ensure static target placement
        max_search_steps=args.max_steps,
        min_visual_servoing_steps=100,  # 10 seconds at 10Hz
        centering_threshold=0.1  # Target must be centered within 0.1 distance
    )
    
    # Create hybrid policy
    visual_config = {
        'p_gain_fast': 1.2,
        'p_gain_slow': 0.5, 
        'max_velocity': 1.3
    }
    
    try:
        hybrid_policy = HybridTrackingPolicy(
            visual_config=visual_config,
            rl_model_path=args.rl_model,
            target_lost_threshold=5,
            target_found_threshold=2
        )
        print("✅ Hybrid policy created successfully")
    except Exception as e:
        print(f"❌ Failed to create hybrid policy: {e}")
        return
    
    # Evaluation metrics
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    mode_switches = []
    visual_ratios = []
    search_ratios = []
    
    # Run evaluation episodes
    for episode in range(args.episodes):
        print(f"\n🎬 Episode {episode + 1}/{args.episodes}")
        
        # Reset environment and policy
        obs, info = env.reset()
        hybrid_policy.reset()
        
        episode_reward = 0
        episode_length = 0
        done = False
        
        episode_start_time = time.time()
        
        while not done and episode_length < args.max_steps:
            # Get hybrid observation (both RL state and base obs)
            rl_state, base_obs = env.get_hybrid_observation()
            
            # Pass both observations to hybrid policy
            hybrid_obs = {
                'base_obs': base_obs,
                'rl_state': rl_state
            }
            action = hybrid_policy.predict(hybrid_obs, deterministic=args.deterministic)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
            
            # Render if requested
            if args.render:
                env.render()
                time.sleep(0.1)  # 10 Hz
        
        episode_time = time.time() - episode_start_time
        
        # Get performance stats
        stats = hybrid_policy.get_performance_stats()
        
        # Determine success based on proper centering
        success = info.get('target_centered', False) and info.get('visual_servoing_steps', 0) >= 100
        if success:
            success_count += 1
        
        # Record metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        mode_switches.append(stats['mode_switches'])
        visual_ratios.append(stats['visual_servoing_ratio'])
        search_ratios.append(stats['search_ratio'])
        
        # Print episode results
        print(f"   Reward: {episode_reward:.1f}, Steps: {episode_length}")
        print(f"   Mode switches: {stats['mode_switches']}")
        print(f"   Visual/Search ratio: {stats['visual_servoing_ratio']:.2f}/{stats['search_ratio']:.2f}")
        print(f"   Final mode: {stats['current_mode']}")
        print(f"   Target found: {info.get('target_found', False)}")
        print(f"   Visual servoing steps: {info.get('visual_servoing_steps', 0)}")
        print(f"   Target centered: {info.get('target_centered', False)} (distance: {info.get('center_distance', -1):.3f})")
        print(f"   Success: {'✅' if success else '❌'}")
    
    # Clean up
    env.close()
    
    # Compute final statistics
    success_rate = success_count / args.episodes
    avg_reward = np.mean(episode_rewards)
    avg_length = np.mean(episode_lengths)
    avg_switches = np.mean(mode_switches)
    avg_visual_ratio = np.mean(visual_ratios)
    avg_search_ratio = np.mean(search_ratios)
    
    print("\n" + "=" * 50)
    print("📊 EVALUATION RESULTS")
    print("=" * 50)
    print(f"Episodes: {args.episodes}")
    print(f"Success rate: {success_rate:.3f} ({success_count}/{args.episodes})")
    print(f"Average reward: {avg_reward:.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average episode length: {avg_length:.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Average mode switches: {avg_switches:.1f} ± {np.std(mode_switches):.1f}")
    print(f"Average visual ratio: {avg_visual_ratio:.3f}")
    print(f"Average search ratio: {avg_search_ratio:.3f}")
    
    # Detailed statistics
    print(f"\nReward range: {np.min(episode_rewards):.1f} to {np.max(episode_rewards):.1f}")
    print(f"Length range: {np.min(episode_lengths)} to {np.max(episode_lengths)} steps")
    print(f"Mode switch range: {np.min(mode_switches)} to {np.max(mode_switches)}")
    
    return {
        'success_rate': success_rate,
        'avg_reward': avg_reward,
        'avg_length': avg_length,
        'avg_switches': avg_switches,
        'avg_visual_ratio': avg_visual_ratio,
        'avg_search_ratio': avg_search_ratio,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths
    }


def main():
    """Main evaluation function"""
    args = parse_arguments()
    
    # Check if RL model exists
    if not os.path.exists(args.rl_model):
        print(f"❌ RL model not found: {args.rl_model}")
        print("   Please train a model first using: python scripts/train_search_rl.py")
        return
    
    # Run evaluation
    results = evaluate_hybrid_policy(args)
    
    print(f"\n🎯 Hybrid policy evaluation completed!")
    print(f"   Success rate: {results['success_rate']:.3f}")
    print(f"   Average reward: {results['avg_reward']:.1f}")


if __name__ == "__main__":
    main()
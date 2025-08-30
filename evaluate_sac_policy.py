#!/usr/bin/env python3
"""
Evaluate trained SAC policy for SO101 active vision
"""

import numpy as np
import time
from stable_baselines3 import SAC
from src.rl_training_wrapper import create_eval_env
import argparse
import cv2

def evaluate_policy(model_path: str, n_episodes: int = 10, render: bool = True):
    """Evaluate trained policy"""
    
    print(f"📂 Loading model from {model_path}")
    model = SAC.load(model_path)
    
    print("🏗️  Creating evaluation environment...")
    env = create_eval_env(render_mode="human" if render else "rgb_array")
    
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    print(f"🧪 Evaluating for {n_episodes} episodes...")
    print("=" * 50)
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        
        print(f"\n📺 Episode {episode + 1}/{n_episodes}")
        
        while True:
            # Get action from trained policy
            action, _states = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            # Print progress
            if episode_length % 20 == 0:
                target_status = "✅ VISIBLE" if info.get('target_visible', False) else "❌ HIDDEN"
                centered_status = "🎯 CENTERED" if info.get('target_centered', False) else "🔍 TRACKING"
                print(f"  Step {episode_length}: {target_status} | {centered_status} | Reward: {reward:.2f}")
            
            if render and not terminated and not truncated:
                time.sleep(0.05)  # Slow down for better visualization
                
            if terminated or truncated:
                break
        
        # Episode summary
        is_success = info.get('is_success', False)
        if is_success:
            success_count += 1
            print(f"🎉 Episode {episode + 1} SUCCESS! Length: {episode_length}, Reward: {episode_reward:.1f}")
        else:
            print(f"❌ Episode {episode + 1} failed. Length: {episode_length}, Reward: {episode_reward:.1f}")
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    
    # Final statistics
    print("\n" + "=" * 50)
    print("📊 EVALUATION RESULTS")
    print("=" * 50)
    print(f"Success Rate: {success_count}/{n_episodes} ({100*success_count/n_episodes:.1f}%)")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Best Reward: {np.max(episode_rewards):.2f}")
    print(f"Worst Reward: {np.min(episode_rewards):.2f}")
    
    env.close()
    
    return {
        'success_rate': success_count / n_episodes,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths
    }

def record_video(model_path: str, video_path: str = "sac_active_vision_demo.mp4"):
    """Record video of policy performance"""
    
    print(f"🎬 Recording video to {video_path}")
    model = SAC.load(model_path)
    env = create_eval_env(render_mode="rgb_array")
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = None
    
    obs, info = env.reset()
    frame_count = 0
    
    print("🎥 Recording episode...")
    
    while frame_count < 1000:  # Max 1000 frames
        # Get action
        action, _states = model.predict(obs, deterministic=True)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Get frame
        if hasattr(env, 'render'):
            frame = env.render()
        else:
            frame = obs['image']
            
        if frame is not None:
            if out is None:
                h, w = frame.shape[:2]
                out = cv2.VideoWriter(video_path, fourcc, 20.0, (w, h))
            
            if len(frame.shape) == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            
            out.write(frame_bgr)
            frame_count += 1
        
        if terminated or truncated:
            print(f"Episode ended at frame {frame_count}")
            break
    
    if out:
        out.release()
    env.close()
    
    print(f"✅ Video saved to {video_path}")

def compare_with_baseline(model_path: str):
    """Compare SAC policy with rule-based policy"""
    
    print("🆚 Comparing SAC vs Rule-based policy")
    print("=" * 50)
    
    # Evaluate SAC
    print("Testing SAC policy...")
    sac_results = evaluate_policy(model_path, n_episodes=5, render=False)
    
    # Evaluate baseline (would need to implement rule-based policy)
    # For now, just report SAC results
    
    print(f"\n📈 SAC Performance:")
    print(f"  Success Rate: {100*sac_results['success_rate']:.1f}%")
    print(f"  Mean Reward: {sac_results['mean_reward']:.2f}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate SAC active vision policy")
    parser.add_argument("--model", "-m", type=str, required=True, 
                       help="Path to trained SAC model")
    parser.add_argument("--episodes", "-e", type=int, default=10,
                       help="Number of episodes to evaluate")
    parser.add_argument("--no-render", action="store_true",
                       help="Disable rendering during evaluation")
    parser.add_argument("--record", "-r", type=str, 
                       help="Record video to specified path")
    parser.add_argument("--compare", action="store_true",
                       help="Compare with baseline policy")
    
    args = parser.parse_args()
    
    if args.record:
        record_video(args.model, args.record)
    elif args.compare:
        compare_with_baseline(args.model)
    else:
        evaluate_policy(args.model, args.episodes, render=not args.no_render)

if __name__ == "__main__":
    main()
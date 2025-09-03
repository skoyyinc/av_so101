#!/usr/bin/env python3
"""
Evaluation script for trained search RL models

This script evaluates the performance of trained PPO models
and compares them against baseline policies.
"""

import sys
import os
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.rl import SearchRLEnv, ComprehensiveMetricsTracker
from src.tracking_policy import ImprovedTrackingPolicy


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate trained search RL model')
    
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--n-episodes', type=int, default=100,
                       help='Number of evaluation episodes (default: 100)')
    parser.add_argument('--render', action='store_true',
                       help='Render episodes during evaluation')
    parser.add_argument('--deterministic', action='store_true', default=True,
                       help='Use deterministic policy (default: True)')
    parser.add_argument('--log-dir', type=str, default='evaluation_logs',
                       help='Directory for evaluation logs')
    parser.add_argument('--compare-baseline', action='store_true',
                       help='Compare against random baseline')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--save-video', action='store_true',
                       help='Save video recordings of episodes')
    parser.add_argument('--show-plots', action='store_true',
                       help='Display plots during evaluation')
    parser.add_argument('--record-episodes', type=int, default=5,
                       help='Number of episodes to record/render (default: 5)')
    parser.add_argument('--visual-servoing-time', type=float, default=10.0,
                       help='Time (seconds) to run visual servoing after RL finds target (default: 10.0)')
    parser.add_argument('--disable-handoff', action='store_true',
                       help='Disable visual servoing handoff after RL success')
    
    return parser.parse_args()


def evaluate_model(model_path, n_episodes=100, render=False, deterministic=True, log_dir="evaluation_logs", record_episodes=5, save_video=False, visual_servoing_time=10.0, disable_handoff=False):
    """Evaluate trained PPO model"""
    
    try:
        from src.rl.search_agent import SearchPPOAgent
    except ImportError:
        print("❌ stable-baselines3 not available. Please install: pip install stable-baselines3[extra]")
        return None
    
    print(f"📊 Evaluating model: {model_path}")
    print(f"   Episodes: {n_episodes}")
    print(f"   Deterministic: {deterministic}")
    
    # Create environment
    from src.so_arm_gym_env import SO101CameraTrackingEnv
    base_env = SO101CameraTrackingEnv(render_mode="human" if render else "rgb_array")
    search_env = SearchRLEnv(base_env=base_env)
    
    # Create agent and load model
    agent = SearchPPOAgent(env=search_env, log_dir=log_dir)
    agent.load_model(model_path)
    
    # Evaluate with visual recording and optional handoff
    results = evaluate_with_visuals(
        agent=agent,
        env=search_env,
        n_episodes=n_episodes,
        deterministic=deterministic,
        render=render,
        record_episodes=record_episodes,
        save_video=save_video,
        log_dir=log_dir,
        visual_servoing_time=visual_servoing_time,
        disable_handoff=disable_handoff
    )
    
    search_env.close()
    return results


def evaluate_with_visuals(agent, env, n_episodes=100, deterministic=True, render=False, record_episodes=5, save_video=False, log_dir="evaluation_logs", visual_servoing_time=10.0, disable_handoff=False):
    """Enhanced evaluation with visual recording and real-time plotting"""
    
    print(f"📊 Evaluating model with visuals over {n_episodes} episodes...")
    print(f"   Recording first {record_episodes} episodes")
    if not disable_handoff:
        print(f"   Visual servoing handoff: {visual_servoing_time:.1f}s after RL success")
    else:
        print(f"   Visual servoing handoff: Disabled")
    
    # Create visual servoing policy for handoff
    visual_policy = None
    if not disable_handoff:
        visual_policy = ImprovedTrackingPolicy({
            'p_gain_fast': 1.2,
            'p_gain_slow': 0.5,
            'max_velocity': 1.3
        })
    
    eval_metrics = {
        'episode_rewards': [],
        'episode_lengths': [],
        'success_count': 0,
        'timeout_count': 0,
        'stuck_count': 0,
        'search_times': [],
        'joint_violations': [],
        'episode_trajectories': [],  # For visualization
        'recorded_frames': [],  # For video generation
        'handoff_episodes': [],  # Episodes that used visual servoing handoff
        'visual_servoing_performance': []  # Performance during visual servoing phase
    }
    
    # Create progress bar for evaluation
    episode_iterator = range(n_episodes)
    if TQDM_AVAILABLE:
        episode_iterator = tqdm(
            episode_iterator, 
            desc="📊 Evaluating with Visuals", 
            unit="episodes",
            ncols=80
        )
    
    for episode in episode_iterator:
        obs, info = env.reset()
        episode_reward = 0.0
        episode_length = 0
        episode_start_time = time.time()
        trajectory = []
        frames = []
        
        # Record visual data for first few episodes
        record_this_episode = episode < record_episodes
        
        while True:
            action, _ = agent.model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            # Record trajectory data
            if record_this_episode:
                trajectory.append({
                    'step': episode_length,
                    'action': action.copy(),
                    'reward': reward,
                    'joint_positions': obs[:6].copy() if len(obs) >= 6 else None,
                    'target_found': terminated and info.get('outcome') == 'success'
                })
                
                # Capture frame if saving video
                if save_video:
                    try:
                        frame = env.render()  # Get RGB array
                        if frame is not None:
                            frames.append(frame)
                    except:
                        pass
            
            # Render if requested
            if render and record_this_episode:
                env.render()
                time.sleep(0.05)  # Slow down for visual observation
            
            if terminated or truncated:
                rl_search_time = time.time() - episode_start_time
                outcome = info.get('outcome', 'unknown')
                
                # If RL found target and handoff enabled, switch to visual servoing
                visual_servoing_reward = 0.0
                visual_servoing_steps = 0
                total_search_time = rl_search_time
                
                if outcome == 'success' and not disable_handoff and visual_policy is not None:
                    print(f"   🎯 RL found target! Handing off to visual servoing for {visual_servoing_time:.1f}s...")
                    
                    # Run visual servoing for specified time
                    handoff_start_time = time.time()
                    control_freq = 10  # 10 Hz control frequency
                    handoff_steps = int(visual_servoing_time * control_freq)
                    
                    for vs_step in range(handoff_steps):
                        # Get base environment observation for visual servoing
                        base_obs = env.base_env._get_observation()
                        
                        # Use visual servoing policy
                        vs_action = visual_policy.predict(base_obs)
                        
                        # Apply action
                        obs_vs, reward_vs, terminated_vs, truncated_vs, info_vs = env.step(vs_action)
                        
                        visual_servoing_reward += reward_vs
                        visual_servoing_steps += 1
                        episode_reward += reward_vs
                        episode_length += 1
                        
                        # Record trajectory if needed
                        if record_this_episode:
                            trajectory.append({
                                'step': episode_length,
                                'action': vs_action.copy(),
                                'reward': reward_vs,
                                'joint_positions': obs_vs[:6].copy() if len(obs_vs) >= 6 else None,
                                'target_found': True,  # Target should be visible during visual servoing
                                'phase': 'visual_servoing'
                            })
                            
                            # Capture frame if saving video
                            if save_video:
                                try:
                                    frame = env.render()
                                    if frame is not None:
                                        frames.append(frame)
                                except:
                                    pass
                        
                        # Render if requested
                        if render and record_this_episode:
                            env.render()
                            time.sleep(0.1)  # Visual servoing at 10Hz
                        
                        # Check if visual servoing phase should end
                        if terminated_vs or truncated_vs:
                            break
                        
                        # Time-based termination for visual servoing
                        if time.time() - handoff_start_time >= visual_servoing_time:
                            break
                    
                    total_search_time = time.time() - episode_start_time
                    eval_metrics['handoff_episodes'].append(episode)
                    eval_metrics['visual_servoing_performance'].append({
                        'reward': visual_servoing_reward,
                        'steps': visual_servoing_steps,
                        'time': total_search_time - rl_search_time
                    })
                    
                    print(f"   📈 Visual servoing: {visual_servoing_steps} steps, reward={visual_servoing_reward:.1f}")
                
                # Record metrics
                eval_metrics['episode_rewards'].append(episode_reward)
                eval_metrics['episode_lengths'].append(episode_length)
                eval_metrics['search_times'].append(total_search_time)
                eval_metrics['joint_violations'].append(info.get('joint_violations', 0))
                
                if record_this_episode:
                    eval_metrics['episode_trajectories'].append(trajectory)
                    if save_video and frames:
                        eval_metrics['recorded_frames'].append(frames)
                
                if outcome == 'success':
                    eval_metrics['success_count'] += 1
                elif outcome == 'timeout':
                    eval_metrics['timeout_count'] += 1
                elif outcome == 'stuck':
                    eval_metrics['stuck_count'] += 1
                
                # Update progress bar with metrics
                if TQDM_AVAILABLE and hasattr(episode_iterator, 'set_postfix'):
                    success_rate = eval_metrics['success_count'] / (episode + 1)
                    avg_reward = np.mean(eval_metrics['episode_rewards'])
                    episode_iterator.set_postfix({
                        'success': f'{success_rate:.3f}',
                        'reward': f'{avg_reward:.1f}'
                    })
                
                break
    
    # Save visual data
    if save_video and eval_metrics['recorded_frames']:
        save_evaluation_videos(eval_metrics['recorded_frames'], log_dir)
    
    # Generate trajectory plots
    if eval_metrics['episode_trajectories']:
        create_trajectory_plots(eval_metrics['episode_trajectories'], log_dir)
    
    # Compute final statistics
    results = {
        'success_rate': eval_metrics['success_count'] / n_episodes,
        'mean_reward': np.mean(eval_metrics['episode_rewards']),
        'std_reward': np.std(eval_metrics['episode_rewards']),
        'mean_episode_length': np.mean(eval_metrics['episode_lengths']),
        'mean_search_time': np.mean(eval_metrics['search_times']),
        'timeout_rate': eval_metrics['timeout_count'] / n_episodes,
        'stuck_rate': eval_metrics['stuck_count'] / n_episodes,
        'avg_joint_violations': np.mean(eval_metrics['joint_violations'])
    }
    
    print(f"\n📈 Evaluation Results:")
    print(f"   Success rate: {results['success_rate']:.3f}")
    print(f"   Mean reward: {results['mean_reward']:.1f} ± {results['std_reward']:.1f}")
    print(f"   Mean search time: {results['mean_search_time']:.1f}s")
    print(f"   Joint violations: {results['avg_joint_violations']:.2f}/episode")
    
    return results


def save_evaluation_videos(recorded_frames, log_dir):
    """Save recorded frames as videos"""
    try:
        import cv2
        
        video_dir = os.path.join(log_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)
        
        for episode_idx, frames in enumerate(recorded_frames):
            if not frames:
                continue
                
            video_path = os.path.join(video_dir, f"episode_{episode_idx+1}.mp4")
            
            # Get frame dimensions
            height, width, channels = frames[0].shape
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, 10.0, (width, height))
            
            for frame in frames:
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            print(f"📹 Video saved: {video_path}")
            
    except ImportError:
        print("⚠️  OpenCV not available for video saving. Install with: pip install opencv-python")
    except Exception as e:
        print(f"⚠️  Could not save videos: {e}")


def create_trajectory_plots(trajectories, log_dir):
    """Create plots of search trajectories and performance"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Reward trajectories
        for i, traj in enumerate(trajectories[:5]):  # Show first 5 episodes
            rewards = [step['reward'] for step in traj]
            cumulative_rewards = np.cumsum(rewards)
            axes[0,0].plot(cumulative_rewards, alpha=0.7, label=f'Episode {i+1}')
        axes[0,0].set_title('Cumulative Reward Trajectories')
        axes[0,0].set_xlabel('Step')
        axes[0,0].set_ylabel('Cumulative Reward')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Joint movement patterns
        if trajectories and trajectories[0] and trajectories[0][0]['joint_positions'] is not None:
            for i, traj in enumerate(trajectories[:3]):  # Show first 3 episodes
                joint_pos = np.array([step['joint_positions'] for step in traj if step['joint_positions'] is not None])
                if len(joint_pos) > 0:
                    # Show base rotation (joint 0) which is most important for search
                    axes[0,1].plot(joint_pos[:, 0], alpha=0.7, label=f'Episode {i+1}')
            axes[0,1].set_title('Base Rotation During Search')
            axes[0,1].set_xlabel('Step')
            axes[0,1].set_ylabel('Base Angle (rad)')
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Action magnitudes
        for i, traj in enumerate(trajectories[:5]):
            action_mags = [np.linalg.norm(step['action']) for step in traj]
            axes[1,0].plot(action_mags, alpha=0.7, label=f'Episode {i+1}')
        axes[1,0].set_title('Action Magnitudes')
        axes[1,0].set_xlabel('Step')
        axes[1,0].set_ylabel('Action Magnitude')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Episode outcomes and lengths
        episode_lengths = [len(traj) for traj in trajectories]
        success_episodes = [i for i, traj in enumerate(trajectories) if any(step['target_found'] for step in traj)]
        
        axes[1,1].bar(range(len(episode_lengths)), episode_lengths, alpha=0.7, color='lightblue', label='All Episodes')
        if success_episodes:
            success_lengths = [episode_lengths[i] for i in success_episodes]
            axes[1,1].bar(success_episodes, success_lengths, alpha=0.9, color='green', label='Successful')
        
        axes[1,1].set_title('Episode Lengths and Outcomes')
        axes[1,1].set_xlabel('Episode')
        axes[1,1].set_ylabel('Steps to Complete')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(log_dir, "evaluation_trajectories.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"📊 Trajectory plots saved: {plot_path}")
        
        # Show plot if requested
        plt.show()
        
    except Exception as e:
        print(f"⚠️  Could not create trajectory plots: {e}")


def evaluate_random_baseline(n_episodes=100, render=False, log_dir="evaluation_logs"):
    """Evaluate random baseline for comparison"""
    
    print(f"🎲 Evaluating random baseline...")
    print(f"   Episodes: {n_episodes}")
    
    # Create environment
    from src.so_arm_gym_env import SO101CameraTrackingEnv
    base_env = SO101CameraTrackingEnv(render_mode="human" if render else "rgb_array")
    search_env = SearchRLEnv(base_env=base_env)
    
    # Create metrics tracker
    tracker = ComprehensiveMetricsTracker(log_dir=f"{log_dir}/random_baseline")
    
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    search_times = []
    
    # Create progress bar
    episode_iterator = range(n_episodes)
    if TQDM_AVAILABLE:
        episode_iterator = tqdm(episode_iterator, desc="🎲 Random Baseline", unit="episodes", ncols=80)
    
    for episode in episode_iterator:
        obs, info = search_env.reset()
        episode_reward = 0.0
        episode_length = 0
        episode_start_time = time.time()
        
        while True:
            # Random action
            action = search_env.action_space.sample()
            obs, reward, terminated, truncated, info = search_env.step(action)
            
            episode_reward += reward
            episode_length += 1
            
            if render:
                search_env.render()
            
            if terminated or truncated:
                search_time = time.time() - episode_start_time
                outcome = info.get('outcome', 'unknown')
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                search_times.append(search_time)
                
                if outcome == 'success':
                    success_count += 1
                
                # Record in tracker
                episode_data = {
                    'total_reward': episode_reward,
                    'steps': episode_length,
                    'outcome': outcome,
                    'search_time': search_time,
                    'workspace_coverage': info.get('exploration_coverage', 0) / 20.0,
                    'joint_violations': info.get('joint_violations', 0),
                    'collisions': 0,
                    'difficulty_level': 1,
                    'target_distance': 0.5
                }
                tracker.update_episode(episode_data)
                
                # Update progress bar
                if TQDM_AVAILABLE and hasattr(episode_iterator, 'set_postfix'):
                    current_success_rate = success_count / (episode + 1)
                    avg_reward = np.mean(episode_rewards)
                    episode_iterator.set_postfix({
                        'success': f'{current_success_rate:.3f}',
                        'reward': f'{avg_reward:.1f}'
                    })
                
                # Print progress (fallback)
                elif (episode + 1) % 20 == 0:
                    current_success_rate = success_count / (episode + 1)
                    avg_reward = np.mean(episode_rewards)
                    print(f"   Episodes {episode+1:3d}: success_rate={current_success_rate:.3f}, avg_reward={avg_reward:.1f}")
                
                break
    
    # Compute results
    results = {
        'success_rate': success_count / n_episodes,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_episode_length': np.mean(episode_lengths),
        'mean_search_time': np.mean(search_times),
    }
    
    search_env.close()
    return results


def compare_results(rl_results, baseline_results):
    """Compare RL model against baseline"""
    
    print("\n📈 COMPARISON RESULTS")
    print("=" * 50)
    
    metrics = ['success_rate', 'mean_reward', 'mean_episode_length', 'mean_search_time']
    
    for metric in metrics:
        rl_value = rl_results.get(metric, 0)
        baseline_value = baseline_results.get(metric, 0)
        
        if baseline_value != 0:
            improvement = ((rl_value - baseline_value) / baseline_value) * 100
            improvement_str = f"({improvement:+.1f}%)"
        else:
            improvement_str = ""
        
        print(f"{metric:20s}: RL={rl_value:.3f}, Baseline={baseline_value:.3f} {improvement_str}")
    
    # Overall assessment
    success_improvement = ((rl_results['success_rate'] - baseline_results['success_rate']) / baseline_results['success_rate']) * 100 if baseline_results['success_rate'] > 0 else 0
    reward_improvement = ((rl_results['mean_reward'] - baseline_results['mean_reward']) / abs(baseline_results['mean_reward'])) * 100 if baseline_results['mean_reward'] != 0 else 0
    
    print(f"\n🎯 KEY IMPROVEMENTS:")
    print(f"   Success Rate: {success_improvement:+.1f}%")
    print(f"   Mean Reward: {reward_improvement:+.1f}%")
    
    # Success criteria check (from plan)
    print(f"\n✅ SUCCESS CRITERIA CHECK:")
    if rl_results['success_rate'] > 0.8:
        print(f"   ✅ Success rate > 80%: {rl_results['success_rate']:.1%}")
    else:
        print(f"   ❌ Success rate < 80%: {rl_results['success_rate']:.1%}")
    
    if rl_results['mean_search_time'] < 15:
        print(f"   ✅ Mean search time < 15s: {rl_results['mean_search_time']:.1f}s")
    else:
        print(f"   ❌ Mean search time > 15s: {rl_results['mean_search_time']:.1f}s")
    
    if success_improvement > 30:
        print(f"   ✅ >30% improvement over baseline: {success_improvement:.1f}%")
    else:
        print(f"   ❌ <30% improvement over baseline: {success_improvement:.1f}%")


def create_evaluation_plots(rl_results, baseline_results=None, save_path="evaluation_plots.png"):
    """Create evaluation comparison plots"""
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        metrics = ['success_rate', 'mean_reward', 'mean_episode_length', 'mean_search_time']
        titles = ['Success Rate', 'Mean Reward', 'Mean Episode Length', 'Mean Search Time']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i//2, i%2]
            
            rl_value = rl_results.get(metric, 0)
            values = [rl_value]
            labels = ['RL Model']
            colors = ['blue']
            
            if baseline_results:
                baseline_value = baseline_results.get(metric, 0)
                values.append(baseline_value)
                labels.append('Random Baseline')
                colors.append('red')
            
            ax.bar(labels, values, color=colors, alpha=0.7)
            ax.set_title(title)
            ax.set_ylabel(metric.replace('_', ' ').title())
            
            # Add value labels on bars
            for j, v in enumerate(values):
                ax.text(j, v + max(values)*0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"📊 Evaluation plots saved: {save_path}")
        
    except Exception as e:
        print(f"⚠️  Could not create plots: {e}")


def main():
    """Main evaluation function"""
    args = parse_arguments()
    
    print("📊 SO101 Search RL Model Evaluation")
    print("=" * 50)
    
    try:
        # Evaluate trained model
        rl_results = evaluate_model(
            model_path=args.model_path,
            n_episodes=args.n_episodes,
            render=args.render,
            deterministic=args.deterministic,
            log_dir=args.log_dir,
            record_episodes=args.record_episodes,
            save_video=args.save_video,
            visual_servoing_time=args.visual_servoing_time,
            disable_handoff=args.disable_handoff
        )
        
        if rl_results is None:
            return False
        
        # Compare against baseline if requested
        baseline_results = None
        if args.compare_baseline:
            baseline_results = evaluate_random_baseline(
                n_episodes=args.n_episodes,
                render=False,  # Don't render baseline
                log_dir=args.log_dir
            )
        
        # Print results
        print(f"\n📈 RL MODEL RESULTS:")
        for key, value in rl_results.items():
            print(f"   {key}: {value:.3f}")
        
        if baseline_results:
            print(f"\n🎲 BASELINE RESULTS:")
            for key, value in baseline_results.items():
                print(f"   {key}: {value:.3f}")
            
            compare_results(rl_results, baseline_results)
        
        # Create plots
        plot_path = os.path.join(args.log_dir, "evaluation_comparison.png")
        create_evaluation_plots(rl_results, baseline_results, plot_path)
        
        print("\n✅ Evaluation completed!")
        return True
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

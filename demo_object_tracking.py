#!/usr/bin/env python3
"""
SO-ARM101 Object Tracking Demo
Camera-as-end-effector object detection and tracking
"""

import gymnasium as gym
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from src.so_arm_gym_env import SO101CameraTrackingEnv
from src.tracking_policy import SimpleTrackingPolicy, LearningTrackingPolicy

def test_tracking_policy(policy_type="simple", num_episodes=5, render=True):
    """Test object tracking policy"""
    print(f"🎯 Testing {policy_type} tracking policy")
    print("=" * 50)
    
    # Create environment
    render_mode = "human" if render else "rgb_array"
    env = SO101CameraTrackingEnv(render_mode=render_mode)
    
    # Create policy
    config = {
        'policy_type': policy_type,
        'p_gain': 0.5,
        'max_velocity': 0.8
    }
    
    if policy_type == "simple":
        policy = SimpleTrackingPolicy(config)
    else:
        policy = LearningTrackingPolicy(config)
    
    # Performance tracking
    episode_rewards = []
    episode_lengths = []
    tracking_errors = []
    success_episodes = 0
    
    print(f"🚀 Starting {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        print(f"\n📍 Episode {episode + 1}/{num_episodes}")
        
        # Reset environment
        observation, info = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_errors = []
        
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            # Get action from policy
            action = policy.predict(observation)
            
            # Step environment
            observation, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            episode_errors.append(observation['target_center_distance'][0])
            
            # Optional: display camera view
            if episode == 0 and episode_length % 20 == 0:  # Every 2 seconds
                camera_image = observation['camera_image']
                target_pos = policy._detect_target_position(camera_image)
                
                if target_pos is not None:
                    # Draw target center and image center
                    display_img = camera_image.copy()
                    cv2.circle(display_img, target_pos, 5, (0, 255, 0), -1)
                    
                    center_x, center_y = camera_image.shape[1]//2, camera_image.shape[0]//2
                    cv2.circle(display_img, (center_x, center_y), 3, (255, 0, 0), -1)
                    cv2.line(display_img, target_pos, (center_x, center_y), (255, 255, 0), 2)
                    
                    # Save frame for later viewing
                    cv2.imwrite(f'tracking_frame_ep{episode}_step{episode_length}.jpg', 
                              cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR))
            
            time.sleep(0.01)  # Small delay for visualization
            
        # Episode finished
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        tracking_errors.extend(episode_errors)
        
        # Check success (target centered for significant time)
        if info['target_centered_steps'] > 30:
            success_episodes += 1
            print(f"✅ Episode {episode + 1}: SUCCESS! Target centered for {info['target_centered_steps']} steps")
        else:
            print(f"❌ Episode {episode + 1}: Target centered for only {info['target_centered_steps']} steps")
            
        print(f"   Reward: {episode_reward:.1f}, Length: {episode_length}, Avg Error: {np.mean(episode_errors):.3f}")
    
    env.close()
    
    # Performance summary
    print(f"\n{'='*20} PERFORMANCE SUMMARY {'='*20}")
    print(f"🎯 Success Rate: {success_episodes}/{num_episodes} ({success_episodes/num_episodes:.1%})")
    print(f"📊 Average Reward: {np.mean(episode_rewards):.1f} ± {np.std(episode_rewards):.1f}")
    print(f"📏 Average Episode Length: {np.mean(episode_lengths):.0f} ± {np.std(episode_lengths):.0f}")
    print(f"🎯 Average Tracking Error: {np.mean(tracking_errors):.3f} ± {np.std(tracking_errors):.3f}")
    
    # Policy-specific metrics
    policy_metrics = policy.get_performance_metrics()
    print(f"\n📈 Policy Performance:")
    for key, value in policy_metrics.items():
        print(f"   {key}: {value}")
    
    # Plot results
    if num_episodes > 1:
        plot_results(episode_rewards, episode_lengths, tracking_errors)
    
    return {
        'success_rate': success_episodes / num_episodes,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'tracking_errors': tracking_errors,
        'policy_metrics': policy_metrics
    }

def plot_results(rewards, lengths, errors):
   """Plot training results"""
   fig, axes = plt.subplots(2, 2, figsize=(12, 8))
   
   # Episode rewards
   axes[0, 0].plot(rewards, 'b-', marker='o')
   axes[0, 0].set_title('Episode Rewards')
   axes[0, 0].set_xlabel('Episode')
   axes[0, 0].set_ylabel('Total Reward')
   axes[0, 0].grid(True)
   
   # Episode lengths
   axes[0, 1].plot(lengths, 'g-', marker='s')
   axes[0, 1].set_title('Episode Lengths')
   axes[0, 1].set_xlabel('Episode')
   axes[0, 1].set_ylabel('Steps')
   axes[0, 1].grid(True)
   
   # Tracking error distribution
   axes[1, 0].hist(errors, bins=30, alpha=0.7, color='red')
   axes[1, 0].set_title('Tracking Error Distribution')
   axes[1, 0].set_xlabel('Center Distance Error')
   axes[1, 0].set_ylabel('Frequency')
   axes[1, 0].grid(True)
   
   # Tracking error over time (sample)
   sample_errors = errors[::max(1, len(errors)//200)]  # Sample for readability
   axes[1, 1].plot(sample_errors, 'purple', alpha=0.7)
   axes[1, 1].set_title('Tracking Error Over Time')
   axes[1, 1].set_xlabel('Time Steps')
   axes[1, 1].set_ylabel('Center Distance Error')
   axes[1, 1].grid(True)
   
   plt.tight_layout()
   plt.savefig('tracking_performance.png', dpi=150, bbox_inches='tight')
   print("📊 Performance plots saved to 'tracking_performance.png'")
   plt.show()

def compare_policies():
   """Compare different tracking policies"""
   print("🔄 Comparing tracking policies...")
   
   policies = ['simple']  # Can add 'learning' when implemented
   results = {}
   
   for policy_type in policies:
       print(f"\n🧪 Testing {policy_type} policy...")
       results[policy_type] = test_tracking_policy(
           policy_type=policy_type, 
           num_episodes=3, 
           render=False
       )
   
   # Comparison summary
   print(f"\n{'='*30} POLICY COMPARISON {'='*30}")
   for policy_type, result in results.items():
       print(f"\n{policy_type.upper()} POLICY:")
       print(f"  Success Rate: {result['success_rate']:.1%}")
       print(f"  Avg Reward: {np.mean(result['episode_rewards']):.1f}")
       print(f"  Avg Tracking Error: {np.mean(result['tracking_errors']):.3f}")
   
   return results

def demo_with_visualization():
   """Demo with enhanced visualization"""
   print("🎥 Running demo with visualization...")
   
   env = SO101CameraTrackingEnv(render_mode="human")
   policy = SimpleTrackingPolicy({'p_gain': 0.5})
   
   observation, _ = env.reset()
   
   print("\n📋 Demo Instructions:")
   print("- Watch the PyBullet window for robot movement")
   print("- Camera images will be saved as JPG files")
   print("- Press Ctrl+C to stop early")
   
   try:
       for step in range(200):  # Run for 20 seconds at 10 FPS
           action = policy.predict(observation)
           observation, reward, terminated, truncated, info = env.step(action)
           
           # Save camera view every 10 steps
           if step % 10 == 0:
               camera_image = observation['camera_image']
               target_in_view = observation['target_in_view'][0]
               center_distance = observation['target_center_distance'][0]
               
               # Annotate image
               display_img = camera_image.copy()
               
               # Draw center crosshair
               h, w = display_img.shape[:2]
               center_x, center_y = w//2, h//2
               cv2.line(display_img, (center_x-20, center_y), (center_x+20, center_y), (0, 255, 255), 2)
               cv2.line(display_img, (center_x, center_y-20), (center_x, center_y+20), (0, 255, 255), 2)
               
               # Draw target if detected
               target_pos = policy._detect_target_position(camera_image)
               if target_pos is not None:
                   cv2.circle(display_img, target_pos, 10, (0, 255, 0), 3)
                   cv2.line(display_img, target_pos, (center_x, center_y), (255, 0, 0), 2)
               
               # Add status text
               status = f"Step: {step} | Target: {'YES' if target_in_view > 0.5 else 'NO'} | Error: {center_distance:.3f}"
               cv2.putText(display_img, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
               
               # Save frame
               filename = f"demo_step_{step:03d}.jpg"
               cv2.imwrite(filename, cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR))
               
               if step % 50 == 0:
                   print(f"📸 Step {step}: Target {'visible' if target_in_view > 0.5 else 'hidden'}, "
                         f"Error: {center_distance:.3f}, Reward: {reward:.1f}")
           
           if terminated or truncated:
               break
               
           time.sleep(0.1)
   
   except KeyboardInterrupt:
       print("\n⏹️  Demo stopped by user")
   
   env.close()
   print("✅ Demo completed! Check the saved images.")

def main():
   """Main function"""
   print("🤖 SO-ARM101 Object Tracking System")
   print("=" * 50)
   
   while True:
       print("\n📋 Available demos:")
       print("1. Quick tracking test (3 episodes)")
       print("2. Extended evaluation (10 episodes)")
       print("3. Demo with visualization")
       print("4. Compare policies")
       print("5. Exit")
       
       try:
           choice = input("\nEnter your choice (1-5): ").strip()
           
           if choice == "1":
               test_tracking_policy(num_episodes=3, render=True)
           elif choice == "2":
               test_tracking_policy(num_episodes=10, render=False)
           elif choice == "3":
               demo_with_visualization()
           elif choice == "4":
               compare_policies()
           elif choice == "5":
               print("👋 Goodbye!")
               break
           else:
               print("❌ Invalid choice. Please enter 1-5.")
               
       except KeyboardInterrupt:
           print("\n👋 Goodbye!")
           break
       except Exception as e:
           print(f"❌ Error: {e}")

if __name__ == "__main__":
   main()
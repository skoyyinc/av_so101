#!/usr/bin/env python3
"""
Test script to validate improved movement dynamics in SearchRLEnv
Analyzes action magnitudes, movement speeds, and reward components
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.rl import SearchRLEnv, ComprehensiveMetricsTracker


def test_movement_dynamics(num_steps=200):
    """Test the improved movement dynamics and reward structure"""
    print(f"🧪 Testing improved movement dynamics ({num_steps} steps)...")
    
    try:
        # Create environment with new settings
        env = SearchRLEnv(max_search_steps=num_steps)
        print(f"✅ Enhanced SearchRLEnv created")
        print(f"   Joint velocities: {env.max_joint_velocities} rad/s")
        
        # Reset environment
        obs, info = env.reset()
        
        # Track movement metrics
        action_magnitudes = []
        movement_distances = []
        reward_components = []
        joint_velocities_used = []
        positions = []
        
        print(f"\n🎬 Running {num_steps}-step movement test...")
        
        for step in range(num_steps):
            # Generate action with varying magnitudes to test reward response
            if step < 50:
                # Small actions (should get penalties)
                action = np.random.uniform(-0.3, 0.3, 6)
            elif step < 100:
                # Medium actions 
                action = np.random.uniform(-0.6, 0.6, 6)
            else:
                # Large actions (should get bonuses)
                action = np.random.uniform(-1.0, 1.0, 6)
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Calculate metrics
            action_magnitude = np.linalg.norm(action)
            actual_velocities = action * np.array(env.max_joint_velocities)
            
            action_magnitudes.append(action_magnitude)
            joint_velocities_used.append(np.linalg.norm(actual_velocities))
            
            # Estimate movement distance
            if len(env.visited_positions) >= 2:
                movement_dist = np.linalg.norm(
                    env.visited_positions[-1] - env.visited_positions[-2]
                )
                movement_distances.append(movement_dist)
            else:
                movement_distances.append(0.0)
            
            # Track position
            if env.visited_positions:
                positions.append(env.visited_positions[-1].copy())
            
            reward_components.append(reward)
            
            # Print every 50 steps
            if (step + 1) % 50 == 0:
                avg_action_mag = np.mean(action_magnitudes[-50:])
                avg_reward = np.mean(reward_components[-50:])
                avg_movement = np.mean(movement_distances[-50:])
                
                print(f"   Steps {step-49:3d}-{step+1:3d}: "
                      f"action_mag={avg_action_mag:.3f}, "
                      f"reward={avg_reward:.2f}, "
                      f"movement={avg_movement:.4f}m")
            
            if terminated or truncated:
                break
        
        env.close()
        
        # Analyze results
        analyze_movement_results(
            action_magnitudes, movement_distances, reward_components,
            joint_velocities_used, positions
        )
        
        return True
        
    except Exception as e:
        print(f"❌ Movement dynamics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def analyze_movement_results(action_mags, movements, rewards, velocities, positions):
    """Analyze and visualize movement test results"""
    
    action_mags = np.array(action_mags)
    movements = np.array(movements)
    rewards = np.array(rewards)
    velocities = np.array(velocities)
    
    print(f"\n📊 Movement Dynamics Analysis:")
    print(f"   Action magnitude: {action_mags.mean():.3f} ± {action_mags.std():.3f}")
    print(f"   Movement distance: {movements.mean():.4f} ± {movements.std():.4f} m")
    print(f"   Joint velocities: {velocities.mean():.2f} ± {velocities.std():.2f} rad/s")
    print(f"   Reward per step: {rewards.mean():.2f} ± {rewards.std():.2f}")
    
    # Analyze correlation between action magnitude and reward
    if len(action_mags) > 10:
        correlation = np.corrcoef(action_mags, rewards)[0, 1]
        print(f"   Action-Reward correlation: {correlation:.3f}")
        
        if correlation > 0.2:
            print("   ✅ Positive correlation: Larger actions → higher rewards")
        elif correlation < -0.2:
            print("   ❌ Negative correlation: Larger actions → lower rewards")
        else:
            print("   ⚠️  Weak correlation: Reward system may need tuning")
    
    # Check movement phases
    print(f"\n📈 Movement Phase Analysis:")
    
    # Phase 1: Small actions (steps 0-49)
    if len(action_mags) > 49:
        phase1_actions = action_mags[:50]
        phase1_rewards = rewards[:50]
        print(f"   Phase 1 (small actions): action={phase1_actions.mean():.3f}, reward={phase1_rewards.mean():.2f}")
    
    # Phase 2: Medium actions (steps 50-99)
    if len(action_mags) > 99:
        phase2_actions = action_mags[50:100]
        phase2_rewards = rewards[50:100]
        print(f"   Phase 2 (medium actions): action={phase2_actions.mean():.3f}, reward={phase2_rewards.mean():.2f}")
    
    # Phase 3: Large actions (steps 100+)
    if len(action_mags) > 100:
        phase3_actions = action_mags[100:]
        phase3_rewards = rewards[100:]
        print(f"   Phase 3 (large actions): action={phase3_actions.mean():.3f}, reward={phase3_rewards.mean():.2f}")
    
    # Workspace coverage
    if len(positions) > 10:
        positions_array = np.array(positions)
        workspace_span = np.ptp(positions_array, axis=0)  # Peak-to-peak (range)
        coverage = np.prod(workspace_span)  # Volume approximation
        print(f"   Workspace coverage: {coverage:.4f} m³")
        print(f"   Position range: X={workspace_span[0]:.3f}m, Y={workspace_span[1]:.3f}m, Z={workspace_span[2]:.3f}m")
    
    # Create visualization
    create_movement_visualization(action_mags, movements, rewards, velocities)


def create_movement_visualization(action_mags, movements, rewards, velocities):
    """Create visualization of movement dynamics"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        steps = np.arange(len(action_mags))
        
        # Plot 1: Action magnitudes over time
        axes[0,0].plot(steps, action_mags, alpha=0.7, color='blue')
        axes[0,0].axhline(y=0.3, color='orange', linestyle='--', alpha=0.7, label='Active threshold')
        axes[0,0].set_title('Action Magnitudes Over Time')
        axes[0,0].set_xlabel('Step')
        axes[0,0].set_ylabel('Action Magnitude')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Rewards over time
        axes[0,1].plot(steps, rewards, alpha=0.7, color='green')
        axes[0,1].set_title('Rewards Over Time')
        axes[0,1].set_xlabel('Step')
        axes[0,1].set_ylabel('Reward')
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Movement distances
        axes[1,0].plot(steps, movements, alpha=0.7, color='red')
        axes[1,0].axhline(y=0.03, color='orange', linestyle='--', alpha=0.7, label='Speed bonus threshold')
        axes[1,0].set_title('Movement Distances')
        axes[1,0].set_xlabel('Step')
        axes[1,0].set_ylabel('Distance (m)')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Action vs Reward correlation
        axes[1,1].scatter(action_mags, rewards, alpha=0.6, color='purple')
        axes[1,1].set_title('Action Magnitude vs Reward')
        axes[1,1].set_xlabel('Action Magnitude')
        axes[1,1].set_ylabel('Reward')
        axes[1,1].grid(True, alpha=0.3)
        
        # Add trend line
        if len(action_mags) > 10:
            z = np.polyfit(action_mags, rewards, 1)
            p = np.poly1d(z)
            axes[1,1].plot(sorted(action_mags), p(sorted(action_mags)), "r--", alpha=0.8)
        
        plt.tight_layout()
        plt.savefig('av_so101/test_logs/movement_dynamics_analysis.png')
        print(f"\n📊 Visualization saved: test_logs/movement_dynamics_analysis.png")
        
    except Exception as e:
        print(f"⚠️  Could not create visualization: {e}")


def main():
    """Run movement dynamics validation"""
    print("🚀 Movement Dynamics Validation Test")
    print("=" * 50)
    print("Testing improved joint velocities and reward structure")
    print("to ensure larger, more dynamic movements are encouraged.")
    
    success = test_movement_dynamics(num_steps=150)
    
    print("\n" + "=" * 50)
    if success:
        print("✅ Movement dynamics test completed!")
        print("🎯 Check analysis above for reward-action correlations.")
    else:
        print("❌ Movement dynamics test failed!")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

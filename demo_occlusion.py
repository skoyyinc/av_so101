#!/usr/bin/env python3
"""
Demo script for occlusion tracking - Interactive demonstration
"""

import numpy as np
import cv2
import time
from src.so_arm_occlusion_env import SO101OcclusionTrackingEnv
from src.occlusion_tracking_policy import OcclusionTrackingPolicy

def demo_occlusion_tracking():
    """Interactive demo of occlusion tracking"""
    print("🎯 SO-ARM101 Occlusion Tracking Demo")
    print("=" * 50)
    print("This demo shows the robot tracking a target object despite occlusion.")
    print("Press CTRL+C to stop the demo at any time.")
    print()
    
    # Create environment and policy
    env = SO101OcclusionTrackingEnv(render_mode="human")
    policy_config = {
        'p_gain_fast': 2.0,
        'p_gain_slow': 1.0,
        'max_velocity': 2.0
    }
    policy = OcclusionTrackingPolicy(policy_config)
    
    try:
        print("🔄 Initializing environment...")
        observation, info = env.reset()
        
        print(f"📊 Demo Setup:")
        print(f"   • Target: Small red sphere (radius: 0.04m)")
        print(f"   • Occlusion: Smaller gray box (0.015×0.015×0.05m)")
        print(f"   • Coverage: ~25% partial occlusion of target")
        print(f"   • Distance: Objects placed 0.4-0.6m from robot")
        print(f"   • Camera: Fixed orientation, 40° downward tilt")
        print(f"   • Deadzone: {policy.deadzone_radius:.3f} (robot slows when centered)")
        print(f"   • Occlusion threshold: {policy.occlusion_threshold:.3f}")
        print()
        
        print("🚀 Starting demo...")
        print("Watch the robot:")
        print("   - Track the red target object")
        print("   - Handle occlusion by the gray box")
        print("   - Search for clear viewpoints")
        print("   - Stop when target is centered and clear")
        print()
        
        # Demo loop
        step = 0
        last_status_time = time.time()
        demo_phases = []
        
        while True:
            # Get current state
            target_in_view = observation['target_in_view'][0]
            center_distance = observation['target_center_distance'][0]
            target_occluded = observation['target_occluded'][0]
            
            # Determine current phase
            if target_in_view < 0.5:
                phase = "🔍 SEARCHING"
            elif target_occluded > policy.occlusion_threshold:
                phase = "🚧 OCCLUDED"
            elif center_distance < policy.deadzone_radius:
                phase = "🎯 CENTERED"
            else:
                phase = "🎯 TRACKING"
            
            demo_phases.append(phase)
            
            # Get action from policy
            action = policy.predict(observation)
            action_magnitude = np.linalg.norm(action)
            
            # Execute action
            observation, reward, terminated, truncated, info = env.step(action)
            
            # Print periodic status
            current_time = time.time()
            if current_time - last_status_time >= 2.0:  # Every 2 seconds
                print(f"   Step {step:3d}: {phase} | "
                      f"Distance: {center_distance:.3f} | "
                      f"Occlusion: {target_occluded:.3f} | "
                      f"Action: {action_magnitude:.3f}")
                last_status_time = current_time
            
            # Save success image when centered (but don't stop)
            if (center_distance < policy.deadzone_radius and 
                target_occluded < 0.3 and 
                action_magnitude < 0.01 and
                step % 100 == 0):  # Save every 100 steps when centered
                success_image = observation['camera_image']
                cv2.imwrite(f"demo_centered_step_{step}.png", cv2.cvtColor(success_image, cv2.COLOR_RGB2BGR))
                print(f"   📸 Centered view saved as 'demo_centered_step_{step}.png'")
                print(f"   🎯 Target centered! Distance: {center_distance:.3f}, Occlusion: {target_occluded:.3f}")
                print("   ⏰ Demo continues running...")
            
            # Check termination conditions (extended timeout)
            if terminated or truncated or step > 1000:
                if step > 1000:
                    print("\\n⏱️ Demo timeout reached (1000 steps)")
                else:
                    print("\\n🏁 Demo completed")
                break
            
            step += 1
            time.sleep(0.1)  # 10 Hz update rate
            
    except KeyboardInterrupt:
        print("\\n🛑 Demo stopped by user")
    except Exception as e:
        print(f"\\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()
        print("\\n👋 Demo finished!")

def quick_occlusion_demo():
    """Quick demo showing just the setup"""
    print("🔧 Quick Occlusion Setup Demo")
    print("=" * 40)
    
    env = SO101OcclusionTrackingEnv(render_mode="human")
    
    try:
        observation, info = env.reset()
        
        print("📊 Environment created with:")
        print(f"   • Target position: {env.target_position}")
        print(f"   • Occlusion position: {env.occlusion_position}")
        print(f"   • Target in view: {observation['target_in_view'][0]:.3f}")
        print(f"   • Target occluded: {observation['target_occluded'][0]:.3f}")
        print(f"   • Center distance: {observation['target_center_distance'][0]:.3f}")
        
        # Save setup image
        setup_image = observation['camera_image']
        cv2.imwrite("demo_setup.png", cv2.cvtColor(setup_image, cv2.COLOR_RGB2BGR))
        print("   📸 Setup image saved as 'demo_setup.png'")
        
        print("\\n✅ Quick demo completed!")
        print("Run 'python demo_occlusion.py' for full interactive demo")
        
        # Wait a bit to see the setup
        time.sleep(5)
        
    except Exception as e:
        print(f"❌ Quick demo error: {e}")
    finally:
        env.close()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        quick_occlusion_demo()
    else:
        demo_occlusion_tracking()
#!/usr/bin/env python3
"""
Visual test script for SearchRLEnv - Phase 1 validation with GUI

This script provides visual validation of the RL search environment
using PyBullet GUI, similar to demo_with_controls.py but focused on
testing the RL search behavior.
"""

import sys
import os
import time
import numpy as np
import pybullet as p

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.rl import SearchRLEnv, ComprehensiveMetricsTracker


class VisualSearchEnvTest:
    """Visual test for SearchRLEnv with GUI controls"""
    
    def __init__(self):
        self.env = None
        self.tracker = None
        self.gui_ids = {}
        self.episode_count = 0
        
    def print_controls(self):
        """Print control instructions"""
        print("\n🎮 VISUAL SEARCH ENV TEST CONTROLS")
        print("=" * 50)
        print("🖱️  MOUSE CONTROLS:")
        print("   • Left-click + drag: Rotate camera view")
        print("   • Right-click + drag: Pan camera")
        print("   • Mouse wheel: Zoom in/out")
        
        print("\n⌨️  KEYBOARD SHORTCUTS:")
        print("   • R: Reset camera to default view")
        print("   • ESC/Q: Quit test")
        
        print("\n🎛️  GUI SLIDERS (will appear in PyBullet window):")
        print("   • Action Noise: Add noise to random actions")
        print("   • Episode Speed: Control test episode speed")
        print("   • Reset Episode: Start new search episode")
        
        print("\n📊 WHAT TO WATCH:")
        print("   • Red cube: Target object to find")
        print("   • Blue cubes: Distractor objects") 
        print("   • Robot arm: Will search randomly (no trained policy yet)")
        print("   • Green text: RL episode information")
        print("   • Console: Real-time metrics and episode results")
        
        print("=" * 50)
    
    def setup_gui(self):
        """Setup GUI controls for visual testing"""
        if not self.env:
            print("⚠️  Cannot setup GUI - environment not created")
            return
            
        print("🎮 Setting up visual test GUI...")
        
        try:
            # Control sliders
            self.gui_ids['noise_slider'] = p.addUserDebugParameter(
                "Action Noise", 0.0, 1.0, 0.3
            )
            
            self.gui_ids['speed_slider'] = p.addUserDebugParameter(
                "Episode Speed", 0.1, 3.0, 1.0
            )
            
            # Control buttons
            self.gui_ids['reset_button'] = p.addUserDebugParameter(
                "Reset Episode", 0, 1, 0
            )
            
            self.gui_ids['pause_button'] = p.addUserDebugParameter(
                "Pause Test", 0, 1, 0
            )
            
            print("✅ GUI controls added to PyBullet window")
            
        except Exception as e:
            print(f"⚠️  GUI setup failed: {e}")
            print("   Test will continue without GUI controls")
    
    def run_visual_test(self):
        """Run visual test of SearchRLEnv"""
        print("🚀 Starting Visual SearchRLEnv Test")
        self.print_controls()
        
        try:
            # Create RL search environment with visual rendering
            print("\n📡 Creating SearchRLEnv with GUI...")
            from src.so_arm_gym_env import SO101CameraTrackingEnv
            
            # Create base environment with GUI
            base_env = SO101CameraTrackingEnv(render_mode="human")
            
            # Wrap in RL environment with extremely fast movements for testing
            self.env = SearchRLEnv(
                base_env=base_env,
                max_search_steps=200,  # Shorter episodes for testing
                max_joint_velocities=[10.0, 8.0, 8.0, 6.0, 6.0, 10.0]  # Extremely fast for visual testing
            )
            print("✅ SearchRLEnv created successfully")
            
            # Create metrics tracker
            self.tracker = ComprehensiveMetricsTracker(log_dir="visual_test_logs")
            print("✅ MetricsTracker initialized")
            
            # Setup GUI after environment is ready
            self.setup_gui()
            
            # Add visual markers
            self._add_visual_markers()
            
            print("\n🎬 Visual test is now running in PyBullet GUI window!")
            print("   Watching RL search environment with random actions...")
            print("   Close PyBullet window or press Ctrl+C to stop")
            
            # Main test loop
            self._run_test_loop()
            
        except Exception as e:
            print(f"❌ Visual test error: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            if self.env:
                self.env.close()
            print("\n👋 Visual test ended!")
    
    def _run_test_loop(self):
        """Main visual test loop"""
        last_reset_state = 0
        gui_available = bool(self.gui_ids)
        
        # Start first episode
        obs, info = self.env.reset()
        episode_reward = 0.0
        episode_steps = 0
        episode_start_time = time.time()
        
        print(f"\n🎬 Episode {self.episode_count + 1} started!")
        
        while True:
            try:
                # Default values
                action_noise = 0.3
                episode_speed = 1.0
                pause_state = 0
                reset_state = 0
                
                # Read GUI parameters if available
                if gui_available:
                    try:
                        action_noise = p.readUserDebugParameter(self.gui_ids['noise_slider'])
                        episode_speed = p.readUserDebugParameter(self.gui_ids['speed_slider'])
                        pause_state = p.readUserDebugParameter(self.gui_ids['pause_button'])
                        reset_state = p.readUserDebugParameter(self.gui_ids['reset_button'])
                    except:
                        gui_available = False
                
                # Handle manual reset
                if reset_state > 0.5 and last_reset_state < 0.5:
                    print(f"\n🔄 Manual reset triggered...")
                    obs, info = self.env.reset()
                    episode_reward = 0.0
                    episode_steps = 0
                    episode_start_time = time.time()
                    self.episode_count += 1
                    print(f"🎬 Episode {self.episode_count + 1} started!")
                    self._add_visual_markers()
                
                last_reset_state = reset_state
                
                # Handle pause
                if pause_state > 0.5:
                    time.sleep(0.1)
                    continue
                
                # Generate action (random with noise for testing)
                base_action = self.env.action_space.sample()
                if action_noise > 0:
                    noise = np.random.normal(0, action_noise, size=base_action.shape)
                    action = np.clip(base_action + noise, -1.0, 1.0)
                else:
                    action = base_action
                
                # Take step in RL environment
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                episode_reward += reward
                episode_steps += 1
                
                # Update visual status
                self._update_visual_status(obs, info, episode_reward, episode_steps)
                
                # Print step info every 25 steps
                if episode_steps % 25 == 0:
                    search_time = time.time() - episode_start_time
                    print(f"   Step {episode_steps}: reward={reward:.2f}, "
                          f"total_reward={episode_reward:.1f}, time={search_time:.1f}s")
                
                # Handle episode end
                if terminated or truncated:
                    search_time = time.time() - episode_start_time
                    outcome = info.get('outcome', 'unknown')
                    
                    print(f"\n📊 Episode {self.episode_count + 1} finished!")
                    print(f"   Outcome: {outcome}")
                    print(f"   Total reward: {episode_reward:.1f}")
                    print(f"   Steps taken: {episode_steps}")
                    print(f"   Search time: {search_time:.1f}s")
                    
                    # Record in metrics tracker
                    episode_data = {
                        'total_reward': episode_reward,
                        'steps': episode_steps,
                        'outcome': outcome,
                        'search_time': search_time,
                        'workspace_coverage': info.get('exploration_coverage', 0) / 20.0,
                        'joint_violations': info.get('joint_violations', 0),
                        'collisions': 0,
                        'difficulty_level': 1,
                        'target_distance': 0.5
                    }
                    
                    self.tracker.update_episode(episode_data)
                    
                    # Print current performance summary
                    if self.episode_count % 3 == 0:  # Every 3 episodes
                        self._print_performance_summary()
                    
                    # Start new episode
                    time.sleep(1.0)  # Brief pause between episodes
                    obs, info = self.env.reset()
                    episode_reward = 0.0
                    episode_steps = 0
                    episode_start_time = time.time()
                    self.episode_count += 1
                    print(f"\n🎬 Episode {self.episode_count + 1} started!")
                    self._add_visual_markers()
                
                # Control test speed
                time.sleep(0.1 / episode_speed)
                
            except KeyboardInterrupt:
                break
    
    def _add_visual_markers(self):
        """Add visual markers to help understand the test"""
        try:
            # Add coordinate frame at origin
            p.addUserDebugLine([0, 0, 0], [0.1, 0, 0], [1, 0, 0], lineWidth=3)  # X-axis red
            p.addUserDebugLine([0, 0, 0], [0, 0.1, 0], [0, 1, 0], lineWidth=3)  # Y-axis green  
            p.addUserDebugLine([0, 0, 0], [0, 0, 0.1], [0, 0, 1], lineWidth=3)  # Z-axis blue
            
            # Add workspace boundary circle
            points = []
            for i in range(16):
                angle = 2 * np.pi * i / 16
                x = 0.6 * np.cos(angle)
                y = 0.6 * np.sin(angle)
                points.append([x, y, 0])
            
            for i in range(len(points)):
                next_i = (i + 1) % len(points)
                p.addUserDebugLine(points[i], points[next_i], [0.7, 0.7, 0.7], lineWidth=2)
                
        except Exception as e:
            print(f"⚠️  Could not add visual markers: {e}")
    
    def _update_visual_status(self, obs, info, episode_reward, episode_steps):
        """Update visual status display"""
        try:
            # Get RL state information
            target_found = obs is not None  # Simple check
            
            # Determine status color and text
            if info.get('outcome') == 'ongoing':
                if episode_steps < 50:
                    status_color = [0, 0.8, 1]  # Light blue for early search
                    status_text = f"RL SEARCH ACTIVE"
                else:
                    status_color = [1, 0.8, 0]  # Orange for extended search
                    status_text = f"EXTENDED SEARCH"
            else:
                if info.get('outcome') == 'success':
                    status_color = [0, 1, 0]  # Green for success
                    status_text = f"TARGET FOUND!"
                else:
                    status_color = [1, 0, 0]  # Red for timeout/failure
                    status_text = f"SEARCH FAILED"
            
            full_status_text = (
                f"Episode {self.episode_count + 1} | {status_text} | "
                f"Steps: {episode_steps} | Reward: {episode_reward:.1f}"
            )
            
            # Remove old status text
            if hasattr(self, '_status_text_id'):
                try:
                    p.removeUserDebugItem(self._status_text_id)
                except:
                    pass
            
            # Add new status text
            self._status_text_id = p.addUserDebugText(
                full_status_text,
                [0, 0, 0.9],
                textColorRGB=status_color,
                textSize=1.5
            )
            
        except Exception as e:
            # Silently fail if text display doesn't work
            pass
    
    def _print_performance_summary(self):
        """Print performance summary"""
        summary = self.tracker.get_performance_summary()
        
        print(f"\n📈 Performance Summary (Episodes {self.episode_count + 1}):")
        print(f"   Success Rate: {summary.get('overall_success_rate', 0):.3f}")
        print(f"   Mean Reward: {summary.get('mean_episode_reward', 0):.1f}")
        print(f"   Mean Search Time: {summary.get('mean_search_time', 0):.1f}s")
        print(f"   Safety Violations: {summary.get('avg_joint_violations', 0):.2f}/episode")


def main():
    """Run visual test"""
    print("🎬 SearchRLEnv Visual Test")
    print("=" * 50)
    print("This test runs the RL search environment with PyBullet GUI")
    print("to visually verify that everything is working correctly.")
    print("The robot will perform random search actions (no trained policy yet).")
    
    demo = VisualSearchEnvTest()
    demo.run_visual_test()


if __name__ == "__main__":
    main()
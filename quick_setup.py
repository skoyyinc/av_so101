#!/usr/bin/env python3
"""
Quick setup for SO-ARM101 object tracking
"""

import subprocess
import sys
from pathlib import Path

def main():
   print("🚀 Quick SO-ARM101 Setup")
   print("=" * 30)
   
   # Step 1: Create directories
   print("\n📁 Creating directories...")
   Path("urdf/assets").mkdir(parents=True, exist_ok=True)
   print("✅ Created urdf/assets/")
   
   # Step 2: Setup URDF files
   print("\n🔧 Setting up URDF files...")
   try:
       from src.setup_manual_so101 import setup_manual_so101
       setup_manual_so101()
   except Exception as e:
       print(f"❌ Setup failed: {e}")
       return
   
   # Step 3: Test basic functionality
   print("\n🧪 Testing basic functionality...")
   try:
       test_passed = run_basic_test()
       if test_passed:
           print("✅ Basic test passed!")
       else:
           print("⚠️  Basic test had issues but should still work")
   except Exception as e:
       print(f"❌ Test failed: {e}")
   
   # Step 4: Instructions for next steps
   print(f"\n🎯 Next Steps:")
   print(f"1. Run full test: python test_tracking.py")
   print(f"2. Run demo: python demo_object_tracking.py")
   print(f"3. For better visuals, download STL files to urdf/assets/")

def run_basic_test():
   """Run basic functionality test"""
   try:
       from src.so_arm_gym_env import SO101CameraTrackingEnv
       from src.tracking_policy import ImprovedTrackingPolicy
       
       # Test environment creation
       env = SO101CameraTrackingEnv(render_mode="rgb_array")
       observation, info = env.reset()
       
       # Test policy
       policy = ImprovedTrackingPolicy({'p_gain_fast': 0.8})
       action = policy.predict(observation)
       
       # Test one step
       observation, reward, terminated, truncated, info = env.step(action)
       
       env.close()
       return True
       
   except Exception as e:
       print(f"Test error: {e}")
       return False

if __name__ == "__main__":
   main()
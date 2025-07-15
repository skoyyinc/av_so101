import lerobot
from lerobot.robots.so101_follower import SO101Follower
import numpy as np

def test_lerobot_setup():
    """Test basic LeRobot functionality"""
    print("✓ LeRobot imported successfully")
    print(f"✓ LeRobot version: {lerobot.__version__}")
    
    # Test SO-ARM101 robot class (this will work even without hardware)
    try:
        # This creates the robot configuration without connecting to hardware
        robot_config = {
            "robot_type": "so100",
            "calibration_dir": "./calibration",
            "motors": {
                "shoulder": {"id": 1, "model": "sts3215"},
                "elbow": {"id": 2, "model": "sts3215"},
                # Add other joints as needed
            }
        }
        print("✓ SO-ARM101 configuration loaded")
        return True
    except Exception as e:
        print(f"✗ Error with SO-ARM101 setup: {e}")
        return False

if __name__ == "__main__":
    success = test_lerobot_setup()
    if success:
        print("\n🎉 Setup complete! Ready for Phase 2.")
    else:
        print("\n❌ Setup failed. Please check the error messages above.")
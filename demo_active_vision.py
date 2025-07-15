#!/usr/bin/env python3
"""
Active Vision Demo for SO-ARM101 with improved error handling
"""

import cv2
import numpy as np
import time
import sys
import threading
from src.active_vision_system import ActiveVisionSystem

def check_opencv_display():
    """Check if OpenCV display is available"""
    try:
        # Test if we can create a window
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imshow('test', test_img)
        cv2.waitKey(1)
        cv2.destroyWindow('test')
        return True
    except Exception as e:
        print(f"⚠️  OpenCV display not available: {e}")
        return False

def main():
    print("🤖 SO-ARM101 Active Vision Demo")
    print("=" * 50)
    
    # Check display availability
    display_available = check_opencv_display()
    
    if not display_available:
        print("🔄 Falling back to headless mode...")
        from src.headless_demo import main as headless_main
        headless_main()
        return
    
    # System configuration
    config = {
        'robot_type': 'so_arm101',
        'simulation_mode': True,
        'camera_config': {
            'resolution': (640, 480),
            'fps': 30
        }
    }
    
    # Initialize active vision system
    try:
        av_system = ActiveVisionSystem(config)
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        print("🔄 Trying headless mode...")
        from src.headless_demo import main as headless_main
        headless_main()
        return
    
    print("\n📋 Demo Instructions:")
    print("- Press 'q' to quit")
    print("- Press 'c' to switch camera manually")  
    print("- Press 's' to show statistics")
    print("- Press 'h' to switch to headless mode")
    
    try:
        # Start active vision in separate thread
        vision_thread = threading.Thread(target=av_system.start_active_vision)
        vision_thread.daemon = True
        vision_thread.start()
        
        print("✅ Active vision started! Watch the display window...")
        
        # Main interaction loop
        start_time = time.time()
        while True:
            if display_available:
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    # Manual camera switch
                    current_cam = av_system.camera_controller.active_camera_id
                    next_cam = (current_cam + 1) % len(av_system.camera_controller.cameras)
                    av_system.camera_controller.set_active_camera(next_cam)
                    print(f"📷 Manually switched to camera {next_cam}")
                elif key == ord('s'):
                    # Show statistics
                    metrics = av_system.get_performance_metrics()
                    runtime = time.time() - start_time
                    print(f"\n📊 Performance Metrics (Runtime: {runtime:.1f}s):")
                    for key, value in metrics.items():
                        print(f"  {key}: {value}")
                elif key == ord('h'):
                    # Switch to headless
                    print("🔄 Switching to headless mode...")
                    av_system.stop()
                    from src.headless_demo import main as headless_main
                    headless_main()
                    return
            else:
                time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
    finally:
        av_system.stop()
        print("\n👋 Demo completed!")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Test script to validate improved target placement in SearchRLEnv
Verifies that targets are only placed in side zones (30° to 150°)
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.rl import SearchRLEnv


def test_target_placement_distribution(num_tests=100):
    """Test target placement distribution to ensure no behind-robot placement"""
    print(f"🧪 Testing target placement distribution ({num_tests} trials)...")
    
    try:
        # Create search environment
        env = SearchRLEnv(max_search_steps=10)  # Short episodes for testing
        
        angles = []
        distances = []
        positions = []
        
        for i in range(num_tests):
            # Reset environment (triggers target placement)
            obs, info = env.reset()
            
            # Get target position
            target_pos = env.base_env.target_position
            x, y, z = target_pos
            
            # Calculate angle from robot front (+X axis)
            angle_rad = np.arctan2(y, x)
            if angle_rad < 0:
                angle_rad += 2 * np.pi  # Convert to [0, 2π]
            
            angle_deg = np.degrees(angle_rad)
            distance = np.sqrt(x**2 + y**2)
            
            angles.append(angle_deg)
            distances.append(distance)
            positions.append([x, y, z])
            
            # Print every 20th trial
            if (i + 1) % 20 == 0:
                print(f"   Trial {i+1}: angle={angle_deg:.1f}°, distance={distance:.2f}m")
        
        env.close()
        
        # Analyze results
        angles = np.array(angles)
        distances = np.array(distances)
        positions = np.array(positions)
        
        print(f"\n📊 Target Placement Analysis:")
        print(f"   Total trials: {num_tests}")
        print(f"   Angle range: {angles.min():.1f}° to {angles.max():.1f}°")
        print(f"   Distance range: {distances.min():.2f}m to {distances.max():.2f}m")
        print(f"   Height range: {positions[:, 2].min():.2f}m to {positions[:, 2].max():.2f}m")
        
        # Check constraints
        issues = []
        
        # 1. Check if any targets in front cone (0° to 30° or 330° to 360°)
        front_cone_violations = np.sum((angles <= 30) | (angles >= 330))
        if front_cone_violations > 0:
            issues.append(f"❌ {front_cone_violations} targets in front cone (0°-30°, 330°-360°)")
        else:
            print("   ✅ No targets in front cone")
        
        # 2. Check if any targets behind robot (150° to 210°)
        back_violations = np.sum((angles >= 150) & (angles <= 210))
        if back_violations > 0:
            issues.append(f"❌ {back_violations} targets behind robot (150°-210°)")
        else:
            print("   ✅ No targets behind robot")
        
        # 3. Check if any targets in far back zones (210° to 330°)
        far_back_violations = np.sum((angles >= 210) & (angles <= 330))
        if far_back_violations > 0:
            issues.append(f"❌ {far_back_violations} targets in far back zones (210°-330°)")
        else:
            print("   ✅ No targets in far back zones")
        
        # 4. Check if all targets are in desired side zones (30° to 150°)
        side_zone_count = np.sum((angles >= 30) & (angles <= 150))
        if side_zone_count == num_tests:
            print(f"   ✅ All {num_tests} targets in side zones (30°-150°)")
        else:
            issues.append(f"❌ Only {side_zone_count}/{num_tests} targets in side zones")
        
        # 5. Check distance constraints
        too_close = np.sum(distances < 0.35)
        if too_close > 0:
            issues.append(f"❌ {too_close} targets too close to robot (<0.35m)")
        else:
            print("   ✅ All targets at safe distance")
        
        # Summary
        if not issues:
            print(f"\n✅ ALL TESTS PASSED! Target placement is working correctly.")
            print(f"   All {num_tests} targets placed in side zones only (30°-150°)")
            success = True
        else:
            print(f"\n❌ ISSUES FOUND:")
            for issue in issues:
                print(f"   {issue}")
            success = False
        
        # Create visualization
        create_placement_visualization(angles, distances, positions)
        
        return success
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_placement_visualization(angles, distances, positions):
    """Create visualization of target placements"""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Polar plot of target positions
        angles_rad = np.radians(angles)
        ax1.set_projection('polar')
        ax1.scatter(angles_rad, distances, alpha=0.6, c='red', s=30)
        ax1.set_title('Target Positions (Polar View)')
        ax1.set_ylim(0, 1.0)
        
        # Highlight forbidden zones
        forbidden_angles = np.linspace(0, 2*np.pi, 360)
        
        # Front cone (0° to 30°, 330° to 360°)
        front_mask = (forbidden_angles <= np.pi/6) | (forbidden_angles >= 11*np.pi/6)
        ax1.fill_between(forbidden_angles[front_mask], 0, 1, alpha=0.3, color='orange', label='Front cone (forbidden)')
        
        # Back zones (150° to 210°)
        back_mask = (forbidden_angles >= 5*np.pi/6) & (forbidden_angles <= 7*np.pi/6)
        ax1.fill_between(forbidden_angles[back_mask], 0, 1, alpha=0.3, color='gray', label='Back zone (forbidden)')
        
        ax1.legend()
        
        # Plot 2: Histogram of angles
        ax2.hist(angles, bins=36, range=(0, 360), alpha=0.7, color='red', edgecolor='black')
        ax2.axvspan(0, 30, alpha=0.3, color='orange', label='Front cone')
        ax2.axvspan(330, 360, alpha=0.3, color='orange')
        ax2.axvspan(150, 210, alpha=0.3, color='gray', label='Back zone')
        ax2.axvspan(210, 330, alpha=0.3, color='lightgray', label='Far back')
        
        ax2.set_xlabel('Angle (degrees)')
        ax2.set_ylabel('Count')
        ax2.set_title('Target Angle Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('av_so101/test_logs/target_placement_validation.png')
        print(f"\n📊 Visualization saved: test_logs/target_placement_validation.png")
        
    except Exception as e:
        print(f"⚠️  Could not create visualization: {e}")


def main():
    """Run target placement validation"""
    print("🎯 Target Placement Validation Test")
    print("=" * 50)
    print("Testing that targets are only placed in side zones (30° to 150°)")
    print("and never in front cone or behind robot.")
    
    success = test_target_placement_distribution(num_tests=50)
    
    print("\n" + "=" * 50)
    if success:
        print("✅ Target placement validation PASSED!")
        print("🎯 Phase 1 adjustment completed successfully.")
    else:
        print("❌ Target placement validation FAILED!")
        print("🔧 Further adjustments needed.")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

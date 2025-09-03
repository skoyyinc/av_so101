#!/usr/bin/env python3
"""
Visualize target position to verify placement

This script creates a top-down view diagram showing where the target
cube is positioned relative to the robot.
"""

import numpy as np
import matplotlib.pyplot as plt

def visualize_target_position():
    """Create a top-down view of robot and target position"""
    
    # Calculate target position
    angle = np.radians(75)   # 75° from front (left side)
    distance = 0.75          # 0.75m from robot
    
    target_x = distance * np.cos(angle)
    target_y = distance * np.sin(angle)
    
    print(f"Target position: x={target_x:.3f}, y={target_y:.3f}")
    print(f"Angle: {np.degrees(angle):.1f}°")
    print(f"Distance: {distance:.3f}m")
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Robot at origin (facing +X direction)
    robot_size = 0.1
    robot = plt.Circle((0, 0), robot_size, color='blue', alpha=0.7, label='Robot')
    ax.add_patch(robot)
    
    # Robot front direction arrow
    ax.arrow(0, 0, 0.3, 0, head_width=0.05, head_length=0.05, fc='blue', ec='blue')
    ax.text(0.35, 0, 'FRONT', ha='left', va='center', fontsize=12, fontweight='bold')
    
    # Target cube
    target = plt.Circle((target_x, target_y), 0.05, color='red', alpha=0.8, label='Target Cube')
    ax.add_patch(target)
    
    # Target position line
    ax.plot([0, target_x], [0, target_y], 'r--', alpha=0.5, linewidth=2)
    
    # Angle arc
    angle_arc = np.linspace(0, angle, 50)
    arc_radius = 0.2
    arc_x = arc_radius * np.cos(angle_arc)
    arc_y = arc_radius * np.sin(angle_arc)
    ax.plot(arc_x, arc_y, 'g-', linewidth=2)
    ax.text(arc_radius * 0.7 * np.cos(angle/2), arc_radius * 0.7 * np.sin(angle/2), 
            f'{np.degrees(angle):.0f}°', ha='center', va='center', fontsize=10, color='green')
    
    # Distance annotation
    mid_x, mid_y = target_x/2, target_y/2
    ax.text(mid_x, mid_y + 0.1, f'{distance:.2f}m', ha='center', va='bottom', 
            fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # Workspace circle
    workspace_circle = plt.Circle((0, 0), 0.8, fill=False, color='gray', linestyle=':', alpha=0.5)
    ax.add_patch(workspace_circle)
    ax.text(0.6, 0.6, 'Workspace\nBoundary', ha='center', va='center', fontsize=8, color='gray')
    
    # Coordinate system
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Labels and formatting
    ax.set_xlim(-0.5, 1.0)
    ax.set_ylim(-0.5, 1.0)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title('SO101 Robot - Target Position (Top View)\nStatic Target: 75° Left Side, 0.75m Distance', fontsize=14)
    ax.legend()
    
    # Add directional labels
    ax.text(0.8, 0.05, 'RIGHT', ha='center', va='center', fontsize=10, rotation=0, color='gray')
    ax.text(0.05, 0.8, 'LEFT', ha='center', va='center', fontsize=10, rotation=90, color='gray')
    ax.text(0.05, -0.4, 'BACK', ha='center', va='center', fontsize=10, rotation=0, color='gray')
    
    # Save and show
    plt.tight_layout()
    plt.savefig('target_position_diagram.png', dpi=150, bbox_inches='tight')
    print("\n📊 Diagram saved as 'target_position_diagram.png'")
    plt.show()

def compare_angles():
    """Compare different angle positions"""
    
    angles = [30, 45, 75, 90, 110, 135]  # Different angles to compare
    distance = 0.75
    
    print("Angle comparison (0° = robot front):")
    print("=" * 50)
    
    for angle_deg in angles:
        angle_rad = np.radians(angle_deg)
        x = distance * np.cos(angle_rad)
        y = distance * np.sin(angle_rad)
        
        # Determine position description
        if 0 <= angle_deg <= 45:
            desc = "Front-right"
        elif 45 < angle_deg <= 90:
            desc = "Right side"
        elif 90 < angle_deg <= 135:
            desc = "Back-right"
        elif 135 < angle_deg <= 180:
            desc = "Back side"
        else:
            desc = "Unknown"
        
        print(f"{angle_deg:3d}°: [{x:6.3f}, {y:6.3f}] - {desc}")
    
    print("=" * 50)
    print("✅ 75° is correct for left side (slightly forward)")
    print("❌ 110° was back-left (behind robot)")

if __name__ == "__main__":
    print("🎯 Target Position Visualization")
    print("=" * 50)
    
    compare_angles()
    print("\nGenerating visual diagram...")
    visualize_target_position()

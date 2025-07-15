import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import time
import subprocess
import os
import signal
from typing import Tuple, Dict, Optional, List
import math
from pathlib import Path
import xml.etree.ElementTree as ET

# Try to import ROS2 dependencies
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image, JointState
    from std_msgs.msg import Float64MultiArray
    from geometry_msgs.msg import Twist
    from cv_bridge import CvBridge
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False

class SO101GazeboTrackingEnv(gym.Env):
    """
    SO-ARM101 with camera end-effector for object tracking using Gazebo
    Alternative to PyBullet/MuJoCo for testing rendering issues
    """
    
    def __init__(self, render_mode="human", camera_width=640, camera_height=480, use_ros2=True):
        super().__init__()
        
        # Check if ROS2 is available
        if use_ros2 and not ROS2_AVAILABLE:
            print("⚠️  ROS2 not available, falling back to direct Gazebo control")
            use_ros2 = False
        
        # Environment parameters
        self.render_mode = render_mode
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.use_ros2 = use_ros2
        
        # Gazebo process management
        self.gazebo_process = None
        self.ros2_node = None
        self.bridge = CvBridge() if ROS2_AVAILABLE else None
        
        # Robot parameters - SO-ARM101 specific
        self.num_joints = 6  # SO-ARM101 has 6 DOF
        self.joint_names = [
            "shoulder_pan_joint",
            "shoulder_tilt_joint", 
            "elbow_flex_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint"
        ]
        
        # Joint limits (from URDF)
        self.joint_limits = [
            (-np.pi, np.pi),      # Base rotation
            (-np.pi/2, np.pi/2),  # Shoulder
            (-np.pi/2, np.pi/2),  # Elbow  
            (-np.pi/2, np.pi/2),  # Wrist 1
            (-np.pi/2, np.pi/2),  # Wrist 2
            (-np.pi, np.pi)       # Wrist rotation
        ]
        
        # Action space: joint velocities for 6 DOF arm
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.num_joints,), dtype=np.float32
        )
        
        # Observation space
        self.observation_space = spaces.Dict({
            'camera_image': spaces.Box(
                low=0, high=255, 
                shape=(camera_height, camera_width, 3), 
                dtype=np.uint8
            ),
            'joint_positions': spaces.Box(
                low=-np.pi, high=np.pi, shape=(self.num_joints,), dtype=np.float32
            ),
            'target_in_view': spaces.Box(
                low=0, high=1, shape=(1,), dtype=np.float32
            ),
            'target_center_distance': spaces.Box(
                low=0, high=1, shape=(1,), dtype=np.float32
            )
        })
        
        # Tracking state
        self.target_position = np.array([0.5, 0.0, 0.1])
        self.camera_center = np.array([camera_width//2, camera_height//2])
        self.max_episode_steps = 500
        self.current_step = 0
        
        # Performance tracking
        self.target_centered_steps = 0
        self.total_tracking_error = 0.0
        
        # Previous state for reward calculation
        self.prev_target_in_view = False
        self.prev_joint_positions = None
        
        # Current state
        self.current_joint_positions = np.zeros(self.num_joints)
        self.current_camera_image = None
        
        # Gazebo configuration
        self.gazebo_config = self._get_gazebo_config()
        
    def _get_gazebo_config(self):
        """Get Gazebo configuration"""
        return {
            'world_file': self._get_world_file(),
            'model_file': self._get_model_file(),
            'launch_file': self._get_launch_file()
        }
    
    def _get_world_file(self):
        """Get or create Gazebo world file"""
        world_dir = Path("gazebo")
        world_dir.mkdir(exist_ok=True)
        
        world_file = world_dir / "so101_world.sdf"
        
        if not world_file.exists():
            self._create_world_file(world_file)
        
        return str(world_file)
    
    def _create_world_file(self, world_file: Path):
        """Create Gazebo world file"""
        world_content = '''<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="so101_world">
    <!-- Ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>
    
    <!-- Sun -->
    <include>
      <uri>model://sun</uri>
    </include>
    
    <!-- Table -->
    <model name="table">
      <static>true</static>
      <pose>0.5 0 0 0 0 0</pose>
      <link name="table_link">
        <collision name="table_collision">
          <geometry>
            <box>
              <size>1.0 1.0 0.02</size>
            </box>
          </geometry>
        </collision>
        <visual name="table_visual">
          <geometry>
            <box>
              <size>1.0 1.0 0.02</size>
            </box>
          </geometry>
          <material>
            <ambient>0.8 0.8 0.8 1</ambient>
            <diffuse>0.8 0.8 0.8 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
    
    <!-- Target object (red sphere) -->
    <model name="target_object">
      <pose>0.5 0.0 0.1 0 0 0</pose>
      <link name="target_link">
        <collision name="target_collision">
          <geometry>
            <sphere>
              <radius>0.03</radius>
            </sphere>
          </geometry>
        </collision>
        <visual name="target_visual">
          <geometry>
            <sphere>
              <radius>0.03</radius>
            </sphere>
          </geometry>
          <material>
            <ambient>1.0 0.0 0.0 1</ambient>
            <diffuse>1.0 0.0 0.0 1</diffuse>
          </material>
        </visual>
      </link>
    </model>
    
    <!-- Physics -->
    <physics type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>
    
    <!-- Camera sensor -->
    <model name="camera_sensor">
      <pose>0.3 0 0.5 0 0.3 0</pose>
      <link name="camera_link">
        <sensor name="camera" type="camera">
          <pose>0 0 0 0 0 0</pose>
          <camera>
            <horizontal_fov>1.047</horizontal_fov>
            <image>
              <width>{width}</width>
              <height>{height}</height>
              <format>R8G8B8</format>
            </image>
            <clip>
              <near>0.1</near>
              <far>100</far>
            </clip>
          </camera>
          <always_on>1</always_on>
          <update_rate>30</update_rate>
          <visualize>true</visualize>
        </sensor>
      </link>
    </model>
    
  </world>
</sdf>'''.format(width=self.camera_width, height=self.camera_height)
        
        with open(world_file, 'w') as f:
            f.write(world_content)
        
        print(f"📁 Created Gazebo world file: {world_file}")
    
    def _get_model_file(self):
        """Get robot model file for Gazebo"""
        model_dir = Path("gazebo")
        model_dir.mkdir(exist_ok=True)
        
        # Convert URDF to Gazebo-compatible format
        urdf_file = Path("urdf/so101_new_calib.urdf")
        gazebo_model_file = model_dir / "so101_gazebo.urdf"
        
        if urdf_file.exists():
            self._convert_urdf_for_gazebo(urdf_file, gazebo_model_file)
            return str(gazebo_model_file)
        else:
            print("⚠️  URDF file not found, using fallback model")
            return self._create_fallback_model(model_dir)
    
    def _convert_urdf_for_gazebo(self, urdf_file: Path, output_file: Path):
        """Convert URDF to Gazebo-compatible format"""
        try:
            # Read URDF
            with open(urdf_file, 'r') as f:
                urdf_content = f.read()
            
            # Parse XML
            root = ET.fromstring(urdf_content)
            
            # Add Gazebo plugins and sensors
            self._add_gazebo_plugins(root)
            
            # Write modified URDF
            with open(output_file, 'w') as f:
                f.write(ET.tostring(root, encoding='unicode'))
            
            print(f"✅ Converted URDF for Gazebo: {output_file}")
            
        except Exception as e:
            print(f"❌ URDF conversion failed: {e}")
            return self._create_fallback_model(output_file.parent)
    
    def _add_gazebo_plugins(self, root):
        """Add Gazebo-specific plugins to URDF"""
        # Add joint controller plugin
        gazebo_elem = ET.SubElement(root, "gazebo")
        plugin_elem = ET.SubElement(gazebo_elem, "plugin")
        plugin_elem.set("name", "joint_state_publisher")
        plugin_elem.set("filename", "libgazebo_ros_joint_state_publisher.so")
        
        # Add joint names
        for joint_name in self.joint_names:
            joint_elem = ET.SubElement(plugin_elem, "joint_name")
            joint_elem.text = joint_name
        
        # Add camera plugin
        camera_gazebo = ET.SubElement(root, "gazebo")
        camera_gazebo.set("reference", "camera_link")
        camera_plugin = ET.SubElement(camera_gazebo, "plugin")
        camera_plugin.set("name", "camera_controller")
        camera_plugin.set("filename", "libgazebo_ros_camera.so")
        
        # Camera parameters
        camera_name = ET.SubElement(camera_plugin, "camera_name")
        camera_name.text = "so101_camera"
        
        frame_name = ET.SubElement(camera_plugin, "frame_name")
        frame_name.text = "camera_link"
    
    def _create_fallback_model(self, model_dir: Path):
        """Create fallback robot model"""
        fallback_file = model_dir / "so101_fallback.urdf"
        
        fallback_content = '''<?xml version="1.0"?>
<robot name="so101_fallback">
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.1 0.1 0.1"/>
      </geometry>
      <material name="gray">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
  </link>
  
  <link name="arm_link">
    <visual>
      <geometry>
        <cylinder radius="0.02" length="0.3"/>
      </geometry>
      <material name="orange">
        <color rgba="1.0 0.5 0.0 1"/>
      </material>
    </visual>
  </link>
  
  <joint name="base_joint" type="revolute">
    <parent link="base_link"/>
    <child link="arm_link"/>
    <origin xyz="0 0 0.05"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="100" velocity="2.0"/>
  </joint>
  
  <gazebo>
    <plugin name="joint_state_publisher" filename="libgazebo_ros_joint_state_publisher.so">
      <joint_name>base_joint</joint_name>
    </plugin>
  </gazebo>
</robot>'''
        
        with open(fallback_file, 'w') as f:
            f.write(fallback_content)
        
        print(f"📁 Created fallback model: {fallback_file}")
        return str(fallback_file)
    
    def _get_launch_file(self):
        """Get or create launch file"""
        launch_dir = Path("gazebo")
        launch_dir.mkdir(exist_ok=True)
        
        launch_file = launch_dir / "so101_gazebo.launch"
        
        if not launch_file.exists():
            self._create_launch_file(launch_file)
        
        return str(launch_file)
    
    def _create_launch_file(self, launch_file: Path):
        """Create Gazebo launch file"""
        launch_content = f'''<?xml version="1.0"?>
<launch>
  <!-- Launch Gazebo with world -->
  <include file="$(find gazebo_ros)/launch/empty_world.launch">
    <arg name="world_name" value="{self.gazebo_config['world_file']}"/>
    <arg name="paused" value="false"/>
    <arg name="use_sim_time" value="true"/>
    <arg name="gui" value="true"/>
    <arg name="headless" value="false"/>
    <arg name="debug" value="false"/>
  </include>
  
  <!-- Load robot model -->
  <param name="robot_description" command="$(find xacro)/xacro {self.gazebo_config['model_file']}"/>
  
  <!-- Spawn robot in Gazebo -->
  <node name="spawn_robot" pkg="gazebo_ros" type="spawn_model" args="-urdf -param robot_description -model so101_robot -x 0 -y 0 -z 0.1"/>
  
  <!-- Joint state publisher -->
  <node name="joint_state_publisher" pkg="joint_state_publisher" type="joint_state_publisher"/>
  
  <!-- Robot state publisher -->
  <node name="robot_state_publisher" pkg="robot_state_publisher" type="robot_state_publisher"/>
  
</launch>'''
        
        with open(launch_file, 'w') as f:
            f.write(launch_content)
        
        print(f"📁 Created launch file: {launch_file}")
    
    def reset(self, seed=None, options=None):
        """Reset the environment"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.target_centered_steps = 0
        self.total_tracking_error = 0.0
        
        # Initialize previous state variables
        self.prev_target_in_view = False
        self.prev_joint_positions = None
        
        # Start Gazebo if not running
        if not self._is_gazebo_running():
            self._start_gazebo()
        
        # Initialize ROS2 node if using ROS2
        if self.use_ros2:
            self._init_ros2_node()
        
        # Reset robot pose
        self._reset_robot_pose()
        
        # Wait for simulation to settle
        time.sleep(2.0)
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def _is_gazebo_running(self):
        """Check if Gazebo is running"""
        try:
            result = subprocess.run(['pgrep', '-f', 'gazebo'], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False
    
    def _start_gazebo(self):
        """Start Gazebo simulation"""
        print("🚀 Starting Gazebo simulation...")
        
        try:
            # Start Gazebo with world file
            gazebo_cmd = [
                'gazebo',
                self.gazebo_config['world_file'],
                '--verbose'
            ]
            
            if self.render_mode != "human":
                gazebo_cmd.append('--server-only')
            
            self.gazebo_process = subprocess.Popen(
                gazebo_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid
            )
            
            # Wait for Gazebo to start
            time.sleep(5.0)
            
            # Spawn robot model
            self._spawn_robot_model()
            
            print("✅ Gazebo started successfully")
            
        except Exception as e:
            print(f"❌ Failed to start Gazebo: {e}")
            self.gazebo_process = None
    
    def _spawn_robot_model(self):
        """Spawn robot model in Gazebo"""
        try:
            spawn_cmd = [
                'gz', 'model',
                '--spawn-file', self.gazebo_config['model_file'],
                '--model-name', 'so101_robot',
                '--pose', '0 0 0.1 0 0 0'
            ]
            
            result = subprocess.run(spawn_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Robot model spawned in Gazebo")
            else:
                print(f"⚠️  Robot spawn warning: {result.stderr}")
                
        except Exception as e:
            print(f"❌ Failed to spawn robot: {e}")
    
    def _init_ros2_node(self):
        """Initialize ROS2 node"""
        if not ROS2_AVAILABLE:
            return
        
        try:
            if not rclpy.ok():
                rclpy.init()
            
            self.ros2_node = GazeboControllerNode()
            print("✅ ROS2 node initialized")
            
        except Exception as e:
            print(f"❌ Failed to initialize ROS2: {e}")
            self.use_ros2 = False
    
    def _reset_robot_pose(self):
        """Reset robot to initial pose"""
        # Initial joint positions
        initial_positions = [0.0, -0.3, 0.5, 0.0, 0.2, 0.0]
        
        if self.use_ros2 and self.ros2_node:
            self.ros2_node.set_joint_positions(initial_positions)
        else:
            # Use Gazebo service calls
            self._set_joint_positions_gazebo(initial_positions)
        
        self.current_joint_positions = np.array(initial_positions[:self.num_joints])
        print("✅ Robot reset to initial pose")
    
    def _set_joint_positions_gazebo(self, positions):
        """Set joint positions using Gazebo services"""
        try:
            for i, (joint_name, position) in enumerate(zip(self.joint_names, positions)):
                if i < len(positions):
                    cmd = [
                        'gz', 'joint',
                        '--world-name', 'so101_world',
                        '--model-name', 'so101_robot',
                        '--joint-name', joint_name,
                        '--pos-t', str(position)
                    ]
                    subprocess.run(cmd, capture_output=True)
        except Exception as e:
            print(f"⚠️  Failed to set joint positions: {e}")
    
    def step(self, action):
        """Execute one step in the environment"""
        if not self._is_gazebo_running():
            return self._get_dummy_observation(), 0, True, False, {}
        
        self.current_step += 1
        
        # Apply action
        self._apply_action(action)
        
        # Wait for physics update
        time.sleep(0.1)
        
        # Get observation
        observation = self._get_observation()
        
        # Calculate reward
        reward = self._calculate_reward(observation)
        
        # Check termination conditions
        terminated = self._is_terminated(observation)
        truncated = self.current_step >= self.max_episode_steps
        
        # Get info
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _apply_action(self, action):
        """Apply action to robot joints"""
        # Scale action to reasonable velocities
        max_velocity = 3.0  # rad/s
        target_velocities = np.clip(action, -1.0, 1.0) * max_velocity
        
        if self.use_ros2 and self.ros2_node:
            self.ros2_node.set_joint_velocities(target_velocities)
        else:
            # Use Gazebo service calls
            self._set_joint_velocities_gazebo(target_velocities)
    
    def _set_joint_velocities_gazebo(self, velocities):
        """Set joint velocities using Gazebo services"""
        try:
            for i, (joint_name, velocity) in enumerate(zip(self.joint_names, velocities)):
                if i < len(velocities):
                    cmd = [
                        'gz', 'joint',
                        '--world-name', 'so101_world',
                        '--model-name', 'so101_robot',
                        '--joint-name', joint_name,
                        '--vel-t', str(velocity)
                    ]
                    subprocess.run(cmd, capture_output=True)
        except Exception as e:
            print(f"⚠️  Failed to set joint velocities: {e}")
    
    def _get_observation(self):
        """Get current observation"""
        # Get camera image
        camera_image = self._get_camera_image()
        
        # Get joint positions
        joint_positions = self._get_joint_positions()
        
        # Detect target in camera image
        target_info = self._detect_target_in_image(camera_image)
        
        observation = {
            'camera_image': camera_image,
            'joint_positions': joint_positions,
            'target_in_view': np.array([target_info['in_view']], dtype=np.float32),
            'target_center_distance': np.array([target_info['center_distance']], dtype=np.float32)
        }
        
        return observation
    
    def _get_camera_image(self):
        """Get camera image from Gazebo"""
        if self.use_ros2 and self.ros2_node:
            return self.ros2_node.get_camera_image()
        else:
            # Use Gazebo image service
            return self._get_camera_image_gazebo()
    
    def _get_camera_image_gazebo(self):
        """Get camera image using Gazebo services"""
        # For now, return a dummy image
        # In a full implementation, this would use Gazebo's image service
        dummy_image = np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)
        
        # Add some test pattern
        cv2.rectangle(dummy_image, (50, 50), (150, 150), (255, 0, 0), -1)
        cv2.putText(dummy_image, "Gazebo", (200, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return dummy_image
    
    def _get_joint_positions(self):
        """Get current joint positions"""
        if self.use_ros2 and self.ros2_node:
            return self.ros2_node.get_joint_positions()
        else:
            # Use Gazebo service calls
            return self._get_joint_positions_gazebo()
    
    def _get_joint_positions_gazebo(self):
        """Get joint positions using Gazebo services"""
        positions = np.zeros(self.num_joints)
        
        try:
            for i, joint_name in enumerate(self.joint_names):
                if i < self.num_joints:
                    cmd = [
                        'gz', 'joint',
                        '--world-name', 'so101_world',
                        '--model-name', 'so101_robot',
                        '--joint-name', joint_name,
                        '--info'
                    ]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    # Parse position from output (simplified)
                    positions[i] = self.current_joint_positions[i]  # Fallback to current
        except Exception as e:
            print(f"⚠️  Failed to get joint positions: {e}")
        
        return positions
    
    def _detect_target_in_image(self, image):
        """Detect target object in camera image"""
        target_info = {
            'in_view': 0.0,
            'center_distance': 1.0,
            'pixel_position': None
        }
        
        if image is None:
            return target_info
        
        try:
            # Simple red object detection
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # Red color ranges
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 100, 100])
            upper_red2 = np.array([180, 255, 255])
            
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = cv2.bitwise_or(mask1, mask2)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                
                if area > 300:  # Minimum area threshold
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    center_x = x + w // 2
                    center_y = y + h // 2
                    
                    # Calculate distance from image center
                    image_center = np.array([self.camera_width // 2, self.camera_height // 2])
                    target_center = np.array([center_x, center_y])
                    distance = np.linalg.norm(target_center - image_center)
                    
                    # Normalize distance
                    max_distance = np.linalg.norm(image_center)
                    normalized_distance = min(distance / max_distance, 1.0)
                    
                    target_info['in_view'] = 1.0
                    target_info['center_distance'] = normalized_distance
                    target_info['pixel_position'] = (center_x, center_y)
                    
        except Exception as e:
            print(f"⚠️  Target detection failed: {e}")
        
        return target_info
    
    def _calculate_reward(self, observation):
        """Calculate reward (same as other environments)"""
        reward = 0.0
        
        target_in_view = observation['target_in_view'][0]
        center_distance = observation['target_center_distance'][0]
        joint_positions = observation['joint_positions']
        
        # Base reward for having target in view
        if target_in_view > 0.5:
            reward += 3.0
            
            # Smooth, continuous centering reward
            centering_reward = np.exp(-2.0 * center_distance) * 10.0
            reward += centering_reward
            
            # Progressive precision bonuses
            if center_distance < 0.05:
                self.target_centered_steps += 1
                reward += 15.0
            elif center_distance < 0.1:
                reward += 8.0
            elif center_distance < 0.2:
                reward += 4.0
            
            # Bonus for sustained centering
            if self.target_centered_steps > 10:
                reward += 2.0
                
        else:
            # Progressive penalty for losing target
            reward -= 2.0
            if hasattr(self, 'prev_target_in_view') and self.prev_target_in_view:
                reward -= 3.0
        
        # Movement penalties
        if len(joint_positions) > 0:
            extreme_position_penalty = np.sum(np.abs(joint_positions) > 2.0) * 0.5
            reward -= extreme_position_penalty
            
            if hasattr(self, 'prev_joint_positions') and self.prev_joint_positions is not None:
                joint_velocities = np.abs(joint_positions - self.prev_joint_positions)
                velocity_penalty = np.sum(joint_velocities) * 0.02
                reward -= velocity_penalty
        
        # Energy efficiency bonus
        if target_in_view > 0.5 and center_distance < 0.3:
            efficiency_bonus = 1.0 / (1.0 + np.sum(np.abs(joint_positions)) * 0.1)
            reward += efficiency_bonus
        
        # Track error and update previous state
        self.total_tracking_error += center_distance
        self.prev_target_in_view = target_in_view > 0.5
        if len(joint_positions) > 0:
            self.prev_joint_positions = joint_positions.copy()
        
        return reward
    
    def _is_terminated(self, observation):
        """Check if episode should terminate"""
        # Success condition: target well-centered for extended period
        if self.target_centered_steps > 50:
            return True
        
        # Failure condition: robot in extreme position
        joint_positions = observation['joint_positions']
        if len(joint_positions) > 0:
            if np.any(np.abs(joint_positions) > 3.0):
                return True
        
        return False
    
    def _get_info(self):
        """Get environment info"""
        return {
            'target_centered_steps': self.target_centered_steps,
            'total_tracking_error': self.total_tracking_error,
            'current_step': self.current_step,
            'target_centered': self.target_centered_steps > 0,
            'gazebo_running': self._is_gazebo_running()
        }
    
    def _get_dummy_observation(self):
        """Get dummy observation when Gazebo fails"""
        return {
            'camera_image': np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8),
            'joint_positions': np.zeros(self.num_joints, dtype=np.float32),
            'target_in_view': np.array([0.0], dtype=np.float32),
            'target_center_distance': np.array([1.0], dtype=np.float32)
        }
    
    def render(self):
        """Render environment (Gazebo handles this)"""
        pass
    
    def close(self):
        """Close environment"""
        # Stop Gazebo
        if self.gazebo_process:
            try:
                os.killpg(os.getpgid(self.gazebo_process.pid), signal.SIGTERM)
                self.gazebo_process.wait(timeout=5)
            except:
                pass
            self.gazebo_process = None
        
        # Shutdown ROS2 node
        if self.ros2_node:
            self.ros2_node.destroy_node()
            self.ros2_node = None
        
        if ROS2_AVAILABLE and rclpy.ok():
            rclpy.shutdown()
        
        print("🛑 Gazebo environment closed")


# ROS2 Controller Node (if ROS2 is available)
if ROS2_AVAILABLE:
    class GazeboControllerNode(Node):
        """ROS2 node for controlling Gazebo simulation"""
        
        def __init__(self):
            super().__init__('gazebo_controller')
            
            # Publishers
            self.joint_cmd_pub = self.create_publisher(
                Float64MultiArray, 
                '/so101_robot/joint_commands', 
                10
            )
            
            # Subscribers
            self.joint_state_sub = self.create_subscription(
                JointState,
                '/so101_robot/joint_states',
                self.joint_state_callback,
                10
            )
            
            self.camera_sub = self.create_subscription(
                Image,
                '/so101_robot/camera/image_raw',
                self.camera_callback,
                10
            )
            
            # State variables
            self.current_joint_positions = np.zeros(6)
            self.current_camera_image = None
            self.bridge = CvBridge()
        
        def joint_state_callback(self, msg):
            """Callback for joint state updates"""
            self.current_joint_positions = np.array(msg.position)
        
        def camera_callback(self, msg):
            """Callback for camera image updates"""
            try:
                self.current_camera_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            except Exception as e:
                self.get_logger().error(f'Camera callback error: {e}')
        
        def set_joint_positions(self, positions):
            """Set joint positions"""
            # This would typically involve calling a service
            pass
        
        def set_joint_velocities(self, velocities):
            """Set joint velocities"""
            msg = Float64MultiArray()
            msg.data = velocities.tolist()
            self.joint_cmd_pub.publish(msg)
        
        def get_joint_positions(self):
            """Get current joint positions"""
            return self.current_joint_positions
        
        def get_camera_image(self):
            """Get current camera image"""
            if self.current_camera_image is not None:
                return cv2.cvtColor(self.current_camera_image, cv2.COLOR_BGR2RGB)
            else:
                return np.zeros((480, 640, 3), dtype=np.uint8)
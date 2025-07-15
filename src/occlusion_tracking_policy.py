import numpy as np
import cv2
from typing import Dict, Tuple, Optional
import math

class OcclusionTrackingPolicy:
    """
    Enhanced visual servoing policy for SO-ARM101 object tracking with occlusion handling
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Control parameters - Increased for better responsiveness
        self.p_gain_fast = config.get('p_gain_fast', 2.0)  # For large errors
        self.p_gain_slow = config.get('p_gain_slow', 1.0)  # For small errors
        self.max_velocity = config.get('max_velocity', 2.0)
        
        # Visual servoing parameters
        self.error_threshold_fast = 0.3  # Switch to slow gain below this
        self.deadzone_radius = 0.08      # Stop moving if error is very small
        self.occlusion_threshold = 0.5   # Consider target occluded above this
        
        # Tracking state
        self.last_target_position = None
        self.lost_target_steps = 0
        self.occlusion_steps = 0
        self.search_direction = 1
        self.search_joint = 0
        
        # Occlusion handling state
        self.last_known_position = None
        self.occlusion_search_phase = 0
        
        # Performance tracking
        self.tracking_errors = []
        self.control_actions = []
        self.target_centered_steps = 0
        
        print(f"🎯 Occlusion tracking policy initialized:")
        print(f"   Fast gain: {self.p_gain_fast}, Slow gain: {self.p_gain_slow}")
        print(f"   Max velocity: {self.max_velocity}")
        print(f"   Occlusion threshold: {self.occlusion_threshold}")
        
    def predict(self, observation: Dict) -> np.ndarray:
        """Predict action based on observation with occlusion handling"""
        camera_image = observation['camera_image']
        joint_positions = observation['joint_positions']
        target_in_view = observation['target_in_view'][0]
        center_distance = observation['target_center_distance'][0]
        target_occluded = observation['target_occluded'][0]
        
        # Enhanced target detection
        target_pixel_pos = self._detect_target_position(camera_image)
        
        if target_pixel_pos is not None and target_in_view > 0.5:
            # Target is visible
            self.last_known_position = target_pixel_pos
            
            # Check occlusion level
            if target_occluded > self.occlusion_threshold:
                # Target is heavily occluded - use occlusion handling
                action = self._handle_occlusion(target_pixel_pos, camera_image.shape, 
                                              joint_positions, center_distance, target_occluded)
                self.occlusion_steps += 1
                print(f"🚧 Target occluded ({target_occluded:.2f}), searching for clear view")
            else:
                # Target is clearly visible
                if center_distance < self.deadzone_radius:
                    # Target is centered and clear - stop moving
                    action = np.zeros(len(joint_positions))
                    self.target_centered_steps += 1
                    print(f"🎯 Target centered and clear! Distance: {center_distance:.3f}")
                else:
                    # Target visible but not centered - use normal visual servoing
                    action = self._improved_visual_servoing(
                        target_pixel_pos, camera_image.shape, joint_positions, center_distance
                    )
                    self.target_centered_steps = 0
                
                self.occlusion_steps = 0
                
            self.lost_target_steps = 0
            self.last_target_position = target_pixel_pos
            
        else:
            # Target lost - intelligent search
            action = self._intelligent_search(joint_positions)
            self.lost_target_steps += 1
            self.target_centered_steps = 0
            self.occlusion_steps = 0
            
        # Smooth and clip action
        action = self._smooth_action(action)
        action = np.clip(action, -1.0, 1.0)
        
        # Track performance
        self.tracking_errors.append(center_distance)
        self.control_actions.append(action.copy())
        
        return action
        
    def _detect_target_position(self, image: np.ndarray) -> Optional[Tuple[int, int]]:
        """Enhanced target detection with better filtering"""
        if image is None or image.size == 0:
            return None
            
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # Enhanced red detection with multiple ranges
            ranges = [
                (np.array([0, 120, 120]), np.array([10, 255, 255])),
                (np.array([160, 120, 120]), np.array([180, 255, 255]))
            ]
            
            masks = []
            for lower, upper in ranges:
                mask = cv2.inRange(hsv, lower, upper)
                masks.append(mask)
            
            # Combine masks
            combined_mask = masks[0]
            for mask in masks[1:]:
                combined_mask = cv2.bitwise_or(combined_mask, mask)
            
            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Filter contours by area and circularity
                valid_contours = []
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 50:  # Lower threshold for partially occluded objects
                        # Check circularity (more lenient for occluded objects)
                        perimeter = cv2.arcLength(contour, True)
                        if perimeter > 0:
                            circularity = 4 * np.pi * area / (perimeter * perimeter)
                            if circularity > 0.2:  # More lenient for occlusion
                                valid_contours.append((contour, area))
                
                if valid_contours:
                    # Get largest valid contour
                    largest_contour = max(valid_contours, key=lambda x: x[1])[0]
                    
                    # Get center using moments
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        return (cx, cy)
                        
        except Exception as e:
            print(f"⚠️  Target detection error: {e}")
            
        return None
        
    def _handle_occlusion(self, target_pos: Tuple[int, int], image_shape: Tuple,
                         joint_positions: np.ndarray, center_distance: float, 
                         occlusion_level: float) -> np.ndarray:
        """Handle occluded target by searching for better viewpoint"""
        height, width = image_shape[:2]
        
        # Use reduced gains for occlusion handling
        occlusion_gain = self.p_gain_slow * 0.7
        
        # Determine search strategy based on occlusion level
        if occlusion_level > 0.8:
            # Heavily occluded - search for completely new viewpoint
            return self._occlusion_search(joint_positions, aggressive=True)
        elif occlusion_level > 0.5:
            # Partially occluded - try to move around the occlusion
            return self._circumvent_occlusion(target_pos, image_shape, joint_positions, center_distance)
        else:
            # Lightly occluded - use normal tracking with reduced gain
            return self._improved_visual_servoing(target_pos, image_shape, joint_positions, 
                                                center_distance, gain_modifier=0.5)
    
    def _circumvent_occlusion(self, target_pos: Tuple[int, int], image_shape: Tuple,
                             joint_positions: np.ndarray, center_distance: float) -> np.ndarray:
        """Try to move around occlusion while keeping target in view"""
        height, width = image_shape[:2]
        center_x, center_y = width // 2, height // 2
        
        # Calculate direction to move camera to circumvent occlusion
        error_x = target_pos[0] - center_x
        error_y = target_pos[1] - center_y
        
        # Determine which direction to move to avoid occlusion
        # Simple heuristic: move perpendicular to target direction
        if abs(error_x) > abs(error_y):
            # Move vertically to avoid horizontal occlusion
            circumvent_y = 0.3 * np.sign(error_y) if error_y != 0 else 0.3
            circumvent_x = 0.1 * np.sign(error_x)
        else:
            # Move horizontally to avoid vertical occlusion
            circumvent_x = 0.3 * np.sign(error_x) if error_x != 0 else 0.3
            circumvent_y = 0.1 * np.sign(error_y)
        
        # Create action to circumvent occlusion
        action = np.zeros(len(joint_positions))
        
        if len(action) >= 6:
            # Move to circumvent occlusion
            action[0] = circumvent_x * 1.0  # Base rotation
            action[1] = circumvent_y * 0.8  # Shoulder
            action[2] = circumvent_y * 0.5  # Elbow
            action[3] = circumvent_y * 0.3  # Wrist pitch
            action[4] = circumvent_x * 0.4  # Wrist roll
            action[5] = 0.0  # Camera joint disabled
            
        return action * self.max_velocity
    
    def _occlusion_search(self, joint_positions: np.ndarray, aggressive: bool = False) -> np.ndarray:
        """Search for unoccluded view of target"""
        action = np.zeros(len(joint_positions))
        
        # Cycle through different search phases
        search_phase = self.occlusion_steps // 15  # Change phase every 1.5 seconds
        
        search_intensity = 1.0 if aggressive else 0.6
        
        if search_phase % 4 == 0:
            # Phase 1: Horizontal search (base rotation)
            action[0] = search_intensity * self.search_direction
            
        elif search_phase % 4 == 1:
            # Phase 2: Vertical search (shoulder movement)
            action[1] = search_intensity * 0.7 * self.search_direction
            
        elif search_phase % 4 == 2:
            # Phase 3: Combined movement
            action[0] = search_intensity * 0.5 * self.search_direction
            action[1] = search_intensity * 0.5 * (-self.search_direction)
            
        else:
            # Phase 4: Elbow adjustment
            action[2] = search_intensity * 0.6 * self.search_direction
            
        # Reverse direction periodically
        if self.occlusion_steps % 60 == 0:  # Every 6 seconds
            self.search_direction *= -1
            
        return action * self.max_velocity
        
    def _improved_visual_servoing(self, target_pos: Tuple[int, int], 
                                image_shape: Tuple, joint_positions: np.ndarray,
                                center_distance: float, gain_modifier: float = 1.0) -> np.ndarray:
        """Improved visual servoing with adaptive gains"""
        height, width = image_shape[:2]
        center_x, center_y = width // 2, height // 2
        
        # Calculate pixel errors
        error_x = target_pos[0] - center_x
        error_y = target_pos[1] - center_y
        
        # Normalize errors to [-1, 1]
        norm_error_x = error_x / (width / 2)
        norm_error_y = error_y / (height / 2)
        
        # Apply deadzone
        if abs(norm_error_x) < self.deadzone_radius:
            norm_error_x = 0
        if abs(norm_error_y) < self.deadzone_radius:
            norm_error_y = 0
            
        # Adaptive gain based on error magnitude
        if center_distance > self.error_threshold_fast:
            gain = self.p_gain_fast
        else:
            gain = self.p_gain_slow
            
        # Apply gain modifier for occlusion handling
        gain *= gain_modifier
            
        # Initialize action
        action = np.zeros(len(joint_positions))
        
        # SO-ARM101 specific joint mapping - Increased scaling for better responsiveness
        if len(action) >= 6:
            # Joint 0: Base rotation (yaw) - for horizontal error
            action[0] = -gain * norm_error_x * 1.5
            
            # Joint 1: Shoulder (pitch) - for vertical error  
            action[1] = gain * norm_error_y * 1.2
            
            # Joint 2: Elbow - assist with vertical movement
            action[2] = gain * norm_error_y * 0.8
            
            # Joint 3: Wrist pitch - fine vertical adjustment
            action[3] = gain * norm_error_y * 0.6
            
            # Joint 4: Wrist roll - fine horizontal adjustment
            action[4] = -gain * norm_error_x * 0.5
            
            # Joint 5: Camera joint - disabled (no movement)
            action[5] = 0.0
            
        # Scale by max velocity
        action = action * self.max_velocity
        
        return action
        
    def _intelligent_search(self, joint_positions: np.ndarray) -> np.ndarray:
        """Intelligent search pattern when target is lost"""
        action = np.zeros(len(joint_positions))
        
        # Multi-phase search strategy - Increased for faster search
        search_phase = self.lost_target_steps // 20  # Change phase every 2 seconds
        
        if search_phase == 0:
            # Phase 1: Rotate base faster
            action[0] = 0.8 * self.search_direction
            
        elif search_phase == 1:
            # Phase 2: Move shoulder up/down faster
            action[1] = 0.6 * self.search_direction
            
        elif search_phase == 2:
            # Phase 3: Combined base + shoulder movement
            action[0] = 0.6 * self.search_direction
            action[1] = 0.4 * (-self.search_direction)
            
        else:
            # Phase 4: Return to center and reverse direction
            # Move joints toward neutral positions
            neutral_positions = [0, -0.3, 0.5, 0, 0.2, 0]
            for i in range(min(len(action), len(neutral_positions))):
                if i < len(joint_positions):
                    error = neutral_positions[i] - joint_positions[i]
                    action[i] = 0.6 * np.sign(error) * min(abs(error), 1.0)
            
            # Reset search after returning to center
            if self.lost_target_steps > 150:  # 15 seconds
                self.lost_target_steps = 0
                self.search_direction *= -1  # Reverse search direction
                
        return action
        
    def _smooth_action(self, action: np.ndarray) -> np.ndarray:
        """Apply smoothing to reduce jittery movements"""
        # Simple exponential smoothing
        if hasattr(self, '_last_action'):
            alpha = 0.7  # Smoothing factor
            action = alpha * action + (1 - alpha) * self._last_action
        
        self._last_action = action.copy()
        return action
        
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics"""
        if not self.tracking_errors:
            return {'no_data': True}
            
        return {
            'average_tracking_error': np.mean(self.tracking_errors),
            'min_tracking_error': np.min(self.tracking_errors),
            'std_tracking_error': np.std(self.tracking_errors),
            'tracking_steps': len(self.tracking_errors),
            'lost_target_episodes': self.lost_target_steps,
            'occlusion_episodes': self.occlusion_steps,
            'recent_performance': np.mean(self.tracking_errors[-50:]) if len(self.tracking_errors) >= 50 else np.mean(self.tracking_errors)
        }
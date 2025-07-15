import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class DetectedObject:
    """Detected object information"""
    id: int
    class_name: str
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float
    center: Tuple[float, float]
    position_3d: Optional[np.ndarray] = None

class SimpleObjectDetector:
    """
    Simple object detector for active vision
    Uses color-based detection for rapid prototyping
    """
    
    def __init__(self):
        # Define enhanced color ranges for different objects (HSV)
        self.color_ranges = {
            'red_object': [
                {'lower': np.array([0, 100, 100]), 'upper': np.array([10, 255, 255])},
                {'lower': np.array([160, 100, 100]), 'upper': np.array([180, 255, 255])}
            ],
            'green_object': [
                {'lower': np.array([40, 100, 100]), 'upper': np.array([80, 255, 255])}
            ],
            'blue_object': [
                {'lower': np.array([100, 100, 100]), 'upper': np.array([130, 255, 255])}
            ],
            'yellow_object': [
                {'lower': np.array([20, 100, 100]), 'upper': np.array([30, 255, 255])}
            ],
            'purple_object': [
                {'lower': np.array([130, 100, 100]), 'upper': np.array([160, 255, 255])}
            ]
        }
        
        self.min_contour_area = 300  # Reduced for better detection
        self.max_contour_area = 50000  # Prevent detecting entire background
        self.detected_objects = []
        
        # YOLO integration preparation
        self.use_yolo = False
        self.yolo_model = None
        
    def detect_objects(self, image: np.ndarray) -> List[DetectedObject]:
        """Detect objects in image using enhanced color-based detection"""
        if image is None:
            return []
            
        # Use YOLO if available, otherwise fall back to color detection
        if self.use_yolo and self.yolo_model is not None:
            return self._detect_with_yolo(image)
        else:
            return self._detect_with_color(image)
    
    def _detect_with_color(self, image: np.ndarray) -> List[DetectedObject]:
        """Color-based detection method"""
            
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Apply Gaussian blur to reduce noise
        hsv_blurred = cv2.GaussianBlur(hsv, (5, 5), 0)
        
        detected_objects = []
        
        for obj_id, (class_name, color_ranges) in enumerate(self.color_ranges.items()):
            # Create combined mask for all color ranges of this object
            combined_mask = None
            
            for color_range in color_ranges:
                mask = cv2.inRange(hsv_blurred, color_range['lower'], color_range['upper'])
                if combined_mask is None:
                    combined_mask = mask
                else:
                    combined_mask = cv2.bitwise_or(combined_mask, mask)
            
            if combined_mask is None:
                continue
                
            # Apply morphological operations to clean up the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if self.min_contour_area < area < self.max_contour_area:
                    # Get bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Filter by aspect ratio (avoid very elongated shapes)
                    aspect_ratio = w / h if h > 0 else 0
                    if 0.2 < aspect_ratio < 5.0:  # Reasonable aspect ratio
                        # Calculate center
                        center_x = x + w // 2
                        center_y = y + h // 2
                        
                        # Enhanced confidence calculation
                        confidence = self._calculate_confidence(contour, area)
                        
                        # Create detected object
                        detected_obj = DetectedObject(
                            id=obj_id,
                            class_name=class_name,
                            bbox=(x, y, w, h),
                            confidence=confidence,
                            center=(center_x, center_y)
                        )
                        
                        detected_objects.append(detected_obj)
        
        # Sort by confidence (highest first)
        detected_objects.sort(key=lambda obj: obj.confidence, reverse=True)
        
        self.detected_objects = detected_objects
        return detected_objects
    
    def _calculate_confidence(self, contour: np.ndarray, area: float) -> float:
        """Calculate confidence based on contour properties"""
        # Base confidence from area
        area_confidence = min(area / 5000.0, 1.0)
        
        # Compactness measure (circle-like shapes get higher confidence)
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            compactness = 4 * np.pi * area / (perimeter * perimeter)
            compactness_confidence = min(compactness * 2, 1.0)
        else:
            compactness_confidence = 0.0
        
        # Combine confidences
        final_confidence = 0.7 * area_confidence + 0.3 * compactness_confidence
        return min(final_confidence, 1.0)
    
    def _detect_with_yolo(self, image: np.ndarray) -> List[DetectedObject]:
        """YOLO-based detection method (placeholder for future implementation)"""
        try:
            # TODO: Implement YOLO detection
            # results = self.yolo_model(image)
            # return self._convert_yolo_results(results)
            print("⚠️  YOLO detection not yet implemented, falling back to color detection")
            return self._detect_with_color(image)
        except Exception as e:
            print(f"❌ YOLO detection failed: {e}, falling back to color detection")
            return self._detect_with_color(image)
    
    def enable_yolo(self, model_path: str = "yolov8n.pt"):
        """Enable YOLO detection (requires ultralytics package)"""
        try:
            # TODO: Implement YOLO model loading
            # from ultralytics import YOLO
            # self.yolo_model = YOLO(model_path)
            # self.use_yolo = True
            print("⚠️  YOLO integration not yet implemented")
            return False
        except ImportError:
            print("❌ ultralytics package not found. Install with: pip install ultralytics")
            return False
        except Exception as e:
            print(f"❌ Failed to load YOLO model: {e}")
            return False
    
    def draw_detections(self, image: np.ndarray, detections: List[DetectedObject]) -> np.ndarray:
        """Draw detection results on image"""
        result_img = image.copy()
        
        for detection in detections:
            x, y, w, h = detection.bbox
            
            # Draw bounding box
            cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw center point
            cv2.circle(result_img, (int(detection.center[0]), int(detection.center[1])), 5, (0, 0, 255), -1)
            
            # Draw label
            label = f"{detection.class_name}: {detection.confidence:.2f}"
            cv2.putText(result_img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return result_img
    
    def get_target_position(self, detection: DetectedObject, camera_id: int) -> np.ndarray:
        """Estimate 3D position of detected object"""
        # Simple depth estimation based on object size
        # In real implementation, this would use stereo vision or depth camera
        
        _, _, w, h = detection.bbox
        object_size = max(w, h)
        
        # Estimate depth based on object size (very rough approximation)
        estimated_depth = 1000.0 / object_size if object_size > 0 else 0.5
        
        # Convert pixel coordinates to 3D (simplified)
        # This assumes known camera calibration parameters
        center_x, center_y = detection.center
        
        # Simple projection (would need proper camera matrix in real setup)
        world_x = (center_x - 320) * estimated_depth / 320
        world_y = (center_y - 240) * estimated_depth / 240
        world_z = estimated_depth
        
        return np.array([world_x, world_y, world_z])
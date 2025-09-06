"""
Vehicle Detection Module

This module provides vehicle detection capabilities using YOLO models.
It's designed specifically for aerial drone footage to detect cars, motorcycles, 
buses, and trucks.

Author: Adriel Clinton
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict
from ultralytics import YOLO
import torch
from loguru import logger


class Detection:
    """Represents a single vehicle detection."""
    
    def __init__(self, bbox: Tuple[int, int, int, int], confidence: float, 
                 class_id: int, class_name: str):
        """
        Initialize a detection.
        
        Args:
            bbox: Bounding box coordinates (x1, y1, x2, y2)
            confidence: Detection confidence score
            class_id: Class ID from model
            class_name: Human-readable class name
        """
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name
        self.center = self._calculate_center()
        self.area = self._calculate_area()
    
    def _calculate_center(self) -> Tuple[int, int]:
        """Calculate the center point of the bounding box."""
        x1, y1, x2, y2 = self.bbox
        return (int((x1 + x2) / 2), int((y1 + y2) / 2))
    
    def _calculate_area(self) -> int:
        """Calculate the area of the bounding box."""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)


class VehicleDetector:
    """
    Vehicle detector using YOLO for aerial drone footage.
    
    This class handles loading YOLO models and performing vehicle detection
    on individual frames from drone videos.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the vehicle detector.
        
        Args:
            config: Configuration dictionary containing model settings
        """
        self.config = config
        self.model_config = config.get('model', {})
        self.detection_config = config.get('detection', {})
        
        # Model parameters
        self.confidence_threshold = self.model_config.get('confidence_threshold', 0.5)
        self.iou_threshold = self.model_config.get('iou_threshold', 0.45)
        self.device = self._get_device()
        
        # Target vehicle classes (COCO dataset)
        self.target_classes = self.detection_config.get('target_classes', [2, 3, 5, 7])
        self.class_names = self.detection_config.get('class_names', {
            2: "car", 3: "motorcycle", 5: "bus", 7: "truck"
        })
        
        # Load model
        self.model = self._load_model()
        
        logger.info(f"VehicleDetector initialized with device: {self.device}")
        logger.info(f"Target classes: {self.target_classes}")
    
    def _get_device(self) -> str:
        """Determine the best available device for inference."""
        device_config = self.model_config.get('device', 'auto')
        
        if device_config == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'
            else:
                return 'cpu'
        else:
            return device_config
    
    def _load_model(self) -> YOLO:
        """Load the YOLO model."""
        weights_path = self.model_config.get('weights_path', 'yolov8n.pt')
        
        try:
            model = YOLO(weights_path)
            model.to(self.device)
            logger.info(f"Model loaded successfully: {weights_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model {weights_path}: {str(e)}")
            # Fallback to default model
            logger.info("Falling back to default YOLOv8n model")
            model = YOLO('yolov8n.pt')
            model.to(self.device)
            return model
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect vehicles in a single frame.
        
        Args:
            frame: Input image frame as numpy array
            
        Returns:
            List of Detection objects
        """
        try:
            # Run inference
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                classes=self.target_classes,
                device=self.device,
                verbose=False
            )
            
            detections = []
            
            # Process results
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.cpu().numpy()
                    
                    for box in boxes:
                        # Extract box information
                        x1, y1, x2, y2 = box.xyxy[0].astype(int)
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        
                        # Get class name
                        class_name = self.class_names.get(class_id, f"class_{class_id}")
                        
                        # Create detection object
                        detection = Detection(
                            bbox=(x1, y1, x2, y2),
                            confidence=confidence,
                            class_id=class_id,
                            class_name=class_name
                        )
                        
                        detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"Error during detection: {str(e)}")
            return []
    
    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """
        Detect vehicles in multiple frames.
        
        Args:
            frames: List of input image frames
            
        Returns:
            List of detection lists for each frame
        """
        all_detections = []
        
        for frame in frames:
            detections = self.detect(frame)
            all_detections.append(detections)
        
        return all_detections
    
    def visualize_detections(self, frame: np.ndarray, 
                           detections: List[Detection]) -> np.ndarray:
        """
        Draw detection bounding boxes on frame.
        
        Args:
            frame: Input frame
            detections: List of detections to visualize
            
        Returns:
            Frame with drawn detections
        """
        vis_frame = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            
            # Get color for class
            color = self._get_class_color(detection.class_id)
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{detection.class_name}: {detection.confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            # Label background
            cv2.rectangle(
                vis_frame,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1
            )
            
            # Label text
            cv2.putText(
                vis_frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        return vis_frame
    
    def _get_class_color(self, class_id: int) -> Tuple[int, int, int]:
        """Get color for a specific class."""
        colors = self.config.get('visualization', {}).get('colors', {})
        class_name = self.class_names.get(class_id, 'unknown')
        
        # Default colors (BGR format)
        default_colors = {
            'car': (0, 255, 0),      # Green
            'motorcycle': (255, 0, 0),  # Blue
            'bus': (0, 0, 255),      # Red
            'truck': (255, 255, 0),  # Cyan
            'unknown': (128, 128, 128)  # Gray
        }
        
        return colors.get(class_name, default_colors.get(class_name, (128, 128, 128)))
    
    def get_detection_stats(self, detections: List[Detection]) -> Dict:
        """
        Get statistics about detections.
        
        Args:
            detections: List of detections
            
        Returns:
            Dictionary with detection statistics
        """
        if not detections:
            return {
                'total_count': 0,
                'class_counts': {},
                'avg_confidence': 0.0,
                'bbox_areas': []
            }
        
        # Count by class
        class_counts = {}
        confidences = []
        bbox_areas = []
        
        for detection in detections:
            class_name = detection.class_name
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            confidences.append(detection.confidence)
            bbox_areas.append(detection.area)
        
        return {
            'total_count': len(detections),
            'class_counts': class_counts,
            'avg_confidence': np.mean(confidences),
            'bbox_areas': bbox_areas
        }

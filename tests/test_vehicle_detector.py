"""
Test Vehicle Detector

Unit tests for the vehicle detection module.

Author: Adriel Clinton
"""

import unittest
import numpy as np
import cv2
import tempfile
import os
from unittest.mock import Mock, patch

import sys
sys.path.append('../src')

from src.detection.vehicle_detector import VehicleDetector, Detection
from src.utils.config_manager import ConfigManager


class TestVehicleDetector(unittest.TestCase):
    """Test cases for VehicleDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'model': {
                'type': 'yolov8',
                'weights_path': 'yolov8n.pt',
                'confidence_threshold': 0.5,
                'iou_threshold': 0.45,
                'device': 'cpu'
            },
            'detection': {
                'target_classes': [2, 3, 5, 7],
                'class_names': {
                    2: "car",
                    3: "motorcycle",
                    5: "bus",
                    7: "truck"
                }
            },
            'visualization': {
                'colors': {
                    'car': [0, 255, 0],
                    'motorcycle': [255, 0, 0],
                    'bus': [0, 0, 255],
                    'truck': [255, 255, 0]
                }
            }
        }
        
        # Create test image
        self.test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    def test_detection_init(self):
        """Test Detection class initialization."""
        detection = Detection(
            bbox=(10, 20, 100, 150),
            confidence=0.85,
            class_id=2,
            class_name="car"
        )
        
        self.assertEqual(detection.bbox, (10, 20, 100, 150))
        self.assertEqual(detection.confidence, 0.85)
        self.assertEqual(detection.class_id, 2)
        self.assertEqual(detection.class_name, "car")
        self.assertEqual(detection.center, (55, 85))
        self.assertEqual(detection.area, 90 * 130)
    
    @patch('src.detection.vehicle_detector.YOLO')
    def test_detector_init(self, mock_yolo):
        """Test VehicleDetector initialization."""
        mock_model = Mock()
        mock_yolo.return_value = mock_model
        
        detector = VehicleDetector(self.config)
        
        self.assertEqual(detector.confidence_threshold, 0.5)
        self.assertEqual(detector.iou_threshold, 0.45)
        self.assertEqual(detector.target_classes, [2, 3, 5, 7])
        self.assertEqual(detector.device, 'cpu')
        mock_yolo.assert_called_once()
    
    @patch('src.detection.vehicle_detector.YOLO')
    def test_detect_empty_results(self, mock_yolo):
        """Test detection with no results."""
        mock_model = Mock()
        mock_result = Mock()
        mock_result.boxes = None
        mock_model.return_value = [mock_result]
        mock_yolo.return_value = mock_model
        
        detector = VehicleDetector(self.config)
        detections = detector.detect(self.test_image)
        
        self.assertEqual(len(detections), 0)
    
    @patch('src.detection.vehicle_detector.YOLO')
    def test_detect_with_results(self, mock_yolo):
        """Test detection with mock results."""
        mock_model = Mock()
        mock_result = Mock()
        
        # Mock boxes
        mock_box = Mock()
        mock_box.xyxy = np.array([[10, 20, 100, 150]])
        mock_box.conf = np.array([0.85])
        mock_box.cls = np.array([2])
        
        mock_boxes = Mock()
        mock_boxes.cpu.return_value.numpy.return_value = mock_box
        mock_result.boxes = mock_boxes
        mock_result.boxes.cpu.return_value = mock_boxes
        
        mock_model.return_value = [mock_result]
        mock_yolo.return_value = mock_model
        
        detector = VehicleDetector(self.config)
        
        # Mock the actual detection results
        with patch.object(detector.model, '__call__') as mock_call:
            mock_call.return_value = [mock_result]
            detections = detector.detect(self.test_image)
        
        self.assertIsInstance(detections, list)
    
    def test_get_class_color(self):
        """Test color retrieval for classes."""
        with patch('src.detection.vehicle_detector.YOLO'):
            detector = VehicleDetector(self.config)
            
            car_color = detector._get_class_color(2)
            self.assertEqual(car_color, (0, 255, 0))
            
            unknown_color = detector._get_class_color(999)
            self.assertEqual(unknown_color, (128, 128, 128))
    
    def test_detection_stats(self):
        """Test detection statistics calculation."""
        detections = [
            Detection((10, 20, 100, 150), 0.85, 2, "car"),
            Detection((200, 50, 300, 200), 0.75, 2, "car"),
            Detection((400, 100, 500, 250), 0.65, 3, "motorcycle")
        ]
        
        with patch('src.detection.vehicle_detector.YOLO'):
            detector = VehicleDetector(self.config)
            stats = detector.get_detection_stats(detections)
        
        self.assertEqual(stats['total_count'], 3)
        self.assertEqual(stats['class_counts']['car'], 2)
        self.assertEqual(stats['class_counts']['motorcycle'], 1)
        self.assertAlmostEqual(stats['avg_confidence'], 0.75, places=2)
    
    def test_empty_detection_stats(self):
        """Test statistics with empty detection list."""
        with patch('src.detection.vehicle_detector.YOLO'):
            detector = VehicleDetector(self.config)
            stats = detector.get_detection_stats([])
        
        self.assertEqual(stats['total_count'], 0)
        self.assertEqual(stats['class_counts'], {})
        self.assertEqual(stats['avg_confidence'], 0.0)


if __name__ == '__main__':
    unittest.main()

"""
Configuration Manager

Handles loading and managing configuration settings for the drone CV system.

Author: Adriel Clinton
"""

import yaml
import os
from typing import Dict, Any
from loguru import logger


class ConfigManager:
    """Manages configuration settings for the application."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not os.path.exists(self.config_path):
            logger.warning(f"Config file not found: {self.config_path}")
            return self._get_default_config()
        
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                logger.info(f"Configuration loaded from {self.config_path}")
                return config
        except Exception as e:
            logger.error(f"Failed to load config: {str(e)}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'model': {
                'type': 'yolov8',
                'weights_path': 'yolov8n.pt',
                'confidence_threshold': 0.5,
                'iou_threshold': 0.45,
                'device': 'auto'
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
            'tracking': {
                'tracker_type': 'bytetrack',
                'max_disappeared': 30,
                'max_distance': 50,
                'match_threshold': 0.8
            },
            'video': {
                'input_format': ['.mp4', '.avi', '.mov', '.mkv'],
                'output_format': '.mp4',
                'resize_factor': 1.0,
                'skip_frames': 0
            },
            'visualization': {
                'show_detections': True,
                'show_tracks': True,
                'show_count': True,
                'colors': {
                    'car': [0, 255, 0],
                    'motorcycle': [255, 0, 0],
                    'bus': [0, 0, 255],
                    'truck': [255, 255, 0]
                }
            },
            'output': {
                'save_video': True,
                'save_frames': False,
                'save_statistics': True,
                'video_output_dir': 'results/videos',
                'stats_output_dir': 'results/statistics'
            },
            'logging': {
                'level': 'INFO',
                'log_file': 'logs/drone_cv.log',
                'console_output': True
            }
        }
    
    def get_config(self) -> Dict[str, Any]:
        """Get the full configuration dictionary."""
        return self.config
    
    def get_section(self, section_name: str) -> Dict[str, Any]:
        """Get a specific configuration section."""
        return self.config.get(section_name, {})
    
    def get_value(self, key_path: str, default=None):
        """
        Get a specific configuration value using dot notation.
        
        Args:
            key_path: Path to the value (e.g., 'model.confidence_threshold')
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        current = self.config
        
        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default
    
    def save_config(self, output_path: str = None):
        """
        Save current configuration to file.
        
        Args:
            output_path: Path to save config (default: original path)
        """
        if output_path is None:
            output_path = self.config_path
        
        try:
            with open(output_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
                logger.info(f"Configuration saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save config: {str(e)}")
    
    def update_config(self, updates: Dict[str, Any]):
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of updates to apply
        """
        def update_nested_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = update_nested_dict(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        self.config = update_nested_dict(self.config, updates)
        logger.info("Configuration updated")
    
    def validate_config(self) -> bool:
        """
        Validate configuration for required fields.
        
        Returns:
            True if configuration is valid
        """
        required_sections = ['model', 'detection', 'tracking', 'video']
        
        for section in required_sections:
            if section not in self.config:
                logger.error(f"Missing required config section: {section}")
                return False
        
        # Validate model section
        model_config = self.config.get('model', {})
        if 'confidence_threshold' not in model_config:
            logger.error("Missing model.confidence_threshold")
            return False
        
        # Validate detection section
        detection_config = self.config.get('detection', {})
        if 'target_classes' not in detection_config:
            logger.error("Missing detection.target_classes")
            return False
        
        logger.info("Configuration validation passed")
        return True

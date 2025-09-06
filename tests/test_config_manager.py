"""
Test Configuration Manager

Unit tests for the configuration management module.

Author: Adriel Clinton
"""

import unittest
import tempfile
import os
import yaml
from unittest.mock import patch

import sys
sys.path.append('../src')

from src.utils.config_manager import ConfigManager


class TestConfigManager(unittest.TestCase):
    """Test cases for ConfigManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_config = {
            'model': {
                'type': 'yolov8',
                'confidence_threshold': 0.5
            },
            'detection': {
                'target_classes': [2, 3, 5, 7]
            }
        }
    
    def test_config_manager_with_valid_file(self):
        """Test ConfigManager with valid config file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(self.test_config, f)
            config_path = f.name
        
        try:
            manager = ConfigManager(config_path)
            config = manager.get_config()
            
            self.assertEqual(config['model']['type'], 'yolov8')
            self.assertEqual(config['model']['confidence_threshold'], 0.5)
            self.assertEqual(config['detection']['target_classes'], [2, 3, 5, 7])
        finally:
            os.unlink(config_path)
    
    def test_config_manager_with_missing_file(self):
        """Test ConfigManager with missing config file."""
        manager = ConfigManager('nonexistent.yaml')
        config = manager.get_config()
        
        # Should return default config
        self.assertIn('model', config)
        self.assertIn('detection', config)
        self.assertEqual(config['model']['type'], 'yolov8')
    
    def test_get_section(self):
        """Test getting specific config section."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(self.test_config, f)
            config_path = f.name
        
        try:
            manager = ConfigManager(config_path)
            model_config = manager.get_section('model')
            
            self.assertEqual(model_config['type'], 'yolov8')
            self.assertEqual(model_config['confidence_threshold'], 0.5)
        finally:
            os.unlink(config_path)
    
    def test_get_value_with_dot_notation(self):
        """Test getting value with dot notation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(self.test_config, f)
            config_path = f.name
        
        try:
            manager = ConfigManager(config_path)
            
            model_type = manager.get_value('model.type')
            self.assertEqual(model_type, 'yolov8')
            
            threshold = manager.get_value('model.confidence_threshold')
            self.assertEqual(threshold, 0.5)
            
            # Test default value
            missing_value = manager.get_value('missing.key', 'default')
            self.assertEqual(missing_value, 'default')
        finally:
            os.unlink(config_path)
    
    def test_update_config(self):
        """Test updating configuration."""
        manager = ConfigManager('nonexistent.yaml')  # Uses default config
        
        updates = {
            'model': {
                'confidence_threshold': 0.7
            },
            'new_section': {
                'new_key': 'new_value'
            }
        }
        
        manager.update_config(updates)
        config = manager.get_config()
        
        self.assertEqual(config['model']['confidence_threshold'], 0.7)
        self.assertEqual(config['new_section']['new_key'], 'new_value')
        # Should preserve existing values
        self.assertEqual(config['model']['type'], 'yolov8')
    
    def test_save_config(self):
        """Test saving configuration to file."""
        with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
            output_path = f.name
        
        try:
            manager = ConfigManager('nonexistent.yaml')  # Uses default config
            manager.save_config(output_path)
            
            # Verify file was created and contains valid YAML
            self.assertTrue(os.path.exists(output_path))
            
            with open(output_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
            
            self.assertIn('model', loaded_config)
            self.assertIn('detection', loaded_config)
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
    
    def test_validate_config_valid(self):
        """Test validation with valid config."""
        manager = ConfigManager('nonexistent.yaml')  # Uses default config
        is_valid = manager.validate_config()
        self.assertTrue(is_valid)
    
    def test_validate_config_invalid(self):
        """Test validation with invalid config."""
        invalid_config = {
            'model': {},  # Missing required fields
            'detection': {}  # Missing required fields
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(invalid_config, f)
            config_path = f.name
        
        try:
            manager = ConfigManager(config_path)
            is_valid = manager.validate_config()
            self.assertFalse(is_valid)
        finally:
            os.unlink(config_path)


if __name__ == '__main__':
    unittest.main()

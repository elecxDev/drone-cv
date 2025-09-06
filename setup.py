#!/usr/bin/env python3
"""
Setup Script for Drone Vehicle Detection System

This script helps set up the project environment and download required models.

Author: Adriel Clinton
"""

import os
import sys
import subprocess
import requests
from pathlib import Path
import argparse
from urllib.parse import urlparse


def print_status(message, status="INFO"):
    """Print status message with formatting."""
    colors = {
        "INFO": "\033[94m",
        "SUCCESS": "\033[92m", 
        "WARNING": "\033[93m",
        "ERROR": "\033[91m",
        "ENDC": "\033[0m"
    }
    
    color = colors.get(status, colors["INFO"])
    print(f"{color}[{status}]{colors['ENDC']} {message}")


def check_python_version():
    """Check if Python version is compatible."""
    print_status("Checking Python version...")
    
    if sys.version_info < (3, 8):
        print_status("Python 3.8 or higher is required", "ERROR")
        return False
    
    print_status(f"Python {sys.version.split()[0]} - OK", "SUCCESS")
    return True


def install_requirements():
    """Install Python requirements."""
    print_status("Installing Python requirements...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print_status("Requirements installed successfully", "SUCCESS")
        return True
    except subprocess.CalledProcessError:
        print_status("Failed to install requirements", "ERROR")
        return False


def download_file(url, destination):
    """Download file from URL."""
    print_status(f"Downloading {os.path.basename(destination)}...")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rProgress: {percent:.1f}%", end="", flush=True)
        
        print()  # New line after progress
        print_status(f"Downloaded {os.path.basename(destination)}", "SUCCESS")
        return True
        
    except Exception as e:
        print_status(f"Failed to download {os.path.basename(destination)}: {str(e)}", "ERROR")
        return False


def download_yolo_models():
    """Download YOLO model files."""
    print_status("Downloading YOLO models...")
    
    models_dir = Path("models/yolo")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # YOLO model URLs (these are example URLs - update with actual URLs)
    model_urls = {
        "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v8.0.0/yolov8n.pt",
        "yolov8s.pt": "https://github.com/ultralytics/assets/releases/download/v8.0.0/yolov8s.pt",
    }
    
    success = True
    for filename, url in model_urls.items():
        destination = models_dir / filename
        
        if destination.exists():
            print_status(f"{filename} already exists - skipping", "WARNING")
            continue
        
        if not download_file(url, destination):
            success = False
    
    return success


def create_directories():
    """Create necessary directories."""
    print_status("Creating project directories...")
    
    directories = [
        "data/raw",
        "data/processed", 
        "data/sample",
        "models/yolo",
        "models/custom",
        "results/videos",
        "results/statistics",
        "results/reports",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print_status("Directories created", "SUCCESS")


def setup_git_hooks():
    """Set up Git hooks (optional)."""
    if not Path(".git").exists():
        print_status("Not a Git repository - skipping hooks", "WARNING")
        return
    
    print_status("Setting up Git hooks...")
    
    # Pre-commit hook to run tests
    pre_commit_hook = """#!/bin/sh
# Run tests before commit
python -m pytest tests/ --quiet
if [ $? -ne 0 ]; then
    echo "Tests failed. Commit aborted."
    exit 1
fi
"""
    
    hooks_dir = Path(".git/hooks")
    hook_file = hooks_dir / "pre-commit"
    
    try:
        with open(hook_file, 'w') as f:
            f.write(pre_commit_hook)
        
        # Make executable on Unix systems
        if os.name != 'nt':
            os.chmod(hook_file, 0o755)
        
        print_status("Git hooks set up", "SUCCESS")
    except Exception as e:
        print_status(f"Failed to set up Git hooks: {str(e)}", "WARNING")


def test_installation():
    """Test the installation."""
    print_status("Testing installation...")
    
    try:
        # Test imports
        sys.path.append('src')
        from src.detection.vehicle_detector import VehicleDetector
        from src.utils.config_manager import ConfigManager
        print_status("Module imports - OK", "SUCCESS")
        
        # Test configuration
        config_manager = ConfigManager("config.yaml")
        config = config_manager.get_config()
        print_status("Configuration loading - OK", "SUCCESS")
        
        # Test model initialization (if models are available)
        yolo_model_path = Path("models/yolo/yolov8n.pt")
        if yolo_model_path.exists():
            try:
                detector = VehicleDetector(config)
                print_status("Model initialization - OK", "SUCCESS")
            except Exception as e:
                print_status(f"Model initialization failed: {str(e)}", "WARNING")
        else:
            print_status("YOLO models not found - detector test skipped", "WARNING")
        
        return True
        
    except ImportError as e:
        print_status(f"Import test failed: {str(e)}", "ERROR")
        return False
    except Exception as e:
        print_status(f"Installation test failed: {str(e)}", "ERROR")
        return False


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup Drone Vehicle Detection System")
    parser.add_argument("--skip-models", action="store_true", 
                       help="Skip downloading YOLO models")
    parser.add_argument("--skip-requirements", action="store_true",
                       help="Skip installing requirements")
    parser.add_argument("--test-only", action="store_true",
                       help="Only run installation test")
    
    args = parser.parse_args()
    
    print_status("Setting up Drone Vehicle Detection System", "INFO")
    print("=" * 50)
    
    if args.test_only:
        success = test_installation()
        sys.exit(0 if success else 1)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not args.skip_requirements:
        if not install_requirements():
            print_status("Setup failed at requirements installation", "ERROR")
            sys.exit(1)
    
    # Download models
    if not args.skip_models:
        if not download_yolo_models():
            print_status("Model download failed - you can download manually later", "WARNING")
    
    # Set up Git hooks
    setup_git_hooks()
    
    # Test installation
    if test_installation():
        print("\n" + "=" * 50)
        print_status("Setup completed successfully!", "SUCCESS")
        print_status("You can now run: python main.py --help", "INFO")
        print_status("Or start with the Jupyter notebook: notebooks/vehicle_detection_analysis.ipynb", "INFO")
    else:
        print_status("Setup completed with warnings", "WARNING")
        print_status("Check the error messages above and fix any issues", "INFO")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Simple Vehicle Detection Runner

A simplified version that runs with minimal dependencies for quick testing.

Usage: python simple_detection.py path/to/video.mp4
"""

import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required packages are available."""
    required_packages = {
        'cv2': 'opencv-python',
        'numpy': 'numpy', 
        'yaml': 'pyyaml'
    }
    
    missing = []
    available = []
    
    for package, pip_name in required_packages.items():
        try:
            __import__(package)
            available.append(package)
        except ImportError:
            missing.append(pip_name)
    
    return available, missing

def install_missing_packages(missing):
    """Try to install missing packages."""
    if not missing:
        return True
    
    print(f"Missing packages: {', '.join(missing)}")
    print("Attempting to install...")
    
    import subprocess
    
    try:
        cmd = [sys.executable, "-m", "pip", "install"] + missing
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Packages installed successfully!")
            return True
        else:
            print(f"Installation failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"Installation error: {str(e)}")
        return False

def simple_video_info(video_path):
    """Get basic video information."""
    try:
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cap.release()
        
        return {
            'frame_count': frame_count,
            'fps': fps,
            'width': width,
            'height': height,
            'duration': frame_count / fps if fps > 0 else 0
        }
    except Exception as e:
        print(f"Error reading video: {str(e)}")
        return None

def run_basic_detection(video_path):
    """Run basic vehicle detection if ultralytics is available."""
    try:
        from ultralytics import YOLO
        import cv2
        import numpy as np
        
        print("Starting vehicle detection...")
        
        # Load model (will download if not available)
        print("Loading YOLO model...")
        model = YOLO('yolov8n.pt')  # Will auto-download
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ Cannot open video file")
            return False
        
        # Get video info
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video: {width}x{height} @ {fps:.1f} FPS")
        
        # Prepare output
        output_path = f"simple_output_{Path(video_path).stem}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        detection_count = 0
        frame_detections = 0
        
        # Try ALL classes first to see what's being detected
        print("Testing detection on first frame...")
        
        # Read first frame for testing
        ret, test_frame = cap.read()
        if ret:
            # Test with all classes and lower confidence
            test_results = model(test_frame, conf=0.1, verbose=False)
            
            print(f"Test frame detection results:")
            for result in test_results:
                boxes = result.boxes
                if boxes is not None:
                    print(f"   Found {len(boxes)} objects total")
                    for i, box in enumerate(boxes):
                        cls = int(box.cls[0].cpu().numpy())
                        conf = box.conf[0].cpu().numpy()
                        # Get class name from COCO dataset
                        class_names = model.names
                        class_name = class_names.get(cls, f'class_{cls}')
                        print(f"   Object {i+1}: {class_name} (class {cls}) - confidence: {conf:.3f}")
                else:
                    print("   No objects detected in test frame")
        
        # Reset video to beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Vehicle classes - let's also try person (0) and other transport
        vehicle_classes = [0, 1, 2, 3, 5, 6, 7, 8]  # person, bicycle, car, motorcycle, bus, train, truck, boat
        
        print("Processing frames with enhanced detection...")
        print(f"Looking for classes: {vehicle_classes}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            frame_detections = 0
            
            # Run detection every frame for better results
            results = model(frame, classes=vehicle_classes, conf=0.3, verbose=False)  # Lower confidence
            
            # Draw results
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        
                        # Get class name
                        class_name = model.names.get(cls, f'class_{cls}')
                        label = f"{class_name}: {conf:.2f}"
                        
                        # Different colors for different classes
                        colors = {
                            0: (255, 0, 0),   # person - blue
                            1: (0, 255, 255), # bicycle - yellow
                            2: (0, 255, 0),   # car - green
                            3: (255, 0, 255), # motorcycle - magenta
                            5: (0, 0, 255),   # bus - red
                            7: (255, 255, 0), # truck - cyan
                        }
                        color = colors.get(cls, (128, 128, 128))
                        
                        # Draw thicker box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                        
                        # Draw label with background
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                        cv2.rectangle(frame, (x1, y1-label_size[1]-10), (x1+label_size[0], y1), color, -1)
                        cv2.putText(frame, label, (x1, y1-5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        detection_count += 1
                        frame_detections += 1
            
            # Add frame info overlay
            info_text = f"Frame: {frame_count} | Detections: {frame_detections} | Total: {detection_count}"
            cv2.rectangle(frame, (10, 10), (600, 50), (0, 0, 0), -1)
            cv2.putText(frame, info_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Write frame
            out.write(frame)
            
            # Progress with more detail
            if frame_count % 30 == 0:
                print(f"  Frame {frame_count}, This frame: {frame_detections}, Total: {detection_count}")
        
        # Cleanup
        cap.release()
        out.release()
        
        print(f"Detection complete!")
        print(f"Results:")
        print(f"   Processed frames: {frame_count}")
        print(f"   Total detections: {detection_count}")
        print(f"   Average detections per frame: {detection_count/frame_count:.2f}")
        print(f"   Output video: {output_path}")
        
        if detection_count == 0:
            print("\nNo detections found. This could mean:")
            print("   • Video has no vehicles/people visible")
            print("   • Objects are too small (very high altitude)")
            print("   • Video quality/lighting issues")
            print("   • Need to adjust confidence threshold")
            print("   • Try viewing a sample frame manually")
        
        return True
        
    except ImportError:
        print("ultralytics not available. Install with: pip install ultralytics")
        return False
    except Exception as e:
        print(f"Detection failed: {str(e)}")
        return False

def main():
    """Main function."""
    print("Simple Drone Vehicle Detection")
    print("=" * 40)
    
    if len(sys.argv) != 2:
        print("Usage: python simple_detection.py path/to/video.mp4")
        print("\nExample:")
        print("  python simple_detection.py data/raw/14324429_2160_3840_25fps.mp4")
        return
    
    video_path = sys.argv[1]
    
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return
    
    # Check dependencies
    print("Checking dependencies...")
    available, missing = check_dependencies()
    
    if missing:
        print(f"Installing missing packages: {', '.join(missing)}")
        if not install_missing_packages(missing):
            print("Failed to install dependencies. Please install manually:")
            print(f"   pip install {' '.join(missing)}")
            return
    
    print("Dependencies OK")
    
    # Get video info
    print("Analyzing video...")
    info = simple_video_info(video_path)
    
    if info is None:
        print("Cannot read video file")
        return
    
    print(f"Video Info:")
    print(f"   Resolution: {info['width']}x{info['height']}")
    print(f"   Duration: {info['duration']:.1f} seconds")
    print(f"   Frames: {info['frame_count']}")
    print(f"   FPS: {info['fps']:.1f}")
    
    # Ask to proceed
    print(f"\nReady to process video. This may take a few minutes...")
    response = input("Continue? (y/n): ").lower().strip()
    
    if response != 'y':
        print("Cancelled.")
        return
    
    # Run detection
    success = run_basic_detection(video_path)
    
    if success:
        print("\nSuccess! Check the output video file.")
    else:
        print("\nDetection failed. Check error messages above.")

if __name__ == "__main__":
    main()

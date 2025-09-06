#!/usr/bin/env python3
"""
Example Usage of Drone Vehicle Detection System

This script demonstrates how to use the system for basic vehicle detection
and counting on drone footage.

Author: Adriel Clinton
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.detection.vehicle_detector import VehicleDetector
from src.tracking.multi_tracker import MultiTracker
from src.utils.config_manager import ConfigManager
from src.visualization.result_visualizer import ResultVisualizer


def create_sample_frame():
    """Create a sample frame with random content for demo purposes."""
    # Create a simple synthetic image
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Add some rectangular shapes to simulate vehicles
    rectangles = [
        ((50, 100), (150, 180)),   # Car 1
        ((300, 150), (380, 220)),  # Car 2
        ((500, 200), (570, 260)),  # Car 3
    ]
    
    for (x1, y1), (x2, y2) in rectangles:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
    
    return frame


def basic_detection_example():
    """Demonstrate basic vehicle detection."""
    print("=== Basic Detection Example ===")
    
    # Load configuration
    config_manager = ConfigManager("config.yaml")
    config = config_manager.get_config()
    
    # Initialize detector
    print("Initializing vehicle detector...")
    detector = VehicleDetector(config)
    
    # Create or load a sample frame
    frame = create_sample_frame()
    print(f"Processing frame of size: {frame.shape}")
    
    # Run detection
    print("Running vehicle detection...")
    detections = detector.detect(frame)
    
    # Display results
    print(f"Found {len(detections)} detections")
    for i, detection in enumerate(detections):
        print(f"  Detection {i+1}:")
        print(f"    Class: {detection.class_name}")
        print(f"    Confidence: {detection.confidence:.3f}")
        print(f"    Bbox: {detection.bbox}")
        print(f"    Center: {detection.center}")
    
    # Get detection statistics
    stats = detector.get_detection_stats(detections)
    print(f"\nDetection Statistics:")
    print(f"  Total count: {stats['total_count']}")
    print(f"  Class counts: {stats['class_counts']}")
    print(f"  Average confidence: {stats['avg_confidence']:.3f}")
    
    return frame, detections


def tracking_example():
    """Demonstrate vehicle tracking across multiple frames."""
    print("\n=== Tracking Example ===")
    
    # Load configuration
    config_manager = ConfigManager("config.yaml")
    config = config_manager.get_config()
    
    # Initialize components
    detector = VehicleDetector(config)
    tracker = MultiTracker(config)
    
    print("Processing multiple frames with tracking...")
    
    # Simulate processing multiple frames
    for frame_id in range(5):
        print(f"\nFrame {frame_id + 1}:")
        
        # Create frame (in real use, this would be from video)
        frame = create_sample_frame()
        
        # Add some noise to simulate movement
        noise = np.random.randint(-20, 20, frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Detection
        detections = detector.detect(frame)
        print(f"  Detections: {len(detections)}")
        
        # Tracking
        tracks = tracker.update(detections, frame)
        print(f"  Active tracks: {len(tracks)}")
        
        # Display track information
        for track in tracks:
            print(f"    Track {track.track_id}: {track.get_most_likely_class()} "
                  f"(length: {track.total_detections})")
    
    # Get final tracking statistics
    tracking_stats = tracker.get_tracking_statistics()
    print(f"\nFinal Tracking Statistics:")
    print(f"  Total tracks created: {tracking_stats['total_tracks_created']}")
    print(f"  Active tracks: {tracking_stats['active_tracks']}")
    print(f"  Average track length: {tracking_stats['avg_track_length']:.2f}")


def visualization_example():
    """Demonstrate result visualization."""
    print("\n=== Visualization Example ===")
    
    # Load configuration
    config_manager = ConfigManager("config.yaml")
    config = config_manager.get_config()
    
    # Initialize components
    detector = VehicleDetector(config)
    tracker = MultiTracker(config)
    visualizer = ResultVisualizer(config)
    
    # Process a frame
    frame = create_sample_frame()
    detections = detector.detect(frame)
    tracks = tracker.update(detections, frame)
    
    print(f"Creating visualizations for {len(detections)} detections and {len(tracks)} tracks")
    
    # Create annotated frame
    annotated_frame = visualizer.draw_annotations(frame, detections, tracks)
    
    # Save result (optional)
    output_path = "results/example_output.jpg"
    Path("results").mkdir(exist_ok=True)
    cv2.imwrite(output_path, annotated_frame)
    print(f"Annotated frame saved to: {output_path}")
    
    # Create visualization with different components
    detection_only = visualizer.draw_detections(frame.copy(), detections)
    tracking_only = visualizer.draw_tracks(frame.copy(), tracks)
    
    print("Individual visualizations created:")
    print("  - Detection boxes")
    print("  - Track information")
    print("  - Combined annotations")


def configuration_example():
    """Demonstrate configuration management."""
    print("\n=== Configuration Example ===")
    
    # Load default configuration
    config_manager = ConfigManager("config.yaml")
    
    # Display current settings
    print("Current configuration:")
    print(f"  Model type: {config_manager.get_value('model.type')}")
    print(f"  Confidence threshold: {config_manager.get_value('model.confidence_threshold')}")
    print(f"  Target classes: {config_manager.get_value('detection.target_classes')}")
    
    # Update configuration
    updates = {
        'model': {
            'confidence_threshold': 0.7
        },
        'visualization': {
            'show_trajectories': False
        }
    }
    
    config_manager.update_config(updates)
    print(f"\nUpdated confidence threshold to: {config_manager.get_value('model.confidence_threshold')}")
    
    # Validate configuration
    is_valid = config_manager.validate_config()
    print(f"Configuration is valid: {is_valid}")


def process_video_example(video_path=None):
    """Demonstrate video processing (if video file is available)."""
    print("\n=== Video Processing Example ===")
    
    if video_path is None or not Path(video_path).exists():
        print("No video file provided or file not found.")
        print("To test video processing:")
        print("  1. Add a video file to data/sample/")
        print("  2. Run: python example_usage.py --video data/sample/your_video.mp4")
        return
    
    print(f"Processing video: {video_path}")
    
    # Use the main DroneVehicleDetector class
    from main import DroneVehicleDetector
    
    # Initialize detector
    detector = DroneVehicleDetector("config.yaml")
    
    # Process video
    results = detector.process_video(video_path)
    
    print("Video processing completed!")
    print(f"  Input: {results['input_path']}")
    print(f"  Output: {results['output_path']}")
    print(f"  Frames processed: {results['frames_processed']}")
    print(f"  Total detections: {results['total_detections']}")


def main():
    """Main function to run all examples."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Drone Vehicle Detection Examples")
    parser.add_argument("--video", help="Path to video file for processing example")
    parser.add_argument("--skip-video", action="store_true", help="Skip video processing example")
    
    args = parser.parse_args()
    
    print("Drone Vehicle Detection System - Example Usage")
    print("=" * 60)
    
    try:
        # Run examples
        basic_detection_example()
        tracking_example()
        visualization_example()
        configuration_example()
        
        if not args.skip_video:
            process_video_example(args.video)
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("\nNext steps:")
        print("  1. Add your drone footage to data/raw/")
        print("  2. Run: python main.py --input data/raw/your_video.mp4")
        print("  3. Check results in results/ directory")
        print("  4. Explore the Jupyter notebook: notebooks/vehicle_detection_analysis.ipynb")
        
    except Exception as e:
        print(f"Error running examples: {str(e)}")
        print("Make sure you have:")
        print("  1. Installed all requirements: pip install -r requirements.txt")
        print("  2. Downloaded YOLO models: python setup.py")
        print("  3. Proper configuration in config.yaml")
        sys.exit(1)


if __name__ == "__main__":
    main()

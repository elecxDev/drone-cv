#!/usr/bin/env python3
"""
Drone Vehicle Detection and Counting System

Main entry point for processing aerial drone footage to detect and count vehicles.
This module orchestrates the entire pipeline from video input to statistical output.

Author: Adriel Clinton
License: MIT
"""

import argparse
import os
import sys
from pathlib import Path
import yaml
from loguru import logger

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.detection.vehicle_detector import VehicleDetector
from src.tracking.multi_tracker import MultiTracker  
from src.utils.video_processor import VideoProcessor
from src.utils.config_manager import ConfigManager
from src.visualization.result_visualizer import ResultVisualizer
from src.utils.statistics_manager import StatisticsManager


class DroneVehicleDetector:
    """
    Main class for drone vehicle detection and counting system.
    
    This class coordinates all components of the system including detection,
    tracking, visualization, and statistical analysis.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the drone vehicle detector.
        
        Args:
            config_path (str): Path to configuration file
        """
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()
        
        # Initialize logging
        self._setup_logging()
        
        # Initialize components
        self.detector = VehicleDetector(self.config)
        self.tracker = MultiTracker(self.config)
        self.video_processor = VideoProcessor(self.config)
        self.visualizer = ResultVisualizer(self.config)
        self.stats_manager = StatisticsManager(self.config)
        
        logger.info("Drone Vehicle Detector initialized successfully")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get('logging', {})
        log_level = log_config.get('level', 'INFO')
        log_file = log_config.get('log_file', 'logs/drone_cv.log')
        console_output = log_config.get('console_output', True)
        
        # Create logs directory
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Configure logger
        logger.remove()  # Remove default handler
        
        if console_output:
            logger.add(sys.stderr, level=log_level)
        
        logger.add(
            log_file,
            level=log_level,
            rotation="10 MB",
            retention="1 month",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {module}:{function}:{line} | {message}"
        )
    
    def process_video(self, input_path: str, output_path: str = None) -> dict:
        """
        Process a video file to detect and count vehicles.
        
        Args:
            input_path (str): Path to input video file
            output_path (str): Path for output video (optional)
            
        Returns:
            dict: Processing results including statistics
        """
        logger.info(f"Starting video processing: {input_path}")
        
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video not found: {input_path}")
        
        # Setup output path
        if output_path is None:
            filename = Path(input_path).stem
            output_dir = self.config['output']['video_output_dir']
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{filename}_processed.mp4")
        
        try:
            # Initialize video processing
            frames_generator = self.video_processor.read_video(input_path)
            writer = self.video_processor.get_video_writer(input_path, output_path)
            
            frame_count = 0
            total_detections = []
            
            # Process each frame
            for frame in frames_generator:
                frame_count += 1
                
                # Detection
                detections = self.detector.detect(frame)
                
                # Tracking
                tracks = self.tracker.update(detections, frame)
                
                # Visualization
                if self.config['output']['save_video']:
                    annotated_frame = self.visualizer.draw_annotations(
                        frame, detections, tracks
                    )
                    writer.write(annotated_frame)
                
                # Store statistics
                self.stats_manager.update_frame_stats(
                    frame_count, detections, tracks
                )
                
                total_detections.extend(detections)
                
                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count} frames")
            
            # Finalize video processing
            writer.release()
            
            # Generate final statistics
            final_stats = self.stats_manager.generate_final_report()
            
            # Save statistics
            stats_path = self._save_statistics(input_path, final_stats)
            
            results = {
                'input_path': input_path,
                'output_path': output_path,
                'frames_processed': frame_count,
                'total_detections': len(total_detections),
                'statistics': final_stats,
                'statistics_path': stats_path
            }
            
            logger.info(f"Video processing completed. Results: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            raise
    
    def _save_statistics(self, input_path: str, stats: dict) -> str:
        """Save statistics to file."""
        filename = Path(input_path).stem
        stats_dir = self.config['output']['stats_output_dir']
        os.makedirs(stats_dir, exist_ok=True)
        
        stats_path = os.path.join(stats_dir, f"{filename}_statistics.yaml")
        
        with open(stats_path, 'w') as f:
            yaml.dump(stats, f, default_flow_style=False)
        
        return stats_path
    
    def process_batch(self, input_dir: str, output_dir: str = None) -> list:
        """
        Process multiple video files in batch.
        
        Args:
            input_dir (str): Directory containing input videos
            output_dir (str): Directory for output videos
            
        Returns:
            list: List of processing results
        """
        logger.info(f"Starting batch processing: {input_dir}")
        
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        # Get video files
        video_extensions = self.config['video']['input_format']
        video_files = []
        
        for ext in video_extensions:
            video_files.extend(Path(input_dir).glob(f"*{ext}"))
        
        if not video_files:
            logger.warning(f"No video files found in {input_dir}")
            return []
        
        results = []
        
        for video_file in video_files:
            try:
                if output_dir:
                    output_path = os.path.join(
                        output_dir, 
                        f"{video_file.stem}_processed.mp4"
                    )
                else:
                    output_path = None
                
                result = self.process_video(str(video_file), output_path)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to process {video_file}: {str(e)}")
                continue
        
        logger.info(f"Batch processing completed. Processed {len(results)} videos")
        return results


def main():
    """Main entry point for command line interface."""
    parser = argparse.ArgumentParser(
        description="Drone Vehicle Detection and Counting System"
    )
    
    parser.add_argument(
        "--input", 
        required=True,
        help="Input video file or directory"
    )
    
    parser.add_argument(
        "--output",
        help="Output video file or directory"
    )
    
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Configuration file path"
    )
    
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process directory in batch mode"
    )
    
    parser.add_argument(
        "--visualize",
        action="store_true", 
        help="Show real-time visualization"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize detector
        detector = DroneVehicleDetector(args.config)
        
        # Process input
        if args.batch:
            results = detector.process_batch(args.input, args.output)
            print(f"Batch processing completed. Processed {len(results)} videos.")
        else:
            result = detector.process_video(args.input, args.output)
            print(f"Video processing completed:")
            print(f"  Input: {result['input_path']}")
            print(f"  Output: {result['output_path']}")
            print(f"  Frames: {result['frames_processed']}")
            print(f"  Detections: {result['total_detections']}")
            print(f"  Statistics: {result['statistics_path']}")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

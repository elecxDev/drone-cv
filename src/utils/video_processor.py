"""
Video Processing Utilities

Handles video input/output operations for the drone CV system.

Author: Adriel Clinton
"""

import cv2
import numpy as np
from typing import Generator, Tuple, Dict, List
import os
from pathlib import Path
from loguru import logger


class VideoProcessor:
    """Handles video processing operations."""
    
    def __init__(self, config: Dict):
        """
        Initialize video processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.video_config = config.get('video', {})
        
        # Video parameters
        self.resize_factor = self.video_config.get('resize_factor', 1.0)
        self.skip_frames = self.video_config.get('skip_frames', 0)
        self.output_format = self.video_config.get('output_format', '.mp4')
        
        logger.info("VideoProcessor initialized")
    
    def read_video(self, video_path: str) -> Generator[np.ndarray, None, None]:
        """
        Read video frames as generator.
        
        Args:
            video_path: Path to input video
            
        Yields:
            Video frames as numpy arrays
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Reading video: {video_path}")
        logger.info(f"Total frames: {total_frames}")
        logger.info(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
        logger.info(f"Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                frame_count += 1
                
                # Skip frames if configured
                if self.skip_frames > 0 and frame_count % (self.skip_frames + 1) != 1:
                    continue
                
                # Resize frame if configured
                if self.resize_factor != 1.0:
                    new_width = int(frame.shape[1] * self.resize_factor)
                    new_height = int(frame.shape[0] * self.resize_factor)
                    frame = cv2.resize(frame, (new_width, new_height))
                
                yield frame
                
        finally:
            cap.release()
            logger.info(f"Finished reading video. Processed {frame_count} frames")
    
    def get_video_info(self, video_path: str) -> Dict:
        """
        Get video information.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video information
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        info = {
            'path': video_path,
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS),
            'codec': cap.get(cv2.CAP_PROP_FOURCC)
        }
        
        cap.release()
        return info
    
    def get_video_writer(self, input_path: str, output_path: str) -> cv2.VideoWriter:
        """
        Create video writer for output.
        
        Args:
            input_path: Path to input video (for properties)
            output_path: Path for output video
            
        Returns:
            OpenCV VideoWriter object
        """
        # Get input video properties
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Apply resize factor
        if self.resize_factor != 1.0:
            width = int(width * self.resize_factor)
            height = int(height * self.resize_factor)
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Define codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Create writer
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not writer.isOpened():
            raise ValueError(f"Could not create video writer for: {output_path}")
        
        logger.info(f"Video writer created: {output_path}")
        logger.info(f"Output resolution: {width}x{height}")
        logger.info(f"Output FPS: {fps}")
        
        return writer
    
    def extract_frames(self, video_path: str, output_dir: str, 
                      frame_interval: int = 30) -> List[str]:
        """
        Extract frames from video at specified intervals.
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save frames
            frame_interval: Save every nth frame
            
        Returns:
            List of saved frame paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        saved_frames = []
        video_name = Path(video_path).stem
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    frame_filename = f"{video_name}_frame_{frame_count:06d}.jpg"
                    frame_path = os.path.join(output_dir, frame_filename)
                    
                    # Apply resize if configured
                    if self.resize_factor != 1.0:
                        new_width = int(frame.shape[1] * self.resize_factor)
                        new_height = int(frame.shape[0] * self.resize_factor)
                        frame = cv2.resize(frame, (new_width, new_height))
                    
                    cv2.imwrite(frame_path, frame)
                    saved_frames.append(frame_path)
                
                frame_count += 1
                
        finally:
            cap.release()
        
        logger.info(f"Extracted {len(saved_frames)} frames from {video_path}")
        return saved_frames
    
    def create_video_from_frames(self, frame_paths: List[str], 
                                output_path: str, fps: float = 30.0) -> str:
        """
        Create video from frame images.
        
        Args:
            frame_paths: List of frame image paths
            output_path: Output video path
            fps: Frames per second
            
        Returns:
            Path to created video
        """
        if not frame_paths:
            raise ValueError("No frame paths provided")
        
        # Read first frame to get dimensions
        first_frame = cv2.imread(frame_paths[0])
        if first_frame is None:
            raise ValueError(f"Could not read first frame: {frame_paths[0]}")
        
        height, width = first_frame.shape[:2]
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        try:
            for frame_path in frame_paths:
                frame = cv2.imread(frame_path)
                if frame is not None:
                    writer.write(frame)
                else:
                    logger.warning(f"Could not read frame: {frame_path}")
        
        finally:
            writer.release()
        
        logger.info(f"Created video from {len(frame_paths)} frames: {output_path}")
        return output_path
    
    def resize_video(self, input_path: str, output_path: str, 
                    scale_factor: float) -> str:
        """
        Resize video by scale factor.
        
        Args:
            input_path: Input video path
            output_path: Output video path
            scale_factor: Scale factor for resizing
            
        Returns:
            Path to resized video
        """
        # Get input video info
        info = self.get_video_info(input_path)
        new_width = int(info['width'] * scale_factor)
        new_height = int(info['height'] * scale_factor)
        
        # Create writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            output_path, fourcc, info['fps'], (new_width, new_height)
        )
        
        # Process frames
        for frame in self.read_video(input_path):
            resized_frame = cv2.resize(frame, (new_width, new_height))
            writer.write(resized_frame)
        
        writer.release()
        logger.info(f"Resized video saved: {output_path}")
        
        return output_path
    
    @staticmethod
    def is_video_file(file_path: str) -> bool:
        """Check if file is a video file based on extension."""
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.m4v'}
        return Path(file_path).suffix.lower() in video_extensions

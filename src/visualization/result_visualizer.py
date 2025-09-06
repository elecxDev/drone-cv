"""
Result Visualization Module

Handles visualization of detection and tracking results for the drone CV system.

Author: Adriel Clinton
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from loguru import logger

from ..detection.vehicle_detector import Detection
from ..tracking.multi_tracker import Track


class ResultVisualizer:
    """Handles visualization of detection and tracking results."""
    
    def __init__(self, config: Dict):
        """
        Initialize result visualizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.vis_config = config.get('visualization', {})
        
        # Visualization settings
        self.show_detections = self.vis_config.get('show_detections', True)
        self.show_tracks = self.vis_config.get('show_tracks', True)
        self.show_count = self.vis_config.get('show_count', True)
        self.show_density_map = self.vis_config.get('show_density_map', False)
        self.show_trajectories = self.vis_config.get('show_trajectories', True)
        self.trail_length = self.vis_config.get('trail_length', 30)
        
        # Style settings
        self.font_scale = self.vis_config.get('font_scale', 0.6)
        self.thickness = self.vis_config.get('thickness', 2)
        self.colors = self.vis_config.get('colors', {})
        
        # Default colors (BGR format for OpenCV)
        self.default_colors = {
            'car': (0, 255, 0),      # Green
            'motorcycle': (255, 0, 0),  # Blue
            'bus': (0, 0, 255),      # Red
            'truck': (255, 255, 0),  # Cyan
            'unknown': (128, 128, 128)  # Gray
        }
        
        logger.info("ResultVisualizer initialized")
    
    def draw_annotations(self, frame: np.ndarray, detections: List[Detection], 
                        tracks: List[Track]) -> np.ndarray:
        """
        Draw all annotations on frame.
        
        Args:
            frame: Input frame
            detections: List of detections
            tracks: List of tracks
            
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        # Draw detections
        if self.show_detections:
            annotated_frame = self.draw_detections(annotated_frame, detections)
        
        # Draw tracks
        if self.show_tracks:
            annotated_frame = self.draw_tracks(annotated_frame, tracks)
        
        # Draw trajectories
        if self.show_trajectories:
            annotated_frame = self.draw_trajectories(annotated_frame, tracks)
        
        # Draw count information
        if self.show_count:
            annotated_frame = self.draw_count_info(annotated_frame, detections, tracks)
        
        return annotated_frame
    
    def draw_detections(self, frame: np.ndarray, 
                       detections: List[Detection]) -> np.ndarray:
        """
        Draw detection bounding boxes on frame.
        
        Args:
            frame: Input frame
            detections: List of detections
            
        Returns:
            Frame with detection annotations
        """
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            color = self._get_class_color(detection.class_name)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.thickness)
            
            # Draw confidence and class label
            label = f"{detection.class_name}: {detection.confidence:.2f}"
            label_size = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.thickness
            )[0]
            
            # Label background
            cv2.rectangle(
                frame,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color,
                -1
            )
            
            # Label text
            cv2.putText(
                frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                (255, 255, 255),
                self.thickness
            )
            
            # Draw center point
            center = detection.center
            cv2.circle(frame, center, 3, color, -1)
        
        return frame
    
    def draw_tracks(self, frame: np.ndarray, tracks: List[Track]) -> np.ndarray:
        """
        Draw track information on frame.
        
        Args:
            frame: Input frame
            tracks: List of tracks
            
        Returns:
            Frame with track annotations
        """
        for track in tracks:
            if not track.is_active:
                continue
            
            detection = track.current_detection
            x1, y1, x2, y2 = detection.bbox
            
            # Get track color (unique for each track)
            color = self._get_track_color(track.track_id)
            
            # Draw bounding box with track color
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.thickness + 1)
            
            # Draw track ID and class
            class_name = track.get_most_likely_class()
            label = f"ID:{track.track_id} {class_name}"
            
            # Track label background
            label_size = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.thickness
            )[0]
            
            cv2.rectangle(
                frame,
                (x1, y2 + 5),
                (x1 + label_size[0], y2 + label_size[1] + 15),
                color,
                -1
            )
            
            # Track label text
            cv2.putText(
                frame,
                label,
                (x1, y2 + label_size[1] + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                (255, 255, 255),
                self.thickness
            )
            
            # Draw velocity vector
            vx, vy = track.get_velocity()
            if abs(vx) > 1 or abs(vy) > 1:  # Only draw if significant movement
                current_pos = track.current_position
                end_pos = (
                    int(current_pos[0] + vx * 10),
                    int(current_pos[1] + vy * 10)
                )
                cv2.arrowedLine(frame, current_pos, end_pos, color, 2)
        
        return frame
    
    def draw_trajectories(self, frame: np.ndarray, tracks: List[Track]) -> np.ndarray:
        """
        Draw track trajectories on frame.
        
        Args:
            frame: Input frame
            tracks: List of tracks
            
        Returns:
            Frame with trajectory annotations
        """
        for track in tracks:
            if not track.is_active or len(track.positions) < 2:
                continue
            
            color = self._get_track_color(track.track_id)
            positions = list(track.positions)
            
            # Limit to recent positions
            recent_positions = positions[-self.trail_length:]
            
            # Draw trajectory lines
            for i in range(1, len(recent_positions)):
                # Fade older positions
                alpha = i / len(recent_positions)
                line_color = tuple(int(c * alpha) for c in color)
                
                cv2.line(
                    frame, 
                    recent_positions[i-1], 
                    recent_positions[i], 
                    line_color, 
                    max(1, int(self.thickness * alpha))
                )
            
            # Draw position markers
            for i, pos in enumerate(recent_positions[::5]):  # Every 5th position
                alpha = (i * 5) / len(recent_positions)
                marker_color = tuple(int(c * alpha) for c in color)
                cv2.circle(frame, pos, 2, marker_color, -1)
        
        return frame
    
    def draw_count_info(self, frame: np.ndarray, detections: List[Detection], 
                       tracks: List[Track]) -> np.ndarray:
        """
        Draw count information on frame.
        
        Args:
            frame: Input frame
            detections: List of detections
            tracks: List of tracks
            
        Returns:
            Frame with count information
        """
        # Count vehicles by class
        detection_counts = defaultdict(int)
        track_counts = defaultdict(int)
        
        for detection in detections:
            detection_counts[detection.class_name] += 1
        
        for track in tracks:
            if track.is_active:
                class_name = track.get_most_likely_class()
                track_counts[class_name] += 1
        
        # Create info panel
        info_height = 150
        info_width = 300
        
        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (info_width, info_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw detection counts
        y_offset = 35
        cv2.putText(
            frame, "DETECTIONS:", (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
        
        y_offset += 20
        for class_name, count in detection_counts.items():
            color = self._get_class_color(class_name)
            cv2.putText(
                frame, f"{class_name}: {count}", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
            )
            y_offset += 15
        
        # Draw track counts
        y_offset += 10
        cv2.putText(
            frame, "ACTIVE TRACKS:", (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
        
        y_offset += 20
        for class_name, count in track_counts.items():
            color = self._get_class_color(class_name)
            cv2.putText(
                frame, f"{class_name}: {count}", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
            )
            y_offset += 15
        
        return frame
    
    def create_density_heatmap(self, frame: np.ndarray, 
                              detections: List[Detection]) -> np.ndarray:
        """
        Create density heatmap overlay.
        
        Args:
            frame: Input frame
            detections: List of detections
            
        Returns:
            Frame with heatmap overlay
        """
        if not detections:
            return frame
        
        # Create heatmap
        h, w = frame.shape[:2]
        heatmap = np.zeros((h, w), dtype=np.float32)
        
        # Add detection centers to heatmap
        for detection in detections:
            center = detection.center
            # Add Gaussian blob around detection center
            cv2.circle(heatmap, center, 30, 1.0, -1)
        
        # Apply Gaussian blur
        heatmap = cv2.GaussianBlur(heatmap, (51, 51), 0)
        
        # Normalize
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # Convert to color
        heatmap_color = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        
        # Blend with original frame
        result = cv2.addWeighted(frame, 0.7, heatmap_color, 0.3, 0)
        
        return result
    
    def _get_class_color(self, class_name: str) -> Tuple[int, int, int]:
        """Get color for vehicle class."""
        return self.colors.get(class_name, self.default_colors.get(class_name, (128, 128, 128)))
    
    def _get_track_color(self, track_id: int) -> Tuple[int, int, int]:
        """Generate unique color for track ID."""
        # Use track ID to generate consistent colors
        np.random.seed(track_id)
        color = tuple(np.random.randint(50, 255, 3).tolist())
        return color
    
    def create_statistics_visualization(self, stats: Dict) -> np.ndarray:
        """
        Create visualization of statistics.
        
        Args:
            stats: Statistics dictionary
            
        Returns:
            Statistics visualization image
        """
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Vehicle Detection Statistics', fontsize=16)
        
        # Class distribution pie chart
        if 'class_distribution' in stats and stats['class_distribution']['detections']:
            class_data = stats['class_distribution']['detections']
            axes[0, 0].pie(class_data.values(), labels=class_data.keys(), autopct='%1.1f%%')
            axes[0, 0].set_title('Vehicle Class Distribution')
        
        # Detection count over time
        if 'temporal_analysis' in stats and stats['temporal_analysis'].get('windows'):
            windows = stats['temporal_analysis']['windows']
            frames = [w['start_frame'] for w in windows]
            detections = [w['avg_detections'] for w in windows]
            axes[0, 1].plot(frames, detections)
            axes[0, 1].set_title('Detection Count Over Time')
            axes[0, 1].set_xlabel('Frame')
            axes[0, 1].set_ylabel('Average Detections')
        
        # Performance metrics
        if 'performance_metrics' in stats:
            perf = stats['performance_metrics']
            metrics = ['FPS', 'Avg Processing Time', 'Total Frames']
            values = [
                perf.get('fps_estimate', 0),
                perf.get('avg_processing_time', 0) * 1000,  # Convert to ms
                perf.get('frames_processed', 0)
            ]
            axes[1, 0].bar(metrics, values)
            axes[1, 0].set_title('Performance Metrics')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Density metrics
        if 'density_metrics' in stats:
            density = stats['density_metrics']
            metrics = ['Current', 'Average', 'Maximum', 'Minimum']
            values = [
                density.get('current_density', 0),
                density.get('avg_density', 0),
                density.get('max_density', 0),
                density.get('min_density', 0)
            ]
            axes[1, 1].bar(metrics, values)
            axes[1, 1].set_title('Traffic Density')
        
        plt.tight_layout()
        
        # Convert to image
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
        # Convert RGB to BGR for OpenCV
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        return img
    
    def save_frame_with_annotations(self, frame: np.ndarray, detections: List[Detection],
                                   tracks: List[Track], output_path: str):
        """Save annotated frame to file."""
        annotated_frame = self.draw_annotations(frame, detections, tracks)
        cv2.imwrite(output_path, annotated_frame)
        logger.info(f"Annotated frame saved: {output_path}")

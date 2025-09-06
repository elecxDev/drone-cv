"""
Statistics Manager

Handles collection and analysis of detection and tracking statistics.

Author: Adriel Clinton
"""

import numpy as np
from typing import Dict, List, Any
from collections import defaultdict, deque
import json
import os
from datetime import datetime
from loguru import logger

from ..detection.vehicle_detector import Detection
from ..tracking.multi_tracker import Track


class StatisticsManager:
    """Manages statistics collection and reporting for vehicle detection and tracking."""
    
    def __init__(self, config: Dict):
        """
        Initialize statistics manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.analysis_config = config.get('analysis', {})
        
        # Statistics storage
        self.frame_stats = []
        self.detection_history = deque(maxlen=1000)
        self.tracking_history = deque(maxlen=1000)
        
        # Counters
        self.total_vehicles_detected = 0
        self.total_vehicles_tracked = 0
        self.class_detections = defaultdict(int)
        self.class_tracks = defaultdict(int)
        
        # Timing
        self.start_time = datetime.now()
        self.processing_times = []
        
        logger.info("StatisticsManager initialized")
    
    def update_frame_stats(self, frame_id: int, detections: List[Detection], 
                          tracks: List[Track]):
        """
        Update statistics for a single frame.
        
        Args:
            frame_id: Frame number
            detections: List of detections in frame
            tracks: List of active tracks in frame
        """
        frame_start_time = datetime.now()
        
        # Count detections by class
        detection_counts = defaultdict(int)
        detection_confidences = []
        
        for detection in detections:
            detection_counts[detection.class_name] += 1
            detection_confidences.append(detection.confidence)
            self.class_detections[detection.class_name] += 1
        
        # Count tracks by class  
        track_counts = defaultdict(int)
        track_lengths = []
        
        for track in tracks:
            class_name = track.get_most_likely_class()
            track_counts[class_name] += 1
            track_lengths.append(track.total_detections)
            
            # Update class tracks (only for new tracks)
            if track.start_frame == frame_id:
                self.class_tracks[class_name] += 1
        
        # Frame statistics
        frame_stat = {
            'frame_id': frame_id,
            'timestamp': datetime.now().isoformat(),
            'detections': {
                'total': len(detections),
                'by_class': dict(detection_counts),
                'avg_confidence': np.mean(detection_confidences) if detection_confidences else 0.0,
                'min_confidence': min(detection_confidences) if detection_confidences else 0.0,
                'max_confidence': max(detection_confidences) if detection_confidences else 0.0
            },
            'tracks': {
                'total': len(tracks),
                'by_class': dict(track_counts),
                'avg_length': np.mean(track_lengths) if track_lengths else 0.0,
                'active_tracks': len([t for t in tracks if t.is_active])
            }
        }
        
        self.frame_stats.append(frame_stat)
        
        # Update totals
        self.total_vehicles_detected += len(detections)
        
        # Store in history
        self.detection_history.append(len(detections))
        self.tracking_history.append(len(tracks))
        
        # Record processing time
        processing_time = (datetime.now() - frame_start_time).total_seconds()
        self.processing_times.append(processing_time)
    
    def calculate_density_metrics(self) -> Dict[str, float]:
        """Calculate traffic density metrics."""
        if not self.detection_history:
            return {}
        
        recent_detections = list(self.detection_history)[-100:]  # Last 100 frames
        
        return {
            'current_density': recent_detections[-1] if recent_detections else 0,
            'avg_density': np.mean(recent_detections),
            'max_density': max(recent_detections),
            'min_density': min(recent_detections),
            'density_std': np.std(recent_detections),
            'density_trend': self._calculate_trend(recent_detections)
        }
    
    def calculate_traffic_flow_metrics(self) -> Dict[str, Any]:
        """Calculate traffic flow metrics from tracks."""
        if not self.frame_stats:
            return {}
        
        # Extract track data
        all_tracks_data = []
        for frame_stat in self.frame_stats[-100:]:  # Last 100 frames
            all_tracks_data.append(frame_stat['tracks']['total'])
        
        if not all_tracks_data:
            return {}
        
        # Calculate flow metrics
        flow_metrics = {
            'vehicles_per_minute': self._estimate_vehicles_per_minute(),
            'peak_traffic_count': max(all_tracks_data),
            'avg_active_tracks': np.mean(all_tracks_data),
            'traffic_consistency': 1.0 - (np.std(all_tracks_data) / np.mean(all_tracks_data)) if np.mean(all_tracks_data) > 0 else 0.0
        }
        
        return flow_metrics
    
    def _estimate_vehicles_per_minute(self) -> float:
        """Estimate vehicles per minute based on new track creation rate."""
        if len(self.frame_stats) < 2:
            return 0.0
        
        # Count new tracks in recent frames
        recent_frames = self.frame_stats[-60:]  # Last 60 frames (assuming ~30 fps = 2 seconds)
        new_tracks = 0
        
        frame_ids = [stat['frame_id'] for stat in recent_frames]
        
        # This is a simplified estimation - in practice you'd track actual new track creation
        # For now, estimate based on detection rate
        total_detections = sum(stat['detections']['total'] for stat in recent_frames)
        minutes_elapsed = len(recent_frames) / (30 * 60)  # Assuming 30 fps
        
        if minutes_elapsed > 0:
            return total_detections / minutes_elapsed * 0.1  # Rough estimate
        return 0.0
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from values."""
        if len(values) < 2:
            return "stable"
        
        # Simple linear regression slope
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.1:
            return "increasing"
        elif slope < -0.1:
            return "decreasing"
        else:
            return "stable"
    
    def calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate system performance metrics."""
        if not self.processing_times:
            return {}
        
        return {
            'avg_processing_time': np.mean(self.processing_times),
            'max_processing_time': max(self.processing_times),
            'min_processing_time': min(self.processing_times),
            'fps_estimate': 1.0 / np.mean(self.processing_times) if np.mean(self.processing_times) > 0 else 0,
            'total_processing_time': sum(self.processing_times),
            'frames_processed': len(self.processing_times)
        }
    
    def generate_class_distribution(self) -> Dict[str, Dict[str, int]]:
        """Generate distribution of vehicle classes."""
        return {
            'detections': dict(self.class_detections),
            'tracks': dict(self.class_tracks)
        }
    
    def generate_temporal_analysis(self) -> Dict[str, Any]:
        """Generate temporal analysis of traffic patterns."""
        if len(self.frame_stats) < 10:
            return {}
        
        # Group by time windows (e.g., every 30 frames)
        window_size = 30
        windows = []
        
        for i in range(0, len(self.frame_stats), window_size):
            window_frames = self.frame_stats[i:i + window_size]
            
            if window_frames:
                window_detections = [f['detections']['total'] for f in window_frames]
                window_tracks = [f['tracks']['total'] for f in window_frames]
                
                windows.append({
                    'start_frame': window_frames[0]['frame_id'],
                    'end_frame': window_frames[-1]['frame_id'],
                    'avg_detections': np.mean(window_detections),
                    'avg_tracks': np.mean(window_tracks),
                    'peak_detections': max(window_detections),
                    'peak_tracks': max(window_tracks)
                })
        
        return {
            'window_size': window_size,
            'windows': windows,
            'peak_window': max(windows, key=lambda x: x['peak_detections']) if windows else None
        }
    
    def generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final report."""
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()
        
        report = {
            'summary': {
                'start_time': self.start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'total_duration_seconds': total_duration,
                'frames_processed': len(self.frame_stats),
                'total_detections': self.total_vehicles_detected,
                'unique_tracks': len(self.class_tracks)
            },
            'density_metrics': self.calculate_density_metrics(),
            'flow_metrics': self.calculate_traffic_flow_metrics(),
            'performance_metrics': self.calculate_performance_metrics(),
            'class_distribution': self.generate_class_distribution(),
            'temporal_analysis': self.generate_temporal_analysis()
        }
        
        # Add detection rate
        if total_duration > 0:
            report['summary']['detections_per_second'] = self.total_vehicles_detected / total_duration
            report['summary']['frames_per_second'] = len(self.frame_stats) / total_duration
        
        return report
    
    def save_detailed_stats(self, output_path: str):
        """Save detailed frame-by-frame statistics."""
        stats_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_frames': len(self.frame_stats),
                'config': self.config
            },
            'frame_stats': self.frame_stats,
            'summary': self.generate_final_report()
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(stats_data, f, indent=2, default=str)
        
        logger.info(f"Detailed statistics saved: {output_path}")
    
    def get_real_time_stats(self) -> Dict[str, Any]:
        """Get current real-time statistics."""
        if not self.frame_stats:
            return {}
        
        recent_stats = self.frame_stats[-10:]  # Last 10 frames
        
        current_detections = recent_stats[-1]['detections']['total'] if recent_stats else 0
        current_tracks = recent_stats[-1]['tracks']['total'] if recent_stats else 0
        
        avg_recent_detections = np.mean([s['detections']['total'] for s in recent_stats])
        avg_recent_tracks = np.mean([s['tracks']['total'] for s in recent_stats])
        
        return {
            'current_frame': recent_stats[-1]['frame_id'] if recent_stats else 0,
            'current_detections': current_detections,
            'current_tracks': current_tracks,
            'avg_recent_detections': avg_recent_detections,
            'avg_recent_tracks': avg_recent_tracks,
            'total_processed': len(self.frame_stats),
            'processing_fps': 1.0 / np.mean(self.processing_times[-10:]) if len(self.processing_times) >= 10 else 0
        }
    
    def reset_stats(self):
        """Reset all statistics."""
        self.frame_stats.clear()
        self.detection_history.clear()
        self.tracking_history.clear()
        self.total_vehicles_detected = 0
        self.total_vehicles_tracked = 0
        self.class_detections.clear()
        self.class_tracks.clear()
        self.start_time = datetime.now()
        self.processing_times.clear()
        
        logger.info("Statistics reset")

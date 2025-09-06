"""
Multi-Object Tracking Module

This module provides vehicle tracking capabilities for maintaining object
identities across frames in drone footage.

Author: Adriel Clinton
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import cv2
from loguru import logger
from collections import defaultdict, deque

from ..detection.vehicle_detector import Detection


class Track:
    """Represents a single tracked object."""
    
    def __init__(self, track_id: int, detection: Detection, frame_id: int):
        """
        Initialize a track.
        
        Args:
            track_id: Unique identifier for this track
            detection: Initial detection
            frame_id: Frame number where track started
        """
        self.track_id = track_id
        self.detections = [detection]
        self.start_frame = frame_id
        self.last_frame = frame_id
        self.is_active = True
        self.missed_frames = 0
        
        # Tracking state
        self.positions = deque(maxlen=30)  # Store last 30 positions
        self.positions.append(detection.center)
        
        # Statistics
        self.total_detections = 1
        self.class_votes = defaultdict(int)
        self.class_votes[detection.class_name] += 1
        
    def update(self, detection: Detection, frame_id: int):
        """Update track with new detection."""
        self.detections.append(detection)
        self.positions.append(detection.center)
        self.last_frame = frame_id
        self.missed_frames = 0
        self.total_detections += 1
        self.class_votes[detection.class_name] += 1
    
    def predict_next_position(self) -> Tuple[int, int]:
        """Predict next position based on movement history."""
        if len(self.positions) < 2:
            return self.positions[-1]
        
        # Simple linear prediction based on last two positions
        last_pos = self.positions[-1]
        second_last_pos = self.positions[-2]
        
        dx = last_pos[0] - second_last_pos[0]
        dy = last_pos[1] - second_last_pos[1]
        
        predicted_x = last_pos[0] + dx
        predicted_y = last_pos[1] + dy
        
        return (predicted_x, predicted_y)
    
    def get_velocity(self) -> Tuple[float, float]:
        """Calculate velocity vector."""
        if len(self.positions) < 2:
            return (0.0, 0.0)
        
        recent_positions = list(self.positions)[-5:]  # Use last 5 positions
        
        if len(recent_positions) < 2:
            return (0.0, 0.0)
        
        # Calculate average velocity
        velocities_x = []
        velocities_y = []
        
        for i in range(1, len(recent_positions)):
            dx = recent_positions[i][0] - recent_positions[i-1][0]
            dy = recent_positions[i][1] - recent_positions[i-1][1]
            velocities_x.append(dx)
            velocities_y.append(dy)
        
        avg_vx = np.mean(velocities_x) if velocities_x else 0.0
        avg_vy = np.mean(velocities_y) if velocities_y else 0.0
        
        return (avg_vx, avg_vy)
    
    def get_most_likely_class(self) -> str:
        """Get the most frequently detected class for this track."""
        return max(self.class_votes.items(), key=lambda x: x[1])[0]
    
    @property
    def current_detection(self) -> Detection:
        """Get the most recent detection."""
        return self.detections[-1]
    
    @property
    def current_position(self) -> Tuple[int, int]:
        """Get current position."""
        return self.positions[-1] if self.positions else (0, 0)


class MultiTracker:
    """
    Multi-object tracker for vehicle tracking in aerial footage.
    
    Uses a simple but effective tracking algorithm based on:
    1. Distance-based association
    2. Kalman filtering for prediction
    3. Track management for handling disappearing objects
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the multi-tracker.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.tracking_config = config.get('tracking', {})
        
        # Tracking parameters
        self.max_disappeared = self.tracking_config.get('max_disappeared', 30)
        self.max_distance = self.tracking_config.get('max_distance', 50)
        self.match_threshold = self.tracking_config.get('match_threshold', 0.8)
        
        # State
        self.tracks = {}  # track_id -> Track
        self.next_track_id = 1
        self.frame_count = 0
        
        # Statistics
        self.total_tracks_created = 0
        self.total_tracks_finished = 0
        
        logger.info("MultiTracker initialized")
    
    def update(self, detections: List[Detection], frame: np.ndarray) -> List[Track]:
        """
        Update tracking with new detections.
        
        Args:
            detections: New detections from current frame
            frame: Current frame (for debugging/visualization)
            
        Returns:
            List of active tracks
        """
        self.frame_count += 1
        
        # Handle empty detections
        if not detections:
            self._handle_missed_detections()
            return list(self.tracks.values())
        
        # Get active tracks
        active_tracks = [track for track in self.tracks.values() if track.is_active]
        
        # Associate detections with existing tracks
        matched_pairs, unmatched_detections, unmatched_tracks = self._associate_detections_to_tracks(
            detections, active_tracks
        )
        
        # Update matched tracks
        for detection_idx, track_id in matched_pairs:
            self.tracks[track_id].update(detections[detection_idx], self.frame_count)
        
        # Handle unmatched tracks (increase missed frames)
        for track in unmatched_tracks:
            track.missed_frames += 1
            if track.missed_frames > self.max_disappeared:
                track.is_active = False
                self.total_tracks_finished += 1
                logger.debug(f"Track {track.track_id} finished (max missed frames)")
        
        # Create new tracks for unmatched detections
        for detection_idx in unmatched_detections:
            self._create_new_track(detections[detection_idx])
        
        # Return active tracks
        return [track for track in self.tracks.values() if track.is_active]
    
    def _associate_detections_to_tracks(self, detections: List[Detection], 
                                      tracks: List[Track]) -> Tuple[List[Tuple[int, int]], 
                                                                   List[int], 
                                                                   List[Track]]:
        """
        Associate detections with existing tracks using Hungarian algorithm or greedy matching.
        
        Returns:
            Tuple of (matched_pairs, unmatched_detections, unmatched_tracks)
        """
        if not tracks or not detections:
            return [], list(range(len(detections))), tracks
        
        # Calculate distance matrix
        distance_matrix = np.zeros((len(detections), len(tracks)))
        
        for i, detection in enumerate(detections):
            for j, track in enumerate(tracks):
                # Calculate distance between detection and track's predicted position
                predicted_pos = track.predict_next_position()
                detection_pos = detection.center
                
                distance = np.sqrt(
                    (detection_pos[0] - predicted_pos[0]) ** 2 + 
                    (detection_pos[1] - predicted_pos[1]) ** 2
                )
                
                # Add class consistency bonus
                if detection.class_name == track.get_most_likely_class():
                    distance *= 0.8  # 20% bonus for same class
                
                distance_matrix[i, j] = distance
        
        # Simple greedy matching (can be replaced with Hungarian algorithm)
        matched_pairs = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = tracks.copy()
        
        # Sort by distance and match greedily
        matches = []
        for i in range(len(detections)):
            for j in range(len(tracks)):
                if distance_matrix[i, j] < self.max_distance:
                    matches.append((distance_matrix[i, j], i, j))
        
        matches.sort(key=lambda x: x[0])  # Sort by distance
        
        used_detections = set()
        used_tracks = set()
        
        for distance, det_idx, track_idx in matches:
            if det_idx not in used_detections and track_idx not in used_tracks:
                matched_pairs.append((det_idx, tracks[track_idx].track_id))
                used_detections.add(det_idx)
                used_tracks.add(track_idx)
        
        # Update unmatched lists
        unmatched_detections = [i for i in range(len(detections)) if i not in used_detections]
        unmatched_tracks = [track for i, track in enumerate(tracks) if i not in used_tracks]
        
        return matched_pairs, unmatched_detections, unmatched_tracks
    
    def _create_new_track(self, detection: Detection):
        """Create a new track for an unmatched detection."""
        track = Track(self.next_track_id, detection, self.frame_count)
        self.tracks[self.next_track_id] = track
        self.next_track_id += 1
        self.total_tracks_created += 1
        
        logger.debug(f"Created new track {track.track_id} for {detection.class_name}")
    
    def _handle_missed_detections(self):
        """Handle case when no detections are found in current frame."""
        for track in self.tracks.values():
            if track.is_active:
                track.missed_frames += 1
                if track.missed_frames > self.max_disappeared:
                    track.is_active = False
                    self.total_tracks_finished += 1
    
    def get_active_tracks(self) -> List[Track]:
        """Get all currently active tracks."""
        return [track for track in self.tracks.values() if track.is_active]
    
    def get_track_by_id(self, track_id: int) -> Optional[Track]:
        """Get track by ID."""
        return self.tracks.get(track_id)
    
    def get_tracking_statistics(self) -> Dict:
        """Get comprehensive tracking statistics."""
        active_tracks = self.get_active_tracks()
        
        # Count vehicles by class
        class_counts = defaultdict(int)
        for track in active_tracks:
            class_name = track.get_most_likely_class()
            class_counts[class_name] += 1
        
        # Calculate average track lengths
        all_tracks = list(self.tracks.values())
        track_lengths = [track.total_detections for track in all_tracks]
        
        stats = {
            'frame_count': self.frame_count,
            'active_tracks': len(active_tracks),
            'total_tracks_created': self.total_tracks_created,
            'total_tracks_finished': self.total_tracks_finished,
            'class_counts': dict(class_counts),
            'avg_track_length': np.mean(track_lengths) if track_lengths else 0,
            'max_track_length': max(track_lengths) if track_lengths else 0,
            'min_track_length': min(track_lengths) if track_lengths else 0
        }
        
        return stats
    
    def visualize_tracks(self, frame: np.ndarray, tracks: List[Track] = None) -> np.ndarray:
        """
        Visualize tracks on frame.
        
        Args:
            frame: Input frame
            tracks: List of tracks to visualize (default: all active tracks)
            
        Returns:
            Frame with track visualization
        """
        if tracks is None:
            tracks = self.get_active_tracks()
        
        vis_frame = frame.copy()
        
        for track in tracks:
            if not track.is_active:
                continue
            
            # Draw current detection
            detection = track.current_detection
            x1, y1, x2, y2 = detection.bbox
            
            # Get color for track
            color = self._get_track_color(track.track_id)
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw track ID and class
            label = f"ID:{track.track_id} {track.get_most_likely_class()}"
            cv2.putText(
                vis_frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
            
            # Draw trajectory
            if len(track.positions) > 1:
                points = list(track.positions)
                for i in range(1, len(points)):
                    cv2.line(vis_frame, points[i-1], points[i], color, 2)
            
            # Draw velocity vector
            if len(track.positions) >= 2:
                vx, vy = track.get_velocity()
                current_pos = track.current_position
                end_pos = (
                    int(current_pos[0] + vx * 5),
                    int(current_pos[1] + vy * 5)
                )
                cv2.arrowedLine(vis_frame, current_pos, end_pos, color, 2)
        
        return vis_frame
    
    def _get_track_color(self, track_id: int) -> Tuple[int, int, int]:
        """Get a unique color for each track ID."""
        # Generate colors based on track ID
        np.random.seed(track_id)
        color = tuple(np.random.randint(0, 255, 3).tolist())
        return color
    
    def reset(self):
        """Reset tracker state."""
        self.tracks.clear()
        self.next_track_id = 1
        self.frame_count = 0
        self.total_tracks_created = 0
        self.total_tracks_finished = 0
        
        logger.info("MultiTracker reset")

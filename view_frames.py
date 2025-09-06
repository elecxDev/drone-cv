#!/usr/bin/env python3
"""
Quick Frame Viewer

View sample frames from your video to see what's in it.

Usage: python view_frames.py path/to/video.mp4
"""

import sys
import os
import cv2
from pathlib import Path

def extract_sample_frames(video_path, num_frames=5):
    """Extract sample frames from video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Cannot open video")
        return False
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"📺 Video has {total_frames} frames at {fps:.1f} FPS")
    
    # Extract frames at regular intervals
    frame_indices = [int(total_frames * i / num_frames) for i in range(num_frames)]
    
    output_dir = Path("sample_frames")
    output_dir.mkdir(exist_ok=True)
    
    saved_frames = []
    
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if ret:
            timestamp = frame_idx / fps
            filename = f"frame_{i+1}_at_{timestamp:.1f}s.jpg"
            filepath = output_dir / filename
            cv2.imwrite(str(filepath), frame)
            saved_frames.append(str(filepath))
            print(f"📸 Saved: {filename}")
    
    cap.release()
    
    print(f"\n✅ Extracted {len(saved_frames)} sample frames to 'sample_frames/' folder")
    print("📁 Open the folder to view the frames and see what's in your video!")
    
    return True

def main():
    if len(sys.argv) != 2:
        print("Usage: python view_frames.py path/to/video.mp4")
        print("Example: python view_frames.py data/raw/14324429_2160_3840_25fps.mp4")
        return
    
    video_path = sys.argv[1]
    
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return
    
    print("🎬 Frame Extractor")
    print("=" * 30)
    
    extract_sample_frames(video_path)

if __name__ == "__main__":
    main()

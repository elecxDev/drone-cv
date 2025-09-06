#!/usr/bin/env python3
"""
Sample Video Downloader

Downloads sample drone footage for testing the vehicle detection system.

Usage: python download_sample_videos.py
"""

import os
import requests
from pathlib import Path
import sys


def download_file(url, filename, destination_dir="data/sample"):
    """Download a file from URL to destination."""
    # Create destination directory
    Path(destination_dir).mkdir(parents=True, exist_ok=True)
    
    destination = Path(destination_dir) / filename
    
    print(f"Downloading {filename}...")
    print(f"URL: {url}")
    
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
        
        print(f"\nDownloaded: {destination}")
        return str(destination)
        
    except Exception as e:
        print(f"Error downloading {filename}: {str(e)}")
        return None


def get_sample_videos():
    """Download sample videos for testing."""
    
    print("Drone Vehicle Detection - Sample Video Downloader")
    print("=" * 55)
    
    # Sample video URLs (these are examples - you'll need real URLs)
    sample_videos = [
        {
            "name": "highway_traffic_sample.mp4",
            "url": "https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4",  # Example URL
            "description": "Highway traffic from aerial view"
        },
        # Add more sample videos here with real URLs from Pexels/Pixabay
    ]
    
    print("Available sample videos:")
    for i, video in enumerate(sample_videos, 1):
        print(f"{i}. {video['name']} - {video['description']}")
    
    print("\nNote: These are example URLs. For actual videos, please:")
    print("1. Visit Pexels.com and search 'drone traffic'")
    print("2. Download high-quality MP4 files")
    print("3. Place them in the data/sample/ directory")
    print("4. Or use the manual download instructions below")
    
    return sample_videos


def manual_download_instructions():
    """Provide manual download instructions."""
    
    instructions = """
    MANUAL DOWNLOAD INSTRUCTIONS
    ============================
    
    1. PEXELS (Recommended):
       - Go to: https://www.pexels.com/search/videos/drone%20traffic/
       - Find videos with:
         * Aerial/bird's eye view of roads
         * Clear vehicle visibility
         * Good lighting conditions
         * 1080p resolution or higher
       - Click on video → Download → Choose quality
       - Save to: data/sample/
    
    2. PIXABAY:
       - Go to: https://pixabay.com/videos/search/aerial%20traffic/
       - Filter by: HD quality
       - Download free videos
       - Save to: data/sample/
    
    3. RECOMMENDED VIDEO CHARACTERISTICS:
       - Resolution: 1080p (1920x1080) or higher
       - Format: MP4
       - Duration: 30 seconds to 5 minutes
       - View: Perpendicular to ground (bird's eye view)
       - Content: Roads with moving vehicles
       - Lighting: Daylight, clear conditions
    
    4. SPECIFIC SEARCH TERMS:
       - "drone highway traffic"
       - "aerial view cars"
       - "bird eye view road"
       - "drone footage traffic intersection"
       - "aerial traffic jam"
    
    5. YOUTUBE-DL METHOD (if allowed):
       pip install yt-dlp
       yt-dlp "https://youtube.com/watch?v=VIDEO_ID"
       
    6. AFTER DOWNLOADING:
       - Place videos in: data/sample/
       - Test with: python main.py --input data/sample/your_video.mp4
       - Or use: python example_usage.py --video data/sample/your_video.mp4
    """
    
    return instructions


def create_sample_data_info():
    """Create info file about sample data requirements."""
    
    info_content = """# Sample Data Information

## What kind of videos work best?

### ✅ GOOD Videos:
- Aerial/drone footage taken from 50-200 meters height
- Perpendicular view of roads (bird's eye view)
- Clear daylight conditions
- Multiple vehicles visible
- Stable camera (minimal shaking)
- 1080p resolution or higher
- MP4 format

### ❌ AVOID:
- Ground-level footage
- Angled/tilted views
- Poor lighting (night, fog)
- Very high altitude (vehicles too small)
- Shaky/unstable footage
- Low resolution (480p or lower)

## Recommended Sources:

1. **Pexels**: https://www.pexels.com/search/videos/drone%20traffic/
2. **Pixabay**: https://pixabay.com/videos/search/aerial%20traffic/
3. **Unsplash**: https://unsplash.com/videos (search "drone traffic")

## Test Videos:

After downloading, test with:
```bash
python main.py --input data/sample/your_video.mp4
```

## Sample Video Examples:

Good search terms:
- "drone highway traffic"
- "aerial view intersection" 
- "bird eye view cars"
- "drone footage parking lot"
- "aerial traffic surveillance"

## File Naming:

Use descriptive names:
- `highway_morning_traffic.mp4`
- `intersection_busy_hour.mp4`
- `parking_lot_aerial.mp4`
- `city_traffic_drone.mp4`
"""
    
    # Write info file
    info_path = Path("data/sample/README.md")
    info_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(info_path, 'w') as f:
        f.write(info_content)
    
    print(f"Sample data info created: {info_path}")


def main():
    """Main function."""
    
    # Create sample directory
    Path("data/sample").mkdir(parents=True, exist_ok=True)
    
    # Show available samples (example URLs)
    sample_videos = get_sample_videos()
    
    print("\n" + "=" * 55)
    print(manual_download_instructions())
    
    # Create info file
    create_sample_data_info()
    
    print("\n" + "=" * 55)
    print("QUICK START:")
    print("1. Go to https://www.pexels.com/search/videos/drone%20traffic/")
    print("2. Download a video (choose 1080p MP4)")
    print("3. Save to data/sample/")
    print("4. Run: python main.py --input data/sample/your_video.mp4")


if __name__ == "__main__":
    main()

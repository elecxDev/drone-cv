# Quick Video Test Commands

## After downloading a video to data/sample/:

# Test the basic detection system
python main.py --input data/sample/your_video.mp4

# Run with specific output location  
python main.py --input data/sample/your_video.mp4 --output results/test_output.mp4

# Try the example script
python example_usage.py --video data/sample/your_video.mp4

# Process multiple videos in batch
python main.py --input data/sample/ --batch

## Best video characteristics:
- Resolution: 1080p (1920x1080) or higher  
- Duration: 30 seconds to 2 minutes (for testing)
- Format: MP4
- View: Straight down (bird's eye view)
- Content: Roads with moving vehicles
- Quality: Clear daylight, stable footage

## Good video names to search on Pexels:
- "drone highway traffic"
- "aerial road cars" 
- "bird eye view traffic"
- "drone intersection"
- "aerial parking lot"

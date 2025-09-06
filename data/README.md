# Data Directory

This directory contains the input data for the drone vehicle detection system.

## Structure

```
data/
├── raw/           # Raw drone footage files
├── processed/     # Processed videos and images  
└── sample/        # Sample data for testing
```

## Raw Data (`raw/`)

Place your drone footage files here. Supported formats:

- **Video**: .mp4, .avi, .mov, .mkv, .flv, .wmv, .m4v
- **Images**: .jpg, .jpeg, .png, .bmp, .tiff

### Recommended Video Specifications

- **Resolution**: 1080p or higher for best results
- **Frame Rate**: 30 FPS minimum
- **Format**: MP4 with H.264 encoding
- **Duration**: Any length (system processes frame-by-frame)

### Drone Footage Guidelines

For optimal vehicle detection:

1. **Altitude**: 50-200 meters above ground
2. **Angle**: Perpendicular to ground (bird's eye view)
3. **Lighting**: Good daylight conditions
4. **Stability**: Minimal camera shake/vibration
5. **Coverage**: Clear view of roads/traffic areas

## Sample Data (`sample/`)

This directory should contain small sample files for testing and development.

### Adding Sample Data

1. Copy a short video clip (10-30 seconds) to `data/sample/`
2. Rename it descriptively (e.g., `highway_traffic.mp4`)
3. Test the system with: `python main.py --input data/sample/highway_traffic.mp4`

### Sample Data Sources

Free drone footage for testing:

- [Pexels](https://www.pexels.com/search/videos/drone%20traffic/) - Free stock videos
- [Pixabay](https://pixabay.com/videos/search/drone/) - Free drone footage
- [YouTube](https://www.youtube.com) - Creative Commons licensed videos

## Processed Data (`processed/`)

This directory stores processed outputs:

- Annotated videos with detection boxes
- Extracted frames with annotations
- Intermediate processing results

### File Naming Convention

```
{original_name}_processed.mp4     # Annotated video
{original_name}_frame_{num}.jpg   # Extracted frames
{original_name}_stats.json        # Processing statistics
```

## Data Organization Tips

### Project Structure
```
data/
├── raw/
│   ├── project1/
│   │   ├── morning_traffic.mp4
│   │   └── evening_traffic.mp4
│   └── project2/
│       └── highway_monitoring.mp4
├── processed/
│   ├── project1/
│   └── project2/
└── sample/
    └── test_clip.mp4
```

### Batch Processing

For processing multiple files:

```bash
# Process all videos in a directory
python main.py --input data/raw/project1/ --output data/processed/project1/ --batch
```

## Storage Requirements

### Disk Space Guidelines

- **Raw 1080p video**: ~100MB per minute
- **Processed video**: ~120MB per minute (with annotations)
- **Extracted frames**: ~1MB per frame at 1080p
- **Statistics files**: ~1KB per video

### Recommendations

- Use external storage for large datasets
- Compress videos when possible (H.264/H.265)
- Clean up processed files periodically
- Backup important footage before processing

## Data Privacy and Security

### Important Considerations

1. **Privacy**: Ensure compliance with local privacy laws
2. **Consent**: Obtain necessary permissions for filming
3. **Storage**: Secure storage of sensitive footage
4. **Sharing**: Be cautious when sharing processed results

### Data Handling Best Practices

- Remove or blur license plates if required
- Avoid storing personal information
- Use secure file permissions
- Implement data retention policies

## Troubleshooting

### Common Issues

1. **File not found**: Check file paths and permissions
2. **Unsupported format**: Convert to supported video format
3. **Corrupted files**: Verify file integrity
4. **Large files**: Consider splitting or compressing

### Performance Tips

1. **Storage speed**: Use SSD for better I/O performance
2. **Network storage**: Avoid processing over slow networks
3. **File size**: Balance quality vs. processing speed
4. **Batch size**: Adjust based on available memory

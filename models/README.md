# Models Directory

This directory contains the machine learning models used for vehicle detection.

## Structure

```
models/
├── yolo/           # YOLO model files
│   ├── yolov8n.pt  # YOLOv8 nano model (download required)
│   ├── yolov8s.pt  # YOLOv8 small model (download required)
│   └── yolov8m.pt  # YOLOv8 medium model (download required)
└── custom/         # Custom trained models
    └── README.md   # Instructions for custom models
```

## YOLO Models

### Download Instructions

The YOLO models are not included in the repository due to size constraints. Download them using the following commands:

```bash
# Navigate to the yolo directory
cd models/yolo

# Download YOLOv8 models
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt
```

Or download manually from: https://github.com/ultralytics/yolov8/releases

### Model Comparison

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| YOLOv8n | 6.2MB | Fastest | Good | Real-time processing |
| YOLOv8s | 21.5MB | Fast | Better | Balanced performance |
| YOLOv8m | 49.7MB | Medium | Best | High accuracy required |

### Configuration

Update the model path in `config.yaml`:

```yaml
model:
  type: "yolov8"
  weights_path: "models/yolo/yolov8n.pt"  # Change as needed
  confidence_threshold: 0.5
  iou_threshold: 0.45
  device: "auto"
```

## Custom Models

For training custom models on your drone footage:

1. Prepare your dataset in YOLO format
2. Train using YOLOv8 training scripts
3. Save the trained model in `models/custom/`
4. Update configuration to use your custom model

## Supported Classes

The system is configured to detect these vehicle classes from the COCO dataset:

- **Class 2**: Car
- **Class 3**: Motorcycle  
- **Class 5**: Bus
- **Class 7**: Truck

To add more classes, update the `target_classes` in `config.yaml`.

## Performance Notes

- **GPU**: Models will automatically use GPU if available (CUDA/MPS)
- **CPU**: All models work on CPU but with reduced performance
- **Memory**: Larger models require more VRAM/RAM
- **Batch Processing**: Larger models benefit more from batch processing

## Troubleshooting

### Model Loading Issues

1. **File not found**: Ensure the model file exists in the specified path
2. **CUDA errors**: Check GPU compatibility and drivers
3. **Out of memory**: Try a smaller model or reduce batch size

### Performance Issues

1. **Slow inference**: Consider using a smaller model
2. **Low accuracy**: Try a larger model or adjust thresholds
3. **High memory usage**: Reduce input resolution or batch size

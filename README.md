# Drone Vehicle Detection and Counting System

## Project Overview
This project processes drone-captured aerial videos using OpenCV to detect and count moving vehicles on roads. It uses object detection and tracking techniques to estimate traffic density and flow.

## Problem Statement
Process drone-captured aerial videos using OpenCV to detect and count moving vehicles on roads. Use object detection and tracking techniques to estimate traffic density and flow.

## Features
- Vehicle detection in aerial footage using YOLO
- Multi-object tracking for counting vehicles
- Traffic density estimation
- Real-time processing capabilities
- Visual output with annotated frames
- Statistical analysis and reporting

## Project Structure
```
drone-cv/
├── data/
│   ├── raw/                 # Raw drone footage
│   ├── processed/           # Processed videos/images
│   └── sample/              # Sample data for testing
├── models/
│   ├── yolo/               # YOLO model files
│   └── custom/             # Custom trained models
├── src/
│   ├── detection/          # Vehicle detection modules
│   ├── tracking/           # Object tracking modules
│   ├── utils/              # Utility functions
│   └── visualization/      # Visualization and output
├── notebooks/              # Jupyter notebooks for analysis
├── tests/                  # Unit tests
├── results/                # Output results and reports
├── requirements.txt        # Python dependencies
└── config.yaml            # Configuration settings
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/elecxDev/drone-cv.git
cd drone-cv
```

2. Create a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
# source venv/bin/activate  # On Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download YOLO weights (instructions in models/README.md)

## Usage

### Basic Usage
```python
from src.main import DroneVehicleDetector

detector = DroneVehicleDetector('config.yaml')
results = detector.process_video('data/raw/sample_video.mp4')
```

### Command Line Interface
```bash
python main.py --input data/raw/sample_video.mp4 --output results/output_video.mp4
```

## Configuration
Edit `config.yaml` to adjust detection parameters, tracking settings, and output preferences.

## Results
- Processed videos with vehicle annotations
- Traffic count statistics
- Density analysis reports
- Performance metrics

## License
MIT License - see LICENSE file for details

## Contributing
Please read CONTRIBUTING.md for contribution guidelines.

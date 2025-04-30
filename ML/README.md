# Real-Time Object Detection and Tracking

This project implements real-time detection of object disappearance and new object placement in video streams using computer vision and deep learning techniques.

## Features

- Real-time object detection using YOLO
- Object tracking with ByteTrack
- Detection of object disappearance
- Detection of new object placement
- High FPS performance optimization

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- See `requirements.txt` for Python dependencies

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script:
```bash
python src/main.py --input <video_path> --output <output_path>
```

## Project Structure

```
├── src/
│   ├── main.py              # Main entry point
│   ├── detector.py          # Object detection module
│   ├── tracker.py           # Object tracking module
│   └── utils.py             # Utility functions
├── models/                  # Model weights and configurations
├── data/                    # Sample videos and test data
├── requirements.txt         # Python dependencies
└── README.md               # Project documentation
```

## Performance Optimization

- TensorRT acceleration
- Frame skipping for high FPS
- Model pruning
- GPU optimization

## License

MIT License 
import torch
import torch.serialization
from ultralytics import YOLO

def download_model():
    # Add YOLO model to safe globals
    torch.serialization.add_safe_globals(['DetectionModel'])
    
    try:
        # Download YOLOv8n model
        model = YOLO('yolov8n.pt')
        print("Model downloaded successfully!")
    except Exception as e:
        print(f"Error downloading model: {str(e)}")

if __name__ == '__main__':
    download_model() 
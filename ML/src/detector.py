import torch
from ultralytics import YOLO
import numpy as np
import os
from .utils import CLASSES

class ObjectDetector:
    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.5):
        """Initialize the YOLO object detector.
        
        Args:
            model_path (str): Path to the YOLO model weights
            conf_threshold (float): Confidence threshold for detections
        """
        try:
            self.model = YOLO(model_path)
            self.conf_threshold = conf_threshold
            self.class_names = [CLASSES.get(i, f'class_{i}') for i in range(len(CLASSES))]
            print(f"Successfully loaded YOLO model from {model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {str(e)}")

    def detect(self, frame):
        """Detect objects in a frame.
        
        Args:
            frame (numpy.ndarray): Input frame in BGR format
            
        Returns:
            numpy.ndarray: Array of detections in format [x1, y1, x2, y2, confidence, class_id]
        """
        try:
            # Run inference
            results = self.model(frame, verbose=False)[0]
            
            # Process results
            detections = []
            for r in results.boxes.data.cpu().numpy():
                x1, y1, x2, y2, conf, class_id = r
                if conf >= self.conf_threshold:
                    detections.append([x1, y1, x2, y2, conf, class_id])
            
            return np.array(detections) if detections else np.empty((0, 6))
            
        except Exception as e:
            print(f"Error during detection: {str(e)}")
            return np.empty((0, 6))  # Return empty array on error 
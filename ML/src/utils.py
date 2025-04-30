import cv2
import numpy as np

# YOLO class names
CLASSES = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    4: 'airplane',
    5: 'bus',
    6: 'train',
    7: 'truck',
    8: 'boat',
    9: 'traffic light',
    10: 'fire hydrant',
    11: 'stop sign',
    12: 'parking meter',
    13: 'bench',
    14: 'bird',
    15: 'cat',
    16: 'dog',
    17: 'horse',
    18: 'sheep',
    19: 'cow',
    20: 'elephant'
}

def draw_results(frame, detections_or_tracks):
    """
    Draw detection and tracking results on the frame.
    
    Args:
        frame (numpy.ndarray): Input frame
        detections_or_tracks: Either a list of tracked objects or numpy array of detections
        
    Returns:
        numpy.ndarray: Frame with drawn results
    """
    # Make a copy of the frame
    output = frame.copy()
    
    # Colors for different classes
    colors = {
        0: (0, 255, 0),    # Person - Green
        1: (255, 0, 0),    # Bicycle - Blue
        2: (0, 0, 255),    # Car - Red
        3: (255, 255, 0),  # Motorcycle - Cyan
        4: (255, 0, 255),  # Bus - Magenta
        5: (0, 255, 255),  # Truck - Yellow
    }
    
    # Check if input is numpy array (raw detections) or list (tracked objects)
    is_raw_detections = isinstance(detections_or_tracks, np.ndarray)
    
    # Draw each object
    for obj in detections_or_tracks:
        if is_raw_detections:
            # For raw detections
            x1, y1, x2, y2, conf, class_id = map(int, obj)
            track_id = None
        else:
            # For tracked objects
            box = obj.box
            x1, y1, x2, y2 = map(int, box)
            class_id = obj.class_id
            track_id = obj.tracker_id
            conf = obj.confidence
        
        # Get color for this class
        color = colors.get(class_id, (128, 128, 128))  # Default to gray if class not in colors
        
        # Draw bounding box
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        
        # Get class name
        class_name = CLASSES.get(class_id, f'class_{class_id}')
        
        # Create label
        if track_id is not None:
            label = f"ID:{track_id} {class_name} {conf:.2f}"
        else:
            label = f"{class_name} {conf:.2f}"
        
        # Calculate label background size
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # Draw label background
        cv2.rectangle(output, 
                     (x1, y1 - label_h - 10), 
                     (x1 + label_w, y1),
                     color, -1)  # -1 fills the rectangle
        
        # Draw label text in white
        cv2.putText(output, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Add frame information
    cv2.putText(output, f"Objects detected: {len(detections_or_tracks)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return output

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1 (list): First bounding box [x1, y1, x2, y2]
        box2 (list): Second bounding box [x1, y1, x2, y2]
        
    Returns:
        float: IoU value
    """
    # Calculate intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    # Calculate IoU
    iou = intersection / union if union > 0 else 0
    
    return iou 
import cv2
import numpy as np
from typing import List, Dict, Tuple

class Visualizer:
    def __init__(self, class_names: List[str], trail_length: int = 30):
        """Initialize visualizer for object detection and tracking.
        
        Args:
            class_names: List of class names for detection
            trail_length: Maximum length of tracking trails to display
        """
        self.class_names = class_names
        self.trail_length = trail_length
        self.colors = self._generate_colors(len(class_names))
        
    def _generate_colors(self, num_classes: int) -> List[tuple]:
        """Generate distinct colors for different classes."""
        colors = []
        for i in range(num_classes):
            # Use HSV color space for more distinct colors
            hue = i / num_classes
            color = tuple(int(x * 255) for x in self._hsv2rgb(hue, 0.8, 0.9))
            colors.append(color)
        return colors
        
    def _hsv2rgb(self, h: float, s: float, v: float) -> tuple:
        """Convert HSV to RGB color."""
        if s == 0.0:
            return (v, v, v)
        i = int(h * 6.0)
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        if i == 0:
            return (v, t, p)
        if i == 1:
            return (q, v, p)
        if i == 2:
            return (p, v, t)
        if i == 3:
            return (p, q, v)
        if i == 4:
            return (t, p, v)
        if i == 5:
            return (v, p, q)
        
    def draw_detections(self, frame: np.ndarray, tracks: List, track_history: Dict) -> np.ndarray:
        """Draw detection boxes, labels and tracking info."""
        vis_frame = frame.copy()
        
        # Draw tracking lines first (under boxes)
        for track_id, history in track_history.items():
            if len(history) > 1:
                points = np.array(history).astype(np.int32)
                cv2.polylines(vis_frame, [points], False, (0, 255, 255), 2)
        
        for track in tracks:
            track_id = int(track[0])
            x1, y1, x2, y2 = map(int, track[1:5])
            conf = track[5]
            class_id = int(track[6])
            
            # Get color for this class
            color = self.colors[class_id]
            
            # Draw box with thicker lines
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 3)
            
            # Prepare label with class name, track ID and confidence
            label = f"{self.class_names[class_id]} #{track_id} ({conf:.2f})"
            
            # Draw better looking label background
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(vis_frame, (x1, y1-label_h-10), (x1+label_w, y1), color, -1)
            
            # Draw white text for better contrast
            cv2.putText(vis_frame, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return vis_frame
        
    def draw_stats(self, frame: np.ndarray, fps: float, frame_num: int, 
                   total_frames: int, processing_time: float) -> np.ndarray:
        """Draw performance stats on frame."""
        # Create semi-transparent overlay for stats
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (300, 100), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
        
        # Draw stats with better visibility
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Frame: {frame_num}/{total_frames}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return frame 
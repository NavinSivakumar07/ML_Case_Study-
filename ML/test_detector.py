import cv2
from src.detector import ObjectDetector
from src.utils import draw_results

def main():
    # Initialize the detector
    detector = ObjectDetector(model_path='yolov8n.pt', conf_threshold=0.5)
    
    # Open the video file
    video_path = 'data/WhatsApp Video 2025-04-25 at 13.06.28_7699c2d1.mp4'
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection
        detections = detector.detect(frame)
        
        # Draw results on frame
        output_frame = draw_results(frame, detections)
        
        # Display the frame
        cv2.imshow('Object Detection', output_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 
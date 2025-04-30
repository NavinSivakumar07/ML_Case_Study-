import cv2
import numpy as np
import argparse
import time
import os
from pathlib import Path
from typing import Tuple, Optional

from detector import ObjectDetector
from tracker import ObjectTracker
from visualizer import Visualizer
from object_analytics import ObjectAnalytics

def get_video_info(cap: cv2.VideoCapture) -> Tuple[int, int, int, float]:
    """Get video metadata.
    
    Args:
        cap: OpenCV video capture object
        
    Returns:
        Tuple of (width, height, total_frames, fps)
    """
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    return width, height, total_frames, fps

def create_video_writer(output_path: str, width: int, height: int, 
                       fps: float) -> Optional[cv2.VideoWriter]:
    """Create video writer with basic settings."""
    try:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Ensure valid dimensions
        width = int(width)
        height = int(height)
        if width % 2 == 1:
            width -= 1
        if height % 2 == 1:
            height -= 1
            
        # Try different codecs
        codecs = [
            ('XVID', '.avi'),
            ('MJPG', '.avi'),
            ('mp4v', '.mp4')
        ]
        
        for codec, ext in codecs:
            try:
                output_file = str(Path(output_path).with_suffix(ext))
                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = cv2.VideoWriter(
                    filename=output_file,
                    fourcc=fourcc,
                    fps=fps,
                    frameSize=(width, height)
                )
                
                if writer.isOpened():
                    print(f"Successfully created video writer: {output_file}")
                    return writer
                    
            except Exception as e:
                print(f"Failed with codec {codec}: {str(e)}")
                continue
                
        print("Failed to create video writer")
        return None
        
    except Exception as e:
        print(f"Error creating video writer: {str(e)}")
        return None

def create_output_folder(output_path: str) -> str:
    """Create and return path for output frames."""
    # Create frames directory
    frames_dir = os.path.join(os.path.dirname(output_path), 'frames')
    if not os.path.exists(frames_dir):
        os.makedirs(frames_dir)
    return frames_dir

def draw_analytics_overlay(frame: np.ndarray, new_objects: set, missing_objects: set,
                         analytics: ObjectAnalytics, class_names: list) -> np.ndarray:
    """Draw analytics information on frame."""
    # Add semi-transparent overlay for alerts
    overlay = frame.copy()
    
    # Draw alerts for new objects
    y_offset = 50
    for track_id in new_objects:
        obj_info = analytics.get_object_info(track_id)
        if obj_info:
            class_name = class_names[obj_info["class"]]
            x1, y1, x2, y2 = obj_info["position"]
            
            # Draw green box around new object
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw alert text
            text = f"New {class_name} detected!"
            cv2.putText(frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
    
    # Draw alerts for missing objects
    for track_id in missing_objects:
        text = f"Object #{track_id} missing!"
        cv2.putText(frame, text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_offset += 30
    
    return frame

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Input video path')
    parser.add_argument('--output', type=str, required=True, help='Output video path')
    parser.add_argument('--weights', type=str, required=True, help='YOLOv5 weights path')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--skip-frames', type=int, default=1, help='Process every nth frame')
    args = parser.parse_args()
    
    # Initialize video capture
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error: Could not open video: {args.input}")
        return
        
    # Get video info
    width, height, total_frames, fps = get_video_info(cap)
    print(f"Input video: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Create video writer
    writer = create_video_writer(args.output, width, height, fps/args.skip_frames)
    if writer is None:
        return
        
    # Initialize components
    detector = ObjectDetector(args.weights, args.conf_thres)
    tracker = ObjectTracker()
    visualizer = Visualizer(detector.class_names)
    analytics = ObjectAnalytics(persistence_threshold=15, missing_threshold=15)
    
    frame_count = 0
    processing_times = []
    last_log_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Skip frames if needed
            if frame_count % args.skip_frames != 0:
                frame_count += 1
                continue
                
            try:
                # Run detection
                detections = detector.detect(frame)
                
                # Update tracks
                tracks = tracker.update(detections, frame_count)
                
                # Update analytics
                new_objects, missing_objects = analytics.update(tracks, frame_count)
                
                # Visualize results
                vis_frame = visualizer.draw_detections(frame, tracks, tracker.get_track_history())
                
                # Draw analytics overlay
                vis_frame = draw_analytics_overlay(vis_frame, new_objects, missing_objects,
                                                analytics, detector.class_names)
                
                # Calculate stats
                current_time = time.time()
                if processing_times:
                    current_fps = len(processing_times) / (current_time - processing_times[0])
                else:
                    current_fps = 0
                
                # Draw stats
                vis_frame = visualizer.draw_stats(vis_frame, current_fps, frame_count,
                                                total_frames, 0)
                
                # Write frame
                writer.write(vis_frame)
                
                # Update processing times
                processing_times.append(current_time)
                if len(processing_times) > 30:
                    processing_times.pop(0)
                
                # Print progress
                if current_time - last_log_time > 5:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames}) "
                          f"FPS: {current_fps:.1f}")
                    last_log_time = current_time
                    
            except Exception as e:
                print(f"Error processing frame {frame_count}: {str(e)}")
                
            frame_count += 1
            
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"Error during processing: {str(e)}")
    finally:
        # Cleanup
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        
        print("\nProcessing complete")

if __name__ == '__main__':
    main() 
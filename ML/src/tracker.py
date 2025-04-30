from collections import defaultdict
import numpy as np
from utils import calculate_iou

class Track:
    def __init__(self, box, confidence, class_id, track_id):
        self.box = box  # [x1, y1, x2, y2]
        self.confidence = confidence
        self.class_id = class_id
        self.tracker_id = track_id
        self.age = 0
        self.total_visible = 1
        self.consecutive_invisible = 0
        self.last_position = box

class ObjectTracker:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.next_id = 1
        self.track_history = defaultdict(list)
        self.max_history_len = 30

    def _match_detections_to_tracks(self, detections):
        if len(self.tracks) == 0 or len(detections) == 0:
            return [], np.arange(len(detections))

        # Calculate IoU between all detections and tracks
        iou_matrix = np.zeros((len(detections), len(self.tracks)))
        for d_idx, detection in enumerate(detections):
            for t_idx, track in enumerate(self.tracks):
                iou_matrix[d_idx, t_idx] = calculate_iou(detection[:4], track.box)

        # Find matches using greedy assignment
        matched_indices = []
        unmatched_detections = list(range(len(detections)))
        
        while True:
            # Find highest IoU
            if len(unmatched_detections) == 0:
                break
                
            highest_iou = 0
            best_match = (-1, -1)
            
            for d_idx in unmatched_detections:
                for t_idx in range(len(self.tracks)):
                    if iou_matrix[d_idx, t_idx] > highest_iou:
                        highest_iou = iou_matrix[d_idx, t_idx]
                        best_match = (d_idx, t_idx)
            
            if highest_iou >= self.iou_threshold:
                d_idx, t_idx = best_match
                matched_indices.append((d_idx, t_idx))
                unmatched_detections.remove(d_idx)
            else:
                break

        return matched_indices, np.array(unmatched_detections)

    def update(self, detections, frame_id):
        """Update tracks using detections from current frame.
        
        Args:
            detections (numpy.ndarray): Array of detections [x1,y1,x2,y2,conf,class_id]
            frame_id (int): Current frame ID
            
        Returns:
            list: List of tracks
        """
        # Convert detections to list if numpy array
        if isinstance(detections, np.ndarray):
            detections = detections.tolist()
            
        # Match detections to existing tracks
        matches, unmatched_det_indices = self._match_detections_to_tracks(detections)
        
        # Update matched tracks
        for det_idx, track_idx in matches:
            detection = detections[det_idx]
            track = self.tracks[track_idx]
            
            # Update track
            track.box = detection[:4]
            track.confidence = detection[4]
            track.age += 1
            track.total_visible += 1
            track.consecutive_invisible = 0
            
            # Update history
            self.track_history[track.tracker_id].append(
                (frame_id, tuple(track.box))
            )
            if len(self.track_history[track.tracker_id]) > self.max_history_len:
                self.track_history[track.tracker_id].pop(0)
        
        # Create new tracks for unmatched detections
        for idx in unmatched_det_indices:
            detection = detections[idx]
            new_track = Track(
                box=detection[:4],
                confidence=detection[4],
                class_id=int(detection[5]),
                track_id=self.next_id
            )
            self.tracks.append(new_track)
            self.next_id += 1
        
        # Update unmatched tracks
        unmatched_tracks = []
        for i, track in enumerate(self.tracks):
            if not any(i == t_idx for _, t_idx in matches):
                track.consecutive_invisible += 1
                track.age += 1
                unmatched_tracks.append(i)
        
        # Remove old tracks
        self.tracks = [track for i, track in enumerate(self.tracks)
                      if track.consecutive_invisible <= self.max_age]
        
        # Convert tracks to output format
        output_tracks = []
        for track in self.tracks:
            if track.age >= self.min_hits:
                output_tracks.append([
                    track.tracker_id,
                    *track.box,
                    track.confidence,
                    track.class_id
                ])
        
        return output_tracks

    def get_track_history(self):
        """Get motion history for all tracks.
        
        Returns:
            dict: Dictionary of track histories
        """
        return self.track_history 
import numpy as np
from typing import Dict, List, Set, Tuple
import time

class ObjectAnalytics:
    def __init__(self, persistence_threshold: int = 30, missing_threshold: int = 30):
        """Initialize object analytics.
        
        Args:
            persistence_threshold: Number of frames an object must be present to be considered stable
            missing_threshold: Number of frames before declaring an object as missing
        """
        self.persistence_threshold = persistence_threshold
        self.missing_threshold = missing_threshold
        
        # Track objects and their states
        self.stable_objects = {}  # {track_id: {"last_seen": frame_num, "position": (x1,y1,x2,y2), "class": class_id}}
        self.new_objects = set()  # track_ids of newly detected objects
        self.missing_objects = set()  # track_ids of missing objects
        self.object_history = {}  # {track_id: count_of_appearances}
        
    def update(self, tracks: List, frame_num: int) -> Tuple[Set, Set]:
        """Update object states and detect missing/new objects.
        
        Args:
            tracks: List of tracks [track_id, x1, y1, x2, y2, score, class_id]
            frame_num: Current frame number
            
        Returns:
            Tuple of (new_object_ids, missing_object_ids)
        """
        # Reset detection sets
        self.new_objects.clear()
        self.missing_objects.clear()
        
        # Get current track IDs
        current_track_ids = {int(track[0]) for track in tracks}
        
        # Check for missing objects
        for track_id, info in list(self.stable_objects.items()):
            if track_id not in current_track_ids:
                frames_missing = frame_num - info["last_seen"]
                if frames_missing >= self.missing_threshold:
                    self.missing_objects.add(track_id)
                    del self.stable_objects[track_id]
                    
        # Update existing and detect new objects
        for track in tracks:
            track_id = int(track[0])
            x1, y1, x2, y2 = map(int, track[1:5])
            class_id = int(track[6])
            
            # Update object history
            if track_id not in self.object_history:
                self.object_history[track_id] = 1
            else:
                self.object_history[track_id] += 1
            
            # Check if object should be considered stable
            if (track_id not in self.stable_objects and 
                self.object_history[track_id] >= self.persistence_threshold):
                self.stable_objects[track_id] = {
                    "last_seen": frame_num,
                    "position": (x1, y1, x2, y2),
                    "class": class_id
                }
                self.new_objects.add(track_id)
            
            # Update last seen for existing stable objects
            elif track_id in self.stable_objects:
                self.stable_objects[track_id]["last_seen"] = frame_num
                self.stable_objects[track_id]["position"] = (x1, y1, x2, y2)
        
        return self.new_objects, self.missing_objects
    
    def get_object_info(self, track_id: int) -> Dict:
        """Get information about a tracked object.
        
        Args:
            track_id: ID of the tracked object
            
        Returns:
            Dictionary containing object information or None if not found
        """
        return self.stable_objects.get(track_id, None)
        
    def get_all_stable_objects(self) -> Dict:
        """Get all currently stable objects.
        
        Returns:
            Dictionary of all stable objects
        """
        return self.stable_objects.copy() 
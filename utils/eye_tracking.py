import numpy as np
import cv2
from scipy.spatial import distance

class EyeTracker:
    def __init__(self):
        # MediaPipe face mesh landmark indices for eyes
        self.left_eye_landmarks = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.right_eye_landmarks = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # IMPROVED: More accurate eye landmarks for EAR calculation
        self.left_eye_points = [33, 160, 158, 133, 153, 144]  # Outer, top, bottom, inner, top, bottom
        self.right_eye_points = [362, 385, 387, 263, 373, 380]  # Outer, top, bottom, inner, top, bottom
        
        # EAR thresholds - ADJUSTED for better accuracy
        self.ear_threshold = 0.22  # Slightly reduced for better sensitivity
        self.consecutive_frames = 4  # Increased for stability
        
        # State tracking
        self.ear_history = []
        self.drowsy_frame_count = 0
        self.last_ear = 0.3  # Better default value
        
        # NEW: Stability and smoothing
        self.stable_drowsy_count = 0
        self.stable_threshold = 3
        
    def calculate_ear(self, landmarks, head_pose_data=None):
        """Calculate Eye Aspect Ratio (EAR) with head pose compensation"""
        try:
            # Get left and right eye EAR
            left_ear = self._calculate_single_eye_ear(landmarks, self.left_eye_points)
            right_ear = self._calculate_single_eye_ear(landmarks, self.right_eye_points)
            
            # Average of both eyes
            ear = (left_ear + right_ear) / 2.0
            
            # NEW: Compensate for head pose effects
            if head_pose_data:
                ear = self._compensate_for_head_pose(ear, head_pose_data)
            
            # SMOOTHING: Apply moving average to reduce noise
            self.ear_history.append(ear)
            if len(self.ear_history) > 8:  # Keep last 8 values for smoothing
                self.ear_history.pop(0)
                
            # Use smoothed EAR
            smoothed_ear = sum(self.ear_history) / len(self.ear_history)
            self.last_ear = smoothed_ear
            return smoothed_ear
            
        except Exception as e:
            print(f"Error calculating EAR: {e}")
            return self.last_ear
            
    def _calculate_single_eye_ear(self, landmarks, eye_points):
        """Calculate EAR for a single eye - IMPROVED accuracy"""
        try:
            # Get eye landmark coordinates with validation
            eye_coords = []
            for point_idx in eye_points:
                if point_idx < len(landmarks):
                    eye_coords.append(landmarks[point_idx])
                else:
                    # Return default if landmarks are missing
                    return 0.25
                    
            if len(eye_coords) < 6:
                return 0.25
                
            # Convert to numpy array for easier calculation
            eye_coords = np.array(eye_coords)
            
            # IMPROVED EAR calculation with multiple measurements
            # EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
            
            # Multiple vertical distances for better accuracy
            vertical_dist1 = np.linalg.norm(eye_coords[1] - eye_coords[5])  # Top to bottom
            vertical_dist2 = np.linalg.norm(eye_coords[2] - eye_coords[4])  # Top to bottom
            
            # Horizontal distance
            horizontal_dist = np.linalg.norm(eye_coords[0] - eye_coords[3])  # Left to right
            
            # Calculate EAR with better error handling
            if horizontal_dist > 0.5:  # Ensure reasonable horizontal distance
                ear = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)
                
                # Clamp EAR to reasonable range
                ear = max(0.0, min(1.0, ear))
            else:
                ear = self.last_ear  # Keep previous value if calculation seems off
                
            return ear
            
        except Exception as e:
            print(f"Error in single eye EAR calculation: {e}")
            return 0.25
            
    def _compensate_for_head_pose(self, ear, head_pose_data):
        """Compensate EAR for head pose effects - NEW FEATURE"""
        try:
            pitch = head_pose_data.get('pitch', 0)
            yaw = head_pose_data.get('yaw', 0)
            
            # Adjust EAR based on head pose
            # Looking up slightly increases apparent EAR
            if pitch > 10:
                ear *= 1.1  # Slight increase
            elif pitch < -10:
                ear *= 0.95  # Slight decrease
                
            # Side turning can affect EAR measurement
            if abs(yaw) > 20:
                ear *= 1.05  # Slight compensation for side view
                
            return ear
            
        except Exception as e:
            print(f"Error in head pose compensation: {e}")
            return ear
            
    def is_drowsy(self, ear=None, head_turned_away=False):
        """Determine drowsiness with head turn consideration - FIXED"""
        if ear is None:
            ear = self.last_ear
            
        # NEW: Don't detect drowsiness if head is significantly turned away
        # This prevents false drowsiness detection when head movement affects eye landmarks
        if head_turned_away:
            # Reset drowsiness detection when head is turned away
            self.drowsy_frame_count = 0
            self.stable_drowsy_count = 0
            return False
            
        if ear < self.ear_threshold:
            self.drowsy_frame_count += 1
            self.stable_drowsy_count += 1
        else:
            self.drowsy_frame_count = 0
            self.stable_drowsy_count = 0
            
        # Require both consecutive frames AND stability
        return (self.drowsy_frame_count >= self.consecutive_frames and 
                self.stable_drowsy_count >= self.stable_threshold)
        
    def get_drowsiness_level(self, ear=None):
        """Get drowsiness level as a percentage - CALIBRATED"""
        if ear is None:
            ear = self.last_ear
            
        if ear >= 0.28:
            return 0  # Fully awake
        elif ear >= 0.24:
            return 20  # Slightly drowsy
        elif ear >= 0.20:
            return 40  # Moderately drowsy
        elif ear >= 0.16:
            return 60  # Very drowsy
        elif ear >= 0.12:
            return 80  # Extremely drowsy
        else:
            return 100  # Eyes closed
            
    def get_average_ear(self):
        """Get average EAR from recent history"""
        if not self.ear_history:
            return 0.0
        return sum(self.ear_history) / len(self.ear_history)
        
    def draw_eye_contours(self, frame, landmarks):
        """Draw eye contours with enhanced visualization"""
        try:
            # Determine colors based on drowsiness
            drowsiness_level = self.get_drowsiness_level()
            
            if drowsiness_level >= 60:
                color = (0, 0, 255)  # Red for high drowsiness
                thickness = 3
            elif drowsiness_level >= 40:
                color = (0, 165, 255)  # Orange for moderate drowsiness
                thickness = 2
            elif drowsiness_level >= 20:
                color = (0, 255, 255)  # Yellow for slight drowsiness
                thickness = 2
            else:
                color = (0, 255, 0)  # Green for normal
                thickness = 1
                
            # Draw left eye
            left_eye_coords = []
            for idx in self.left_eye_landmarks:
                if idx < len(landmarks):
                    left_eye_coords.append(landmarks[idx])
                    
            if len(left_eye_coords) > 3:
                left_eye_coords = np.array(left_eye_coords, dtype=np.int32)
                cv2.polylines(frame, [left_eye_coords], True, color, thickness)
                
            # Draw right eye
            right_eye_coords = []
            for idx in self.right_eye_landmarks:
                if idx < len(landmarks):
                    right_eye_coords.append(landmarks[idx])
                    
            if len(right_eye_coords) > 3:
                right_eye_coords = np.array(right_eye_coords, dtype=np.int32)
                cv2.polylines(frame, [right_eye_coords], True, color, thickness)
                
            # Draw key points
            for point_idx in self.left_eye_points + self.right_eye_points:
                if point_idx < len(landmarks):
                    cv2.circle(frame, tuple(landmarks[point_idx]), 2, color, -1)
                
        except Exception as e:
            print(f"Error drawing eye contours: {e}")
            
    def reset_state(self):
        """Reset the eye tracker state"""
        self.ear_history.clear()
        self.drowsy_frame_count = 0
        self.stable_drowsy_count = 0
        self.last_ear = 0.3
        
    def get_eye_status(self):
        """Get current eye status with more detail"""
        ear = self.last_ear
        drowsiness = self.get_drowsiness_level()
        
        if drowsiness >= 80:
            return "Eyes Closed"
        elif drowsiness >= 60:
            return "Very Drowsy"
        elif drowsiness >= 40:
            return "Moderately Drowsy"
        elif drowsiness >= 20:
            return "Slightly Drowsy"
        elif ear >= 0.35:
            return "Wide Awake"
        else:
            return "Normal"
            
    def is_reliable_measurement(self, head_pose_data=None):
        """Check if current EAR measurement is reliable given head pose"""
        if head_pose_data is None:
            return True
            
        pitch = abs(head_pose_data.get('pitch', 0))
        yaw = abs(head_pose_data.get('yaw', 0))
        
        # EAR is less reliable when head is significantly turned
        if pitch > 30 or yaw > 35:
            return False
            
        return True
        
    def get_eye_statistics(self):
        """Get statistics about eye tracking"""
        if not self.ear_history:
            return {
                'average_ear': 0.0,
                'current_ear': self.last_ear,
                'drowsiness_level': 0,
                'measurements_count': 0
            }
            
        return {
            'average_ear': self.get_average_ear(),
            'current_ear': self.last_ear,
            'drowsiness_level': self.get_drowsiness_level(),
            'measurements_count': len(self.ear_history),
            'min_ear': min(self.ear_history),
            'max_ear': max(self.ear_history)
        }
import numpy as np
import cv2

class YawnDetector:
    def __init__(self):
        # MediaPipe face mesh landmark indices for mouth
        self.mouth_landmarks = [
            # Outer lip landmarks
            61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318,
            # Inner lip landmarks  
            78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415
        ]
        
        # Key points for MAR calculation - FIXED landmark indices
        self.mouth_points = [
            13, 14,    # Top lip center points
            78, 308,   # Left and right mouth corners  
            18, 175    # Bottom lip center points
        ]
        
        # MAR thresholds and parameters - ADJUSTED for better accuracy
        self.mar_threshold = 0.65  # Reduced from 0.7 for better sensitivity
        self.yawn_consecutive_frames = 4  # Increased from 3 for stability
        self.max_yawn_duration = 45  # Increased from 30 for longer yawns
        
        # State tracking
        self.mar_history = []
        self.yawn_frame_count = 0
        self.is_yawning = False
        self.yawn_start_time = None
        self.last_mar = 0.0
        self.yawn_events = []
        
        # NEW: Stability tracking to prevent false positives
        self.stable_yawn_count = 0
        self.stable_threshold = 3
        
    def calculate_mar(self, landmarks):
        """Calculate Mouth Aspect Ratio (MAR) from facial landmarks - IMPROVED"""
        try:
            # Get mouth coordinates with better error handling
            mouth_coords = []
            for point_idx in self.mouth_points:
                if point_idx < len(landmarks):
                    mouth_coords.append(landmarks[point_idx])
                else:
                    # Return previous MAR if landmarks are missing
                    return self.last_mar
                    
            if len(mouth_coords) < 6:
                return self.last_mar
                
            mouth_coords = np.array(mouth_coords)
            
            # IMPROVED MAR calculation with multiple measurement points
            # MAR = average of multiple vertical distances / horizontal distance
            
            # Multiple vertical distances for better accuracy
            vertical_dist1 = np.linalg.norm(mouth_coords[0] - mouth_coords[4])  # Top center to bottom center
            vertical_dist2 = np.linalg.norm(mouth_coords[1] - mouth_coords[5])  # Another vertical pair
            
            # Horizontal distance (mouth width)
            horizontal_dist = np.linalg.norm(mouth_coords[2] - mouth_coords[3])  # Left to right corner
            
            # Calculate MAR with improved formula
            if horizontal_dist > 1.0:  # Ensure reasonable horizontal distance
                mar = (vertical_dist1 + vertical_dist2) / (2.0 * horizontal_dist)
            else:
                mar = self.last_mar  # Keep previous value if calculation seems off
                
            # SMOOTHING: Apply moving average to reduce noise
            self.mar_history.append(mar)
            if len(self.mar_history) > 10:  # Keep last 10 values for smoothing
                self.mar_history.pop(0)
                
            # Use smoothed MAR
            smoothed_mar = sum(self.mar_history) / len(self.mar_history)
            self.last_mar = smoothed_mar
            return smoothed_mar
            
        except Exception as e:
            print(f"Error calculating MAR: {e}")
            return self.last_mar
            
    def detect_yawn(self, mar=None, head_turned_away=False):
        """Detect yawn with head turn consideration - FIXED LOGIC"""
        if mar is None:
            mar = self.last_mar
            
        # NEW: Don't detect yawn if head is significantly turned away
        # This prevents false yawn detection when head movement affects mouth landmarks
        if head_turned_away:
            # Reset yawn detection when head is turned away
            self.yawn_frame_count = 0
            if self.is_yawning:
                self.is_yawning = False
                print("Yawn detection stopped due to head turn")
            return False
            
        # Check if MAR exceeds threshold
        if mar > self.mar_threshold:
            self.yawn_frame_count += 1
            self.stable_yawn_count += 1
            
            # Confirm yawning after consecutive frames AND stability check
            if (self.yawn_frame_count >= self.yawn_consecutive_frames and 
                self.stable_yawn_count >= self.stable_threshold and 
                not self.is_yawning):
                
                self.is_yawning = True
                self.yawn_start_time = len(self.mar_history)
                self.yawn_events.append({
                    'start_frame': len(self.mar_history),
                    'max_mar': mar,
                    'duration': 0
                })
                print(f"Yawn detected! MAR: {mar:.3f}")
                
        else:
            # End of yawn - IMPROVED logic
            if self.is_yawning and self.yawn_frame_count > 0:
                self.is_yawning = False
                duration = self.yawn_frame_count
                if self.yawn_events:
                    self.yawn_events[-1]['duration'] = duration
                print(f"Yawn ended. Duration: {duration} frames")
                
            # Reset counters
            self.yawn_frame_count = 0
            self.stable_yawn_count = 0
            
        # Prevent extremely long yawn detection (likely false positive)
        if self.yawn_frame_count > self.max_yawn_duration:
            self.is_yawning = False
            self.yawn_frame_count = 0
            self.stable_yawn_count = 0
            print("Long yawn detection reset - likely false positive")
            
        return self.is_yawning
        
    def get_yawn_intensity(self, mar=None):
        """Get yawn intensity as a percentage - CALIBRATED"""
        if mar is None:
            mar = self.last_mar
            
        if mar <= 0.25:
            return 0  # No yawn
        elif mar <= 0.4:
            return 20  # Slight mouth opening
        elif mar <= 0.55:
            return 40  # Moderate opening  
        elif mar <= 0.7:
            return 60  # Strong yawn
        elif mar <= 0.9:
            return 80  # Very strong yawn
        else:
            return 100  # Extreme yawn
            
    def get_average_mar(self):
        """Get average MAR from recent history"""
        if not self.mar_history:
            return 0.0
        return sum(self.mar_history) / len(self.mar_history)
        
    def draw_mouth_contour(self, frame, landmarks):
        """Draw mouth contour on the frame - ENHANCED"""
        try:
            mouth_coords = []
            for idx in self.mouth_landmarks:
                if idx < len(landmarks):
                    mouth_coords.append(landmarks[idx])
                    
            if len(mouth_coords) > 3:
                mouth_coords = np.array(mouth_coords, dtype=np.int32)
                
                # ENHANCED: Color coding based on yawn intensity
                intensity = self.get_yawn_intensity()
                if self.is_yawning:
                    if intensity >= 80:
                        color = (0, 0, 255)  # Red for strong yawn
                    elif intensity >= 60:
                        color = (0, 165, 255)  # Orange for moderate yawn
                    else:
                        color = (0, 255, 255)  # Yellow for mild yawn
                    thickness = 3
                else:
                    color = (255, 0, 0)  # Blue for normal
                    thickness = 1
                    
                cv2.polylines(frame, [mouth_coords], True, color, thickness)
                
                # Draw key points with different colors
                for i, point_idx in enumerate(self.mouth_points):
                    if point_idx < len(landmarks):
                        point_color = (0, 255, 255) if self.is_yawning else (255, 255, 0)
                        cv2.circle(frame, tuple(landmarks[point_idx]), 2, point_color, -1)
                        
        except Exception as e:
            print(f"Error drawing mouth contour: {e}")
            
    def is_mouth_open(self, mar=None):
        """Check if mouth is significantly open - ADJUSTED threshold"""
        if mar is None:
            mar = self.last_mar
        return mar > 0.4  # Reduced from 0.5 for better sensitivity
        
    def get_mouth_status(self):
        """Get current mouth status as a string - ENHANCED"""
        mar = self.last_mar
        
        if self.is_yawning:
            intensity = self.get_yawn_intensity()
            if intensity >= 80:
                return "Strong Yawn"
            elif intensity >= 60:
                return "Moderate Yawn"
            else:
                return "Mild Yawn"
        elif mar > 0.5:
            return "Wide Open"
        elif mar > 0.35:
            return "Open"
        elif mar > 0.25:
            return "Slightly Open"
        else:
            return "Closed"
            
    def reset_state(self):
        """Reset the yawn detector state"""
        self.mar_history.clear()
        self.yawn_frame_count = 0
        self.is_yawning = False
        self.yawn_start_time = None
        self.last_mar = 0.0
        self.yawn_events.clear()
        self.stable_yawn_count = 0
        
    def get_yawn_statistics(self):
        """Get statistics about yawning events"""
        if not self.yawn_events:
            return {
                'total_yawns': 0,
                'average_duration': 0,
                'max_mar': 0,
                'average_intensity': 0
            }
            
        total_yawns = len(self.yawn_events)
        durations = [event['duration'] for event in self.yawn_events if event['duration'] > 0]
        avg_duration = sum(durations) / len(durations) if durations else 0
        max_mar = max([event['max_mar'] for event in self.yawn_events])
        
        # Calculate average intensity
        avg_intensity = sum([self.get_yawn_intensity(event['max_mar']) for event in self.yawn_events]) / total_yawns
        
        return {
            'total_yawns': total_yawns,
            'average_duration': avg_duration,
            'max_mar': max_mar,
            'average_intensity': avg_intensity,
            'recent_yawns': self.yawn_events[-5:]  # Last 5 yawn events
        }
        
    def is_false_positive_likely(self, head_pose_data=None):
        """Check if current yawn detection might be a false positive"""
        if head_pose_data is None:
            return False
            
        # Check if head movement is too extreme for reliable yawn detection
        pitch = abs(head_pose_data.get('pitch', 0))
        yaw = abs(head_pose_data.get('yaw', 0))
        
        # If head is tilted/turned too much, yawn detection may be unreliable
        if pitch > 25 or yaw > 30:
            return True
            
        return False
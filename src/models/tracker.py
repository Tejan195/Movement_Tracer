import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime

class MovementTracker:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.mp_drawing = mp.solutions.drawing_utils
        self.punch_count = 0
        self.skip_count = 0
        self.last_punch_time = datetime.now()
        self.punch_speed = 0
        self.punch_cooldown = 1.0  # Cooldown in seconds between punch detections
        self.punch_in_progress = False
        self.prev_wrist_positions = []
        self.punch_direction_forward = False  # Track punch direction
        self.min_punch_distance = 0.15  # Minimum distance for a punch to be counted
        self.boxing_calories_per_punch = 0.1  # Calories burned per punch
        self.skipping_calories_per_skip = 0.2  # Calories burned per skip
        self.total_boxing_calories = 0
        self.total_skipping_calories = 0
        
    def calculate_calories(self, weight, duration_mins, activity_type):
        # Base MET values (Metabolic Equivalent of Task)
        met_values = {
            'boxing': 7.5,
            'skipping': 12.3
        }
        
        # Calculate base calories from duration
        base_calories = met_values[activity_type] * weight * (duration_mins / 60)
        
        # Add activity-specific calories based on actual movements
        if activity_type == 'boxing':
            # Return combination of time-based and count-based calories
            # Only count 70% of the time-based calories to avoid double counting
            return (base_calories * 0.7) + self.total_boxing_calories
        elif activity_type == 'skipping':
            # Return combination of time-based and count-based calories
            # Only count 70% of the time-based calories to avoid double counting
            return (base_calories * 0.7) + self.total_skipping_calories
        
        return base_calories

    def detect_punch(self, landmarks):
        # Get wrist and shoulder positions
        right_wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
        right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        right_elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        
        # Current time
        current_time = datetime.now()
        time_diff = (current_time - self.last_punch_time).total_seconds()
        
        # Store wrist position for velocity calculation
        self.prev_wrist_positions.append((right_wrist.x, right_wrist.y, current_time))
        if len(self.prev_wrist_positions) > 10:  # Keep only last 10 positions
            self.prev_wrist_positions.pop(0)
        
        # Calculate punch velocity using multiple frames
        if len(self.prev_wrist_positions) >= 5:
            old_pos = self.prev_wrist_positions[0]
            new_pos = self.prev_wrist_positions[-1]
            time_delta = (new_pos[2] - old_pos[2]).total_seconds()
            if time_delta > 0:
                distance_x = new_pos[0] - old_pos[0]
                distance_y = new_pos[1] - old_pos[1]
                distance = np.sqrt(distance_x**2 + distance_y**2)
                self.punch_speed = distance / time_delta
                
                # Check if the movement is forward (away from shoulder)
                # In a mirrored view, a right punch moves left (x decreases)
                wrist_to_shoulder_x = right_wrist.x - right_shoulder.x
                self.punch_direction_forward = (wrist_to_shoulder_x < -0.2)  # Threshold for forward extension
        
        # Check if arm is extended (punch position)
        shoulder_to_elbow = np.sqrt((right_elbow.x - right_shoulder.x)**2 + 
                                  (right_elbow.y - right_shoulder.y)**2)
        elbow_to_wrist = np.sqrt((right_wrist.x - right_elbow.x)**2 + 
                               (right_wrist.y - right_elbow.y)**2)
        shoulder_to_wrist = np.sqrt((right_wrist.x - right_shoulder.x)**2 + 
                                  (right_wrist.y - right_shoulder.y)**2)
        
        # Arm is extended when the distance from shoulder to wrist is close to
        # the sum of shoulder-to-elbow and elbow-to-wrist distances
        arm_extended = shoulder_to_wrist > 0.9 * (shoulder_to_elbow + elbow_to_wrist)
        
        # Calculate distance moved for this potential punch
        punch_distance = 0
        if len(self.prev_wrist_positions) >= 2:
            start_pos = self.prev_wrist_positions[0]
            end_pos = self.prev_wrist_positions[-1]
            punch_distance = np.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
        
        # Detect punch based on multiple criteria
        if time_diff > self.punch_cooldown:  # Ensure cooldown between punches
            # More accurate punch detection with multiple criteria
            if (self.punch_speed > 1.0 and  # Speed threshold
                arm_extended and           # Arm must be extended
                self.punch_direction_forward and  # Must be moving forward
                punch_distance > self.min_punch_distance and  # Must move minimum distance
                not self.punch_in_progress):  # Not already in a punch
                
                self.punch_count += 1
                self.last_punch_time = current_time
                self.punch_in_progress = True
                
                # Add calories for this punch - more calories for faster/stronger punches
                punch_intensity = min(2.0, self.punch_speed) / 2.0  # Normalize to 0-1 range
                calories_for_this_punch = self.boxing_calories_per_punch * (1.0 + punch_intensity)
                self.total_boxing_calories += calories_for_this_punch
                
                return True
        
        # Reset punch_in_progress when arm is retracted
        if not arm_extended and self.punch_in_progress:
            self.punch_in_progress = False
            
        return False

    def detect_skipping(self, landmarks):
        # Get ankle and hip positions
        right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value]
        
        # Calculate ankle height relative to hip
        ankle_hip_ratio = (right_hip.y - right_ankle.y)
        
        # Basic jump detection based on ankle height relative to hip
        # This is more reliable than absolute position
        if ankle_hip_ratio > 0.3:  # Adjusted threshold
            if not hasattr(self, 'skip_in_progress') or not self.skip_in_progress:
                self.skip_count += 1
                self.skip_in_progress = True
                
                # Add calories for this skip
                self.total_skipping_calories += self.skipping_calories_per_skip
                
                return True
        else:
            self.skip_in_progress = False
            
        return False
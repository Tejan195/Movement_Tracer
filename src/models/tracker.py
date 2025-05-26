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
        self.last_punch_time = datetime.now()
        self.punch_speed = 0
        self.punch_cooldown = 0.5  # Cooldown in seconds between punch detections
        self.punch_in_progress = False
        self.prev_wrist_positions = {"left": [], "right": []}
        self.min_punch_distance = 0.15  # Minimum distance for a punch to be counted
        self.boxing_calories_per_punch = 0.1  # Calories burned per punch
        self.total_boxing_calories = 0

    def calculate_calories(self, weight, duration_mins):
        met_value = 7.5  # MET value for boxing
        base_calories = met_value * weight * (duration_mins / 60)
        # Combine time-based and punch-based calories
        return (base_calories * 0.7) + self.total_boxing_calories

    def detect_punch(self, landmarks):
        current_time = datetime.now()
        time_diff = (current_time - self.last_punch_time).total_seconds()
        punch_detected = False
        for side in ["left", "right"]:
            if side == "left":
                wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
                shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
            else:
                wrist = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value]
                shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                elbow = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value]

            # Store wrist position for velocity calculation
            self.prev_wrist_positions[side].append((wrist.x, wrist.y, current_time))
            if len(self.prev_wrist_positions[side]) > 10:
                self.prev_wrist_positions[side].pop(0)

            # Calculate punch velocity using multiple frames
            if len(self.prev_wrist_positions[side]) >= 5:
                old_pos = self.prev_wrist_positions[side][0]
                new_pos = self.prev_wrist_positions[side][-1]
                time_delta = (new_pos[2] - old_pos[2]).total_seconds()
                if time_delta > 0:
                    distance_x = new_pos[0] - old_pos[0]
                    distance_y = new_pos[1] - old_pos[1]
                    distance = np.sqrt(distance_x**2 + distance_y**2)
                    self.punch_speed = distance / time_delta
                    # Check if the movement is forward (away from shoulder)
                    wrist_to_shoulder_x = wrist.x - shoulder.x
                    if side == "right":
                        direction_forward = (wrist_to_shoulder_x < -0.2)
                    else:
                        direction_forward = (wrist_to_shoulder_x > 0.2)
                else:
                    direction_forward = False
            else:
                direction_forward = False

            # Arm extension check
            shoulder_to_elbow = np.sqrt((elbow.x - shoulder.x)**2 + (elbow.y - shoulder.y)**2)
            elbow_to_wrist = np.sqrt((wrist.x - elbow.x)**2 + (wrist.y - elbow.y)**2)
            shoulder_to_wrist = np.sqrt((wrist.x - shoulder.x)**2 + (wrist.y - shoulder.y)**2)
            arm_extended = shoulder_to_wrist > 0.9 * (shoulder_to_elbow + elbow_to_wrist)

            # Calculate distance moved for this potential punch
            if len(self.prev_wrist_positions[side]) >= 2:
                start_pos = self.prev_wrist_positions[side][0]
                end_pos = self.prev_wrist_positions[side][-1]
                punch_distance = np.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
            else:
                punch_distance = 0

            # Detect punch based on multiple criteria
            if time_diff > self.punch_cooldown:
                if (self.punch_speed > 1.0 and arm_extended and direction_forward and punch_distance > self.min_punch_distance and not self.punch_in_progress):
                    self.punch_count += 1
                    self.last_punch_time = current_time
                    self.punch_in_progress = True
                    punch_intensity = min(2.0, self.punch_speed) / 2.0
                    calories_for_this_punch = self.boxing_calories_per_punch * (1.0 + punch_intensity)
                    self.total_boxing_calories += calories_for_this_punch
                    punch_detected = True
            # Reset punch_in_progress when arm is retracted
            if not arm_extended and self.punch_in_progress:
                self.punch_in_progress = False
        return punch_detected
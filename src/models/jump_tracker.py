import numpy as np
from datetime import datetime
import mediapipe as mp

class JumpRopeTracker:
    def __init__(self):
        import mediapipe as mp
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)
        self.jump_count = 0
        self.last_jump_time = datetime.now()
        self.jump_cooldown = 0.4  # seconds
        self.prev_ankle_positions = []
        self.min_jump_height = 0.08  # Tune as needed
        self.jumping_calories_per_jump = 0.07  # Approximate calories per jump
        self.total_jumping_calories = 0

    def calculate_calories(self, weight, duration_mins):
        met_value = 12.3  # MET value for jump rope
        base_calories = met_value * weight * (duration_mins / 60)
        return (base_calories * 0.7) + self.total_jumping_calories

    def detect_jump(self, landmarks):
        current_time = datetime.now()
        left_ankle = landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle = landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        avg_ankle_y = (left_ankle.y + right_ankle.y) / 2
        self.prev_ankle_positions.append((avg_ankle_y, current_time))
        if len(self.prev_ankle_positions) > 10:
            self.prev_ankle_positions.pop(0)
        jump_detected = False
        if len(self.prev_ankle_positions) >= 2:
            prev_y, prev_time = self.prev_ankle_positions[-2]
            y_diff = prev_y - avg_ankle_y
            time_diff = (current_time - self.last_jump_time).total_seconds()
            if y_diff > self.min_jump_height and time_diff > self.jump_cooldown:
                self.jump_count += 1
                self.last_jump_time = current_time
                calories_for_this_jump = self.jumping_calories_per_jump
                self.total_jumping_calories += calories_for_this_jump
                jump_detected = True
        return jump_detected
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
        self.last_punch_time = {"left": datetime.now(), "right": datetime.now()}
        self.punch_speed = {"left": 0, "right": 0}
        self.punch_cooldown = 0.4  # seconds
        self.punch_in_progress = {"left": False, "right": False}
        self.prev_wrist_positions = {"left": [], "right": []}
        self.min_punch_distance = 0.12
        self.boxing_calories_per_punch = 0.1
        self.total_boxing_calories = 0

    def calculate_angle(self, a, b, c):
        a = np.array([a.x, a.y, a.z])
        b = np.array([b.x, b.y, b.z])
        c = np.array([c.x, c.y, c.z])
        ba = a - b
        bc = c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)

    def calculate_calories(self, weight, duration_mins):
        met_value = 7.5  # MET value for boxing
        base_calories = met_value * weight * (duration_mins / 60)
        return (base_calories * 0.7) + self.total_boxing_calories

    def detect_punch(self, landmarks):
        current_time = datetime.now()
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
            self.prev_wrist_positions[side].append((wrist.x, wrist.y, wrist.z, current_time))
            if len(self.prev_wrist_positions[side]) > 10:
                self.prev_wrist_positions[side].pop(0)

            # Calculate velocity and acceleration
            if len(self.prev_wrist_positions[side]) >= 5:
                old_pos = self.prev_wrist_positions[side][0]
                new_pos = self.prev_wrist_positions[side][-1]
                time_delta = (new_pos[3] - old_pos[3]).total_seconds()
                if time_delta > 0:
                    dx = new_pos[0] - old_pos[0]
                    dy = new_pos[1] - old_pos[1]
                    dz = new_pos[2] - old_pos[2]
                    distance = np.sqrt(dx**2 + dy**2 + dz**2)
                    velocity = distance / time_delta
                    self.punch_speed[side] = velocity
                else:
                    velocity = 0
            else:
                velocity = 0

            # Angle analysis
            elbow_angle = self.calculate_angle(shoulder, elbow, wrist)
            arm_extended = elbow_angle > 155  # nearly straight

            # Forward movement (z-axis for depth)
            wrist_to_shoulder_z = wrist.z - shoulder.z
            if side == "right":
                direction_forward = wrist_to_shoulder_z < -0.08
            else:
                direction_forward = wrist_to_shoulder_z < -0.08

            # Distance moved
            if len(self.prev_wrist_positions[side]) >= 2:
                start_pos = self.prev_wrist_positions[side][0]
                end_pos = self.prev_wrist_positions[side][-1]
                punch_distance = np.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2 + (end_pos[2] - start_pos[2])**2)
            else:
                punch_distance = 0

            # Temporal filtering and cooldown
            time_diff = (current_time - self.last_punch_time[side]).total_seconds()
            if time_diff > self.punch_cooldown:
                if (velocity > 1.0 and arm_extended and direction_forward and punch_distance > self.min_punch_distance and not self.punch_in_progress[side]):
                    self.punch_count += 1
                    self.last_punch_time[side] = current_time
                    self.punch_in_progress[side] = True
                    punch_intensity = min(2.0, velocity) / 2.0
                    calories_for_this_punch = self.boxing_calories_per_punch * (1.0 + punch_intensity)
                    self.total_boxing_calories += calories_for_this_punch
                    punch_detected = True
            # Reset punch_in_progress when arm is retracted
            if not arm_extended and self.punch_in_progress[side]:
                self.punch_in_progress[side] = False
        return punch_detected
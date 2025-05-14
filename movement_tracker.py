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
        
    def calculate_calories(self, weight, duration_mins, activity_type):
        # MET values (Metabolic Equivalent of Task)
        met_values = {
            'boxing': 7.5,
            'skipping': 12.3
        }
        
        # Calories = MET × Weight (kg) × Duration (hours)
        calories = met_values[activity_type] * weight * (duration_mins / 60)
        return calories

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
        
        # Detect punch based on speed threshold, cooldown, and arm extension
        if time_diff > self.punch_cooldown:  # Ensure cooldown between punches
            if self.punch_speed > 1.0 and arm_extended and not self.punch_in_progress:  # Higher threshold
                self.punch_count += 1
                self.last_punch_time = current_time
                self.punch_in_progress = True
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
                return True
        else:
            self.skip_in_progress = False
            
        return False

def main():
    # Get user information
    weight = float(input("Enter your weight in kg: "))
    height = float(input("Enter your height in cm: "))
    gender = input("Enter your gender (M/F): ")
    
    tracker = MovementTracker()
    cap = cv2.VideoCapture(0)
    start_time = datetime.now()
    
    # Set camera resolution for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip the frame horizontally for a more intuitive mirror view
        frame = cv2.flip(frame, 1)
            
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = tracker.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            # Draw pose landmarks
            tracker.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, tracker.mp_pose.POSE_CONNECTIONS)
            
            # Detect movements
            landmarks = results.pose_landmarks.landmark
            punch_detected = tracker.detect_punch(landmarks)
            skip_detected = tracker.detect_skipping(landmarks)
            
            # Visual feedback for detected movements
            if punch_detected:
                cv2.putText(frame, "PUNCH!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
            if skip_detected:
                cv2.putText(frame, "JUMP!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
            
        # Calculate duration in minutes
        duration = (datetime.now() - start_time).total_seconds() / 60
        
        # Calculate calories
        boxing_calories = tracker.calculate_calories(weight, duration, 'boxing')
        skipping_calories = tracker.calculate_calories(weight, duration, 'skipping')
        total_calories = boxing_calories + skipping_calories
        
        # Display stats
        stats = f"Punches: {tracker.punch_count} | Speed: {tracker.punch_speed:.2f} | "
        stats += f"Skips: {tracker.skip_count} | Calories: {total_calories:.2f}"
        cv2.putText(frame, stats, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Movement Tracker', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
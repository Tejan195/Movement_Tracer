# Main application entry point
from src.models.jump_tracker import JumpRopeTracker
import cv2
from datetime import datetime

def main():
    weight = float(input("Enter your weight in kg: "))
    height = float(input("Enter your height in cm: "))
    gender = input("Enter your gender (M/F): ")
    
    tracker = JumpRopeTracker()
    cap = cv2.VideoCapture(0)
    start_time = datetime.now()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = tracker.pose.process(rgb_frame)
        if results.pose_landmarks:
            tracker.mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, tracker.mp_pose.POSE_CONNECTIONS)
            landmarks = results.pose_landmarks.landmark
            jump_detected = tracker.detect_jump(landmarks)
            if jump_detected:
                cv2.putText(frame, "JUMP!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        duration = (datetime.now() - start_time).total_seconds() / 60
        jump_calories = tracker.calculate_calories(weight, duration)
        stats = f"Jumps: {tracker.jump_count} | Calories: {jump_calories:.2f}"
        cv2.putText(frame, stats, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Jump Rope Tracker', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
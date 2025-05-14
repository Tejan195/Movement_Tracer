# Main application entry point
from models.tracker import MovementTracker
import cv2
from datetime import datetime

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
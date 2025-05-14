import cv2

# Try to open the camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    print("Success: Camera opened successfully.")
    
    # Try to read a frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
    else:
        print(f"Success: Frame read successfully. Shape: {frame.shape}")
        
        # Try to display the frame
        try:
            cv2.imshow('Camera Test', frame)
            print("Success: Frame displayed successfully.")
            cv2.waitKey(3000)  # Wait for 3 seconds
        except Exception as e:
            print(f"Error displaying frame: {e}")

# Release the camera
cap.release()
cv2.destroyAllWindows()
print("Test completed.")
import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Selfie Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Background color (black in this example)
background_color = (0, 0, 0)  # RGB for black

# Open the webcam
cap = cv2.VideoCapture(0)

print("Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to access webcam.")
        break

    # Optionally flip the frame (uncomment if desired)
    # frame = cv2.flip(frame, 1)

    # Resize and preprocess the frame
    resized_frame = cv2.resize(frame, (640, 480))  # Resize for consistency
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    # Process the frame to get segmentation mask
    results = segmentation.process(rgb_frame)

    # Create a binary mask (lower threshold for better hand detection)
    mask = results.segmentation_mask > 0.3

    # Replace the background with a blurred version of the frame
    blurred_background = cv2.GaussianBlur(resized_frame, (55, 55), 0)
    output_frame = np.where(mask[:, :, None], resized_frame, blurred_background)

    # Optional: Flip the output for a mirrored effect
    output_frame = cv2.flip(output_frame, 1)

    # Display the output frame
    cv2.imshow("Enhanced Background Removal", output_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

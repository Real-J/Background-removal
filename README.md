# Background Removal using MediaPipe and OpenCV

This project demonstrates real-time background removal and replacement using the MediaPipe Selfie Segmentation model and OpenCV. The program captures video from your webcam, segments the foreground (e.g., a person), and replaces the background with a blurred version of the original frame.

## Features
- **Real-Time Background Removal:** Uses MediaPipe's Selfie Segmentation model to differentiate the foreground from the background.
- **Background Blur:** Replaces the background with a Gaussian-blurred version of the original frame.
- **Customizable Thresholds:** Allows fine-tuning of segmentation quality.
- **Mirrored Output:** Optionally flips the output frame horizontally for a mirrored effect.

## Prerequisites
Ensure you have the following installed:
- Python 3.6 or higher
- OpenCV
- MediaPipe
- NumPy

You can install the required Python libraries using pip:
```bash
pip install opencv-python mediapipe numpy
```

## How to Run
1. Save the code into a file, for example, `background_removal.py`.
2. Run the script using Python:
   ```bash
   python backgroundremoval.py
   ```
3. The webcam feed will open, showing the real-time background replacement effect.
4. Press the `q` key to exit the application.

## Code Explanation

### Initialization
- **MediaPipe Selfie Segmentation:**
  ```python
  mp_selfie_segmentation = mp.solutions.selfie_segmentation
  segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
  ```
  Initializes the MediaPipe Selfie Segmentation model with `model_selection=1`, which is optimized for general use.

- **Background Color:**
  ```python
  background_color = (0, 0, 0)
  ```
  Specifies the background color (not used directly since we apply a blurred background instead).

### Webcam Capture
- **Open the webcam:**
  ```python
  cap = cv2.VideoCapture(0)
  ```
  Accesses the default webcam.

- **Resize and preprocess the frame:**
  ```python
  resized_frame = cv2.resize(frame, (640, 480))
  rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
  ```
  Resizes the frame to a consistent size and converts it to RGB for processing.

### Segmentation and Background Replacement
- **Generate Segmentation Mask:**
  ```python
  results = segmentation.process(rgb_frame)
  mask = results.segmentation_mask > 0.3
  ```
  Processes the frame to generate a segmentation mask, with a threshold of 0.3 to filter the background.

- **Apply Background Blur:**
  ```python
  blurred_background = cv2.GaussianBlur(resized_frame, (55, 55), 0)
  output_frame = np.where(mask[:, :, None], resized_frame, blurred_background)
  ```
  Blurs the original frame and overlays the segmented foreground on the blurred background.

### Display and Controls
- **Display the Output:**
  ```python
  cv2.imshow("Enhanced Background Removal", output_frame)
  ```
  Shows the processed video feed in a window.

- **Exit the Application:**
  ```python
  if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  ```
  Allows the user to exit by pressing the `q` key.

### Cleanup
- **Release Resources:**
  ```python
  cap.release()
  cv2.destroyAllWindows()
  ```
  Closes the webcam and destroys all OpenCV windows.

## Customization
- **Adjust Segmentation Quality:**
  Modify the threshold value in `mask = results.segmentation_mask > 0.3` for finer control over background detection.

- **Change Background Effect:**
  Replace the `blurred_background` with any image or effect to customize the background.
  ```python
  custom_background = cv2.imread('background.jpg')
  output_frame = np.where(mask[:, :, None], resized_frame, custom_background)
  ```

## Troubleshooting
- **Webcam Not Detected:** Ensure the webcam is properly connected and accessible.
- **Performance Issues:** Reduce the frame size or use a smaller Gaussian kernel for faster processing.

## License
This project is open-source and available for educational and personal use.




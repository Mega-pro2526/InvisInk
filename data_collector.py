import cv2
import numpy as np
import os
import mediapipe as mp

# --- Constants and Configuration ---
# Create directories to store the dataset
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Define the classes (symbols) to be collected
CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '*', '/', '(', ')']
NUM_CLASSES = len(CLASSES)
SAMPLES_PER_CLASS = 50 # Aim for at least this many samples

# On Windows, some characters are invalid in directory names (e.g., '*' and '/')
# Map display labels to filesystem-safe directory names
SAFE_NAME_MAP = {
    '*': 'STAR',
    '/': 'SLASH',
}

def get_safe_dirname(label: str) -> str:
    return SAFE_NAME_MAP.get(label, label)

# Create subdirectories for each class
for cls in CLASSES:
    safe_cls = get_safe_dirname(cls)
    if not os.path.exists(os.path.join(DATA_DIR, safe_cls)):
        os.makedirs(os.path.join(DATA_DIR, safe_cls))

# MediaPipe Hand Tracking Setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
# Relax thresholds and enable tracking to improve robustness
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# --- Main Data Collection Loop ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Try to use an HD resolution for more reliable landmark detection
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

current_class_index = 0
sample_count = 0
drawing = False
points = []

print("--- InvisInk Data Collector ---")
print("Instructions:")
print("1. A window will show your camera feed.")
print("2. Hold your hand up; your index fingertip will be tracked.")
print(f"3. We will collect data for class: '{CLASSES[current_class_index]}'")
print("4. Press 'd' to START/STOP drawing a symbol with your fingertip.")
print("5. When you are done drawing a symbol, press 's' to save it.")
print("6. The canvas will clear, ready for the next sample.")
print("7. After collecting enough samples, the script will move to the next class.")
print("8. Press 'q' to quit at any time.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1) # Flip for a more natural, mirror-like view
    
    # Create a separate canvas for drawing
    canvas = np.zeros_like(frame)

    # Process the frame to find hand landmarks
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Display instructions and status on the frame
    status_text = f"Collecting for Class: '{CLASSES[current_class_index]}' ({sample_count}/{SAMPLES_PER_CLASS})"
    cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, "Press 'd' to draw, 's' to save, 'q' to quit", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if drawing:
        cv2.putText(frame, "MODE: DRAWING", (500, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if results.multi_hand_landmarks:
        # Get landmarks for the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Get the coordinate of the index fingertip (landmark 8)
        index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        h, w, _ = frame.shape
        x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
        
        # Circle the fingertip for visual feedback
        cv2.circle(frame, (x, y), 10, (255, 0, 0), -1)

        if drawing:
            points.append((x, y))

    # Draw the captured points on the canvas
    if len(points) > 1:
        for i in range(1, len(points)):
            if points[i - 1] is None or points[i] is None:
                continue
            # Check distance to avoid drawing lines when finger moves far away quickly
            if np.linalg.norm(np.array(points[i-1]) - np.array(points[i])) < 50:
                 cv2.line(canvas, points[i - 1], points[i], (255, 255, 255), 10)

    # Combine the camera feed and the drawing canvas
    # If no hand detected, show a hint
    if not results.multi_hand_landmarks:
        cv2.putText(frame, "No hand detected - show your palm, good lighting", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    combined_display = cv2.addWeighted(frame, 0.8, canvas, 1, 0)
    cv2.imshow('InvisInk Data Collector', combined_display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('d'):
        drawing = not drawing
        if not drawing: # If we stopped drawing
            points.append(None) # Add a separator for discontinuous lines
    elif key == ord('s'):
        if len(points) > 10: # Only save if something was drawn
            try:
                # --- Image Processing ---
                # Create a blank image to draw the symbol on
                img_to_save = np.zeros((400, 400), dtype=np.uint8)
                
                # Filter out None points for bounding box calculation
                valid_points = [p for p in points if p is not None]
                if not valid_points: continue

                # Get bounding box of the drawing
                x_coords = [p[0] for p in valid_points]
                y_coords = [p[1] for p in valid_points]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                # Extract the drawing from the canvas
                drawn_symbol_roi = canvas[y_min:y_max, x_min:x_max]
                gray_roi = cv2.cvtColor(drawn_symbol_roi, cv2.COLOR_BGR2GRAY)

                # Resize to 28x28 (standard for digit recognition)
                resized_symbol = cv2.resize(gray_roi, (28, 28))

                # --- Save the Image ---
                class_label = CLASSES[current_class_index]
                class_path = os.path.join(DATA_DIR, get_safe_dirname(class_label))
                save_path = os.path.join(class_path, f"{sample_count}.png")
                cv2.imwrite(save_path, resized_symbol)
                print(f"Saved: {save_path}")

                sample_count += 1
                points.clear() # Clear points for the next sample
                
                # Check if we have collected enough samples for the current class
                if sample_count >= SAMPLES_PER_CLASS:
                    sample_count = 0
                    current_class_index += 1
                    if current_class_index >= NUM_CLASSES:
                        print("All classes collected! You can now run model_trainer.py")
                        break
                    else:
                        print(f"--- Moving to next class: '{CLASSES[current_class_index]}' ---")

            except Exception as e:
                print(f"Error saving image: {e}")
                points.clear()


cap.release()
cv2.destroyAllWindows()

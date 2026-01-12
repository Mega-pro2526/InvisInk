import cv2 
import numpy as np
import mediapipe as mp
import tensorflow as tf
import time

# --- Configuration and Initialization ---
MODEL_PATH = 'invisink_model.h5'
# Define the class mapping manually to match the trainer
CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '*', '/', '(', ')']

# Load the trained model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure 'invisink_model.h5' is in the same directory.")
    exit()

# MediaPipe Hands Setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# --- Gesture Recognition Logic ---
def get_gesture(hand_landmarks):
    """
    Recognizes gestures based on finger positions.
    Returns: A string representing the gesture ('FINGERTIP', 'FIST', 'THUMBS_UP', 'OPEN_HAND', 'UNKNOWN').
    """
    # Landmark indices for fingertips
    tip_ids = [4, 8, 12, 16, 20]
    
    # Check for OPEN HAND (all fingers extended)
    # A finger is extended if its tip is above its PIP joint (2 landmarks below the tip)
    fingers_extended = []
    # Thumb (special case: check x-position relative to its base)
    if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x:
         fingers_extended.append(1)
    else:
         fingers_extended.append(0)

    # Other four fingers
    for i in range(1, 5):
        if hand_landmarks.landmark[tip_ids[i]].y < hand_landmarks.landmark[tip_ids[i] - 2].y:
            fingers_extended.append(1)
        else:
            fingers_extended.append(0)
            
    total_fingers = sum(fingers_extended)
    
    # FINGERTIP: Only index finger is extended
    if total_fingers == 1 and fingers_extended[1] == 1:
        return 'FINGERTIP'
    
    # THUMBS UP: Only thumb is extended
    if total_fingers == 1 and fingers_extended[0] == 1:
        return 'THUMBS_UP'
        
    # FIST: No fingers are extended
    if total_fingers == 0:
        return 'FIST'
        
    # OPEN HAND: All five fingers are extended
    if total_fingers == 5:
        return 'OPEN_HAND'
        
    return 'UNKNOWN'


# --- Main Application Loop ---
def find_working_camera_index(max_index=5):
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                return i
        else:
            cap.release()
    return None

cam_index = find_working_camera_index()
if cam_index is None:
    print("Error: Could not open any camera (tried indices 0-4).")
    exit()
print(f"Using camera index {cam_index}")
cap = cv2.VideoCapture(cam_index)
if not cap.isOpened():
    print(f"Error: Could not open camera index {cam_index}.")
    exit()

# State variables
drawing_points = []
current_expression = ""
last_gesture = "UNKNOWN"
last_fist_time = 0
debounce_time = 1.5 # 1.5 seconds debounce for fist gesture

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # Create a canvas to draw on
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # Process frame with MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    gesture = "UNKNOWN"
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        gesture = get_gesture(hand_landmarks)
        
        # Get fingertip coordinates for drawing
        index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
        
        # --- Gesture-based Control Logic ---
        if gesture == 'FINGERTIP':
            cv2.circle(frame, (x,y), 10, (0, 255, 0), -1) # Green circle when drawing
            drawing_points.append((x,y))
        
        # FIST: Recognize the drawn symbol
        elif gesture == 'FIST':
            # Use debounce to prevent multiple recognitions
            current_time = time.time()
            if len(drawing_points) > 10 and (current_time - last_fist_time > debounce_time):
                last_fist_time = current_time
                
                # --- Preprocess and Predict ---
                # Get bounding box
                x_coords = [p[0] for p in drawing_points]
                y_coords = [p[1] for p in drawing_points]
                x_min, x_max = max(0, min(x_coords) - 20), min(w, max(x_coords) + 20)
                y_min, y_max = max(0, min(y_coords) - 20), min(h, max(y_coords) + 20)
                
                # Create a temporary canvas for the isolated symbol
                symbol_canvas = np.zeros((h, w), dtype=np.uint8)
                for i in range(1, len(drawing_points)):
                    cv2.line(symbol_canvas, drawing_points[i - 1], drawing_points[i], (255, 255, 255), 15)
                
                # Extract ROI and predict
                symbol_roi = symbol_canvas[y_min:y_max, x_min:x_max]
                if symbol_roi.size > 0:
                    resized_symbol = cv2.resize(symbol_roi, (30, 30))
                    processed_symbol = resized_symbol.reshape(1, 30, 30, 1) / 255.0
                    
                    prediction = model.predict(processed_symbol)
                    predicted_class = CLASSES[np.argmax(prediction)]
                    
                    current_expression += predicted_class
                    
                drawing_points.clear()

        # THUMBS UP: Solve the equation
        elif gesture == 'THUMBS_UP' and last_gesture != 'THUMBS_UP':
             if current_expression:
                try:
                    # Replace visual multiplication/division symbols if needed
                    # Note: eval() is used for simplicity. For a production app, a safer math parser is recommended.
                    result = eval(current_expression)
                    current_expression += f" = {result}"
                except Exception as e:
                    current_expression = "Error"

        # OPEN HAND: Clear everything
        elif gesture == 'OPEN_HAND':
             current_expression = ""
             drawing_points.clear()

    last_gesture = gesture

    # --- UI and Display ---
    # Draw the current path
    if len(drawing_points) > 1:
        for i in range(1, len(drawing_points)):
            cv2.line(canvas, drawing_points[i-1], drawing_points[i], (255, 255, 0), 8)

    # Combine frame and canvas
    frame = cv2.add(frame, canvas)
    
    # Display UI Panel
    cv2.rectangle(frame, (10, 10), (w - 10, 90), (50, 50, 50), -1)
    cv2.putText(frame, f"Gesture: {gesture}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f"Expression: {current_expression}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow('InvisInk - Air Drawing Calculator', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

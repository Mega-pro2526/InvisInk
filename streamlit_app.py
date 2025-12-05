import time
from typing import Deque, List, Optional, Tuple

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

import mediapipe as mp
import tensorflow as tf


# --- Configuration ---
MODEL_PATH = 'invisink_model.h5'
CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '*', '/', '(', ')']


@st.cache_resource(show_spinner=False)
def load_model() -> tf.keras.Model:
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as exc:
        st.error(f"Error loading model from {MODEL_PATH}: {exc}")
        st.stop()


class InvisInkProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        # MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # State
        self.drawing_points: List[Tuple[int, int]] = []
        self.current_expression: str = ""
        self.last_gesture: str = "UNKNOWN"
        self.last_fist_time: float = 0.0
        self.debounce_time: float = 1.5

        # Model
        self.model = load_model()

    def get_gesture(self, hand_landmarks) -> str:
        tip_ids = [4, 8, 12, 16, 20]

        fingers_extended: List[int] = []
        # Thumb: x-position vs base
        if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x:
            fingers_extended.append(1)
        else:
            fingers_extended.append(0)

        # Other four fingers: tip higher than PIP
        for i in range(1, 5):
            if hand_landmarks.landmark[tip_ids[i]].y < hand_landmarks.landmark[tip_ids[i] - 2].y:
                fingers_extended.append(1)
            else:
                fingers_extended.append(0)

        total_fingers = sum(fingers_extended)

        if total_fingers == 1 and fingers_extended[1] == 1:
            return 'FINGERTIP'
        if total_fingers == 1 and fingers_extended[0] == 1:
            return 'THUMBS_UP'
        if total_fingers == 0:
            return 'FIST'
        if total_fingers == 5:
            return 'OPEN_HAND'
        return 'UNKNOWN'

    def _predict_symbol(self, symbol_roi: np.ndarray) -> Optional[str]:
        if symbol_roi.size == 0:
            return None
        resized_symbol = cv2.resize(symbol_roi, (30, 30))
        processed_symbol = resized_symbol.reshape(1, 30, 30, 1) / 255.0
        try:
            prediction = self.model.predict(processed_symbol, verbose=0)
            predicted_class = CLASSES[int(np.argmax(prediction))]
            return predicted_class
        except Exception:
            return None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        # Canvas for drawing path
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

        # MediaPipe processing
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        gesture = "UNKNOWN"
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            self.mp_drawing.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

            gesture = self.get_gesture(hand_landmarks)

            index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x, y = int(index_tip.x * w), int(index_tip.y * h)

            if gesture == 'FINGERTIP':
                cv2.circle(img, (x, y), 10, (0, 255, 0), -1)
                self.drawing_points.append((x, y))

            elif gesture == 'FIST':
                current_time = time.time()
                if len(self.drawing_points) > 10 and (current_time - self.last_fist_time > self.debounce_time):
                    self.last_fist_time = current_time

                    x_coords = [p[0] for p in self.drawing_points]
                    y_coords = [p[1] for p in self.drawing_points]
                    x_min, x_max = max(0, min(x_coords) - 20), min(w, max(x_coords) + 20)
                    y_min, y_max = max(0, min(y_coords) - 20), min(h, max(y_coords) + 20)

                    symbol_canvas = np.zeros((h, w), dtype=np.uint8)
                    for i in range(1, len(self.drawing_points)):
                        cv2.line(symbol_canvas, self.drawing_points[i - 1], self.drawing_points[i], (255, 255, 255), 15)

                    symbol_roi = symbol_canvas[y_min:y_max, x_min:x_max]
                    predicted = self._predict_symbol(symbol_roi)
                    if predicted is not None:
                        self.current_expression += predicted
                    self.drawing_points.clear()

            elif gesture == 'THUMBS_UP' and self.last_gesture != 'THUMBS_UP':
                if self.current_expression:
                    try:
                        result = eval(self.current_expression)
                        self.current_expression += f" = {result}"
                    except Exception:
                        self.current_expression = "Error"

            elif gesture == 'OPEN_HAND':
                self.current_expression = ""
                self.drawing_points.clear()

        self.last_gesture = gesture

        # Draw path on canvas
        if len(self.drawing_points) > 1:
            for i in range(1, len(self.drawing_points)):
                cv2.line(canvas, self.drawing_points[i - 1], self.drawing_points[i], (255, 255, 0), 8)

        img = cv2.add(img, canvas)

        # UI overlay
        cv2.rectangle(img, (10, 10), (w - 10, 90), (50, 50, 50), -1)
        cv2.putText(img, f"Gesture: {gesture}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img, f"Expression: {self.current_expression}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


st.set_page_config(page_title="InvisInk - Streamlit", layout="wide")
st.title("InvisInk - Air Drawing Calculator (Streamlit)")
st.markdown("Use your index finger to draw. Make a fist to recognize, thumbs up to evaluate, open hand to clear. Press Stop to end.")

webrtc_streamer(
    key="invisink",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=InvisInkProcessor,
    media_stream_constraints={"video": True, "audio": False},
)






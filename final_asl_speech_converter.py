import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque
import pyttsx3
# Load the trained model (make sure to have this file in the same directory or provide correct path)
MODEL_PATH = "asl_landmark_cnn.h5"
model = load_model(MODEL_PATH)
# Label map (MUST match training order)
labels = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T',
    'U','V','W','X','Y','Z','del','nothing','space'
]
#text to speech setup
engine = pyttsx3.init()
engine.setProperty("rate", 160)
engine.setProperty("volume", 1.0)
def speak_text(text):
    if text.strip() != "":
        engine.say(text)
        engine.runAndWait()
#mediapipe hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils
#normalization function to make the model invariant to hand position and size
def normalize_landmarks(landmarks):
    pts = np.array(landmarks).reshape(21, 3)
    wrist = pts[0]
    pts = pts - wrist
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale > 0:
        pts = pts / scale
    return pts.flatten()
#prediction queue for stable predictions
pred_queue = deque(maxlen=15)
confidence_threshold = 0.85
final_text = ""
#camera setup
cap = cv2.VideoCapture(0)
print("ESC = Exit | R = Speak text | C = Clear text")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    current_pred = None
    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        data = normalize_landmarks(landmarks)
        data = np.expand_dims(data, axis=0)
        probs = model.predict(data, verbose=0)[0]
        pred_index = np.argmax(probs)
        confidence = probs[pred_index]
        if confidence > confidence_threshold:
            current_pred = labels[pred_index]
            pred_queue.append(current_pred)
    #process the prediction queue to get stable predictions
    if len(pred_queue) == pred_queue.maxlen:
        most_common = max(set(pred_queue), key=pred_queue.count)
        if most_common == "space":
            final_text += " "
            pred_queue.clear()
        elif most_common == "del":
            final_text = final_text[:-1]
            pred_queue.clear()
        elif most_common != "nothing":
            final_text += most_common
            pred_queue.clear()
    #display the final text and current prediction on the frame
    cv2.rectangle(frame, (0, 0), (640, 110), (0, 0, 0), -1)
    cv2.putText(frame, f"Text: {final_text}", (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if current_pred:
        cv2.putText(frame, f"Pred: {current_pred}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(frame, "R: Speak  |  C: Clear  |  ESC: Exit",
                (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 255), 2)
    cv2.imshow("Final ASL Speech Converter", frame)
    #control keys
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('r') or key == ord('R'):
        speak_text(final_text)
    elif key == ord('c') or key == ord('C'):
        final_text = ""
cap.release()
cv2.destroyAllWindows()
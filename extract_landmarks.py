import cv2
import mediapipe as mp
import numpy as np
import os
RAW_DIR = r"Z:\PROJECT\data\raw"
OUT_DIR = r"Z:\PROJECT\data\landmarks"
os.makedirs(OUT_DIR, exist_ok=True)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)
total_saved = 0
total_failed = 0
for label in sorted(os.listdir(RAW_DIR)):
    label_path = os.path.join(RAW_DIR, label)
    if not os.path.isdir(label_path):
        continue
    save_label_dir = os.path.join(OUT_DIR, label)
    os.makedirs(save_label_dir, exist_ok=True)
    print(f"Processing label: {label}")
    for img_name in os.listdir(label_path):
        img_path = os.path.join(label_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            np.save(
                os.path.join(
                    save_label_dir,
                    os.path.splitext(img_name)[0] + ".npy"
                ),
                np.array(landmarks)
            )
            total_saved += 1
        else:
            total_failed += 1
print("Total saved:", total_saved)
print("Total failed:", total_failed)

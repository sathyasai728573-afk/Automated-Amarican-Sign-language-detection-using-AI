import os
import numpy as np
SRC_DIR = r"Z:\PROJECT\data\landmarks"
DST_DIR = r"Z:\PROJECT\data\landmarks_norm"
os.makedirs(DST_DIR, exist_ok=True)
def normalize_landmarks(landmarks):
    """
    landmarks: shape (63,)
    returns: normalized landmarks (63,)
    """
    pts = landmarks.reshape(21, 3)
    #1.Use wrist (landmark 0) as origin
    wrist = pts[0]
    pts = pts - wrist

    #2.Scale by hand size (max distance from wrist)
    scale = np.max(np.linalg.norm(pts, axis=1))
    if scale > 0:
        pts = pts / scale
    return pts.flatten()
total_files = 0
for label in sorted(os.listdir(SRC_DIR)):
    src_label_dir = os.path.join(SRC_DIR, label)
    if not os.path.isdir(src_label_dir):
        continue
    dst_label_dir = os.path.join(DST_DIR, label)
    os.makedirs(dst_label_dir, exist_ok=True)
    print(f"Normalizing label: {label}")
    for file in os.listdir(src_label_dir):
        src_path = os.path.join(src_label_dir, file)
        data = np.load(src_path)
        norm_data = normalize_landmarks(data)
        dst_path = os.path.join(dst_label_dir, file)
        np.save(dst_path, norm_data)
        total_files += 1
print("Total normalized samples:", total_files)
import os
import numpy as np
#CHANGE HERE: checking normalized landmarks for S
LABEL = "S"
BASE_DIR = r"Z:\PROJECT\data\landmarks_norm"
LABEL_DIR = os.path.join(BASE_DIR, LABEL)
files = os.listdir(LABEL_DIR)
print(f"Checking label: {LABEL}")
print("Total samples:", len(files))
if len(files) == 0:
    print("No normalized landmark files found")
    exit()
sample_file = files[0]
sample_path = os.path.join(LABEL_DIR, sample_file)
data = np.load(sample_path)
print("Sample file:", sample_file)
print("Shape:", data.shape)
print("First 10 values:", data[:10])
print("Min value:", data.min())
print("Max value:", data.max())
# Extra geometry sanity check
pts = data.reshape(21, 3)
print("Wrist (should be near 0,0,0):", pts[0])

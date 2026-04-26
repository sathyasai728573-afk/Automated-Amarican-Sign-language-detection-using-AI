import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
DATA_DIR = r"Z:\PROJECT\data\landmarks_norm" #path to the directory where the preprocessed landmark data is stored as .npy files in subfolders named after the labels (like "A", "B", etc)
# Load labels
labels = sorted(os.listdir(DATA_DIR))
label_map = {label: i for i, label in enumerate(labels)}
X = []
y = []
for label in labels:
    label_dir = os.path.join(DATA_DIR, label)
    for file in os.listdir(label_dir):
        data = np.load(os.path.join(label_dir, file))
        X.append(data)
        y.append(label_map[label])
X = np.array(X)
y = np.array(y)
print("Dataset shape:", X.shape, y.shape)
print("Label map:", label_map)
# Train / validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
# CNN model (landmark-based)
model = Sequential([
    Dense(256, activation="relu", input_shape=(63,)),
    Dropout(0.4),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dense(len(labels), activation="softmax")
])
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()
#TRAIN WITH BATCH SIZE = 32
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32
)
# Save trained model
model.save("asl_landmark_cnn.h5")
print("Model saved as asl_landmark_cnn.h5")

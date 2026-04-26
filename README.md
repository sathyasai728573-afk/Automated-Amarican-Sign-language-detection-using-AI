# Automated-Amarican-Sign-language-detection-using-AI
all files exist in Z:\PROJECT
to activate environment DO-->

for CMD:>
win + R
type -->Z:
type -->cd Z:\PROJECT
type -->MCEV\Scripts\activate

for terminal:>
open terminal
type -->Z:
type -->cd Z:\PROJECT
type -->.\MCEV\Scripts\Activate.ps1

ENVIRONMENT REQUIREMENTS:

used for creation of landmarks:
Python == 3.10.x
numpy==2.2.6  #numpy==2.2.6
opencv-python==4.13.0.90  #Computer Vision
mediapipe==0.10.9
sounddevice==0.5.3  #Audio dependency required by MediaPipe
#tensorflow should not be downloaded at this time
#it may make errors with mediaipe

For CNN:
Python == 3.10.x
numpy==2.2.6
opencv-python==4.13.0.90
mediapipe==0.10.9
sounddevice==0.5.3
tensorflow==2.13.0


# Core
import os
import sys
import time
import math
import random

# Numerical & Data
import numpy as np
import pandas as pd

# Computer Vision
import cv2

# MediaPipe
import mediapipe as mp

# Deep Learning (TensorFlow / Keras)
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Conv2D,
    MaxPooling2D,
    Flatten,
    BatchNormalization,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical

# Machine Learning utilities
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Visualization
import matplotlib.pyplot as plt

# Progress bar
from tqdm import tqdm

environment shud have:
numpy==1.26.4
opencv-python==4.9.0.80
mediapipe==0.10.9
tensorflow==2.15.0
protobuf==3.20.3
matplotlib==3.8.3
scikit-learn==1.4.1.post1
pandas==2.2.1
tqdm==4.66.2

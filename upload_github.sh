"""
Audio Classification with Convolutional Neural Networks (CNN)
Using MFCC Features and ESC-10 Dataset

Author: Elizangela de Macedo Brito
Year: 2026
License: MIT

Description:
This script performs environmental sound classification using a
Convolutional Neural Network (CNN). The ESC-10 dataset is downloaded
automatically, MFCC features are extracted from audio signals, and
a CNN model is trained and evaluated.

The code is written as a single Python file for simplicity and clarity,
while maintaining good practices such as reproducibility, error handling,
and technical documentation.
"""

# ==============================
# 1. Imports
# ==============================
import os
import random
import numpy as np
import librosa
import tensorflow as tf
import kagglehub

from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ==============================
# 2. Reproducibility Configuration
# ==============================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# ==============================
# 3. Dataset and Signal Parameters
# ==============================
# ESC-10 class names
CLASSES = [
    "chainsaw", "crackling_fire", "dog", "rain", "sea_waves",
    "clock_tick", "crying_baby", "helicopter", "rooster", "sneezing"
]

# Audio processing parameters
FS = 44100                 # Sampling rate (Hz)
N_MFCC = 40                # Number of MFCC coefficients
HOP_LENGTH = 512           # Hop length for STFT
MAX_LEN = 431              # Fixed number of frames (~5 seconds)

# ==============================
# 4. Feature Extraction Function
# ==============================
def extract_mfcc(file_path):
    """
    Load an audio file and extract MFCC features with fixed length.
    Padding or truncation is applied to ensure consistent input size
    for the CNN.
    """
    try:
        signal, sr = librosa.load(file_path, sr=FS)
        mfcc = librosa.feature.mfcc(
            y=signal,
            sr=sr,
            n_mfcc=N_MFCC,
            hop_length=HOP_LENGTH
        )

        # Pad or truncate MFCCs to fixed length
        if mfcc.shape[1] < MAX_LEN:
            pad_width = MAX_LEN - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)))
        else:
            mfcc = mfcc[:, :MAX_LEN]

        return mfcc

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

# ==============================
# 5. Dataset Loading and Preparation
# ==============================
print("Downloading ESC-10 dataset...")
dataset_path = kagglehub.dataset_download("sreyared15/esc10enlarged")

X = []
y = []

print("Extracting features...")
for class_name in CLASSES:
    class_dir = os.path.join(dataset_path, class_name)

    if not os.path.isdir(class_dir):
        continue

    for file_name in os.listdir(class_dir):
        file_path = os.path.join(class_dir, file_name)
        mfcc = extract_mfcc(file_path)

        if mfcc is not None:
            X.append(mfcc)
            y.append(class_name)

X = np.array(X)
y = np.array(y)

# Encode class labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Add channel dimension for CNN input
X = X[..., np.newaxis]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=SEED, stratify=y_encoded
)

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")

# ==============================
# 6. CNN Model Definition
# ==============================
input_shape = (N_MFCC, MAX_LEN, 1)
num_classes = len(CLASSES)

model = models.Sequential([
    layers.Input(shape=input_shape),

    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),

    layers.Dense(num_classes, activation="softmax")
])

# ==============================
# 7. Model Compilation
# ==============================
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==============================
# 8. Model Training
# ==============================
print("Training model...")
history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# ==============================
# 9. Model Evaluation
# ==============================
print("Evaluating model...")
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# ==============================
# 10. Final Notes
# ==============================
"""
This implementation represents a baseline approach for environmental
sound classification using MFCC features and CNNs.

Due to the small size of ESC-10, results may vary and overfitting can occur.
For more advanced performance, consider:
- Data augmentation techniques
- Log-Mel spectrograms
- Transfer learning with pretrained audio models
"""

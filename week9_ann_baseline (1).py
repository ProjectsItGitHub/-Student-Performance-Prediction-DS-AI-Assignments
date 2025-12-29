# ======================================================
# WEEK 9 — NEURAL NETWORK BASICS (ANN)
# Project: Image-Based Waste Processing
# ======================================================

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ----------------------------------
# Dummy feature data (ANN baseline)
# ----------------------------------
# In real project, ANN is used as a baseline
# Image features are flattened for ANN

X = np.random.rand(600, 100)
y = np.random.choice(
    ["plastic", "glass", "metal", "paper", "cardboard", "trash"], 600
)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_cat = to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42
)

# ----------------------------------
# ANN Model
# ----------------------------------
model = Sequential()
model.add(Dense(128, activation="relu", input_shape=(100,)))
model.add(Dropout(0.3))
model.add(Dense(64, activation="relu"))
model.add(Dense(6, activation="softmax"))

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

loss, acc = model.evaluate(X_test, y_test)
print("ANN Accuracy:", acc)

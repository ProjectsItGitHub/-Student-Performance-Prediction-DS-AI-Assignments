# ======================================================
# WEEK 10 — ADVANCED DEEP LEARNING (CNN)
# Transfer Learning using MobileNetV2
# Project: Image-Based Waste Processing
# ======================================================

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

# ----------------------------------
# Base Model: MobileNetV2
# ----------------------------------
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model layers
base_model.trainable = False

# ----------------------------------
# Custom Classification Head
# ----------------------------------
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(256, activation="relu")(x)
outputs = Dense(6, activation="softmax")(x)

# Final Model
model = Model(inputs=base_model.input, outputs=outputs)

# ----------------------------------
# Compile Model
# ----------------------------------
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Model Summary
model.summary()

# ======================================================
# STUDENT PERFORMANCE PREDICTION USING ANN
# Course Project (Deep Learning Basics)
# ======================================================

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ----------------------------------
# Step 1: Dataset Creation (Dummy)
# ----------------------------------
# Features:
# study_hours, attendance, previous_score, assignments, sleep_hours

X = np.random.rand(500, 5)

performance = np.random.choice(
    ["Fail", "Pass", "Good", "Excellent"], 500
)

# Encode output labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(performance)
y_cat = to_categorical(y_encoded)

# Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, random_state=42
)

# ----------------------------------
# Step 2: ANN Model
# ----------------------------------
model = Sequential()
model.add(Dense(64, activation="relu", input_shape=(5,)))
model.add(Dropout(0.3))
model.add(Dense(32, activation="relu"))
model.add(Dense(4, activation="softmax"))

# Compile model
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ----------------------------------
# Step 3: Training
# ----------------------------------
model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=16,
    validation_split=0.1
)

# ----------------------------------
# Step 4: Evaluation
# ----------------------------------
loss, accuracy = model.evaluate(X_test, y_test)
print("Model Accuracy:", accuracy)

# ----------------------------------
# Step 5: Student Performance Prediction
# ----------------------------------
# New student data:
# study_hours, attendance, previous_score, assignments, sleep_hours

new_student = np.array([[0.7, 0.9, 0.8, 0.6, 0.5]])
new_student = scaler.transform(new_student)

prediction = model.predict(new_student)
predicted_class = np.argmax(prediction)

predicted_performance = encoder.inverse_transform([predicted_class])

print("Predicted Student Performance:", predicted_performance[0])

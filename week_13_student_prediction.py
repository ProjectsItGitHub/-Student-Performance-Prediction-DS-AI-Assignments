# ======================================================
# WEEK 13 — MODEL DEPLOYMENT (Gradio)
# ======================================================

import gradio as gr
import numpy as np

# Dummy prediction function for student performance
def predict_performance(attendance, study_hours, previous_grade):
    # Dummy logic for demonstration
    score = (attendance * 0.4) + (study_hours * 0.4) + (previous_grade * 0.2)
    if score >= 85:
        result = "Excellent"
    elif score >= 70:
        result = "Good"
    elif score >= 50:
        result = "Average"
    else:
        result = "Poor"
    return {"Performance": result, "Score": round(score, 2)}

# Gradio interface
inputs = [
    gr.Number(label="Attendance (%)", value=75),
    gr.Number(label="Study Hours per Week", value=10),
    gr.Number(label="Previous Grade (%)", value=70)
]

outputs = [
    gr.Label(num_top_classes=1),
    gr.Number(label="Predicted Score")
]

gr.Interface(
    fn=predict_performance,
    inputs=inputs,
    outputs=outputs,
    title="Student Performance Prediction Demo",
    description="Enter student attendance, study hours, and previous grade to predict performance."
).launch()

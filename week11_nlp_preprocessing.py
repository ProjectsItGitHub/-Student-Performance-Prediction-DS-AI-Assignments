# ======================================================
# WEEK 11 — NLP PREPROCESSING
# Student Performance Prediction Project
# ======================================================

import nltk
import re
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK resources
nltk.download("stopwords")
nltk.download("punkt")

# ----------------------------------
# Step 1: Load your dataset
# ----------------------------------
# Example CSV structure: 'student_id', 'feedback', 'performance'
# Replace 'student_feedback.csv' with your actual file path
df = pd.read_csv("student_feedback.csv")

# Extract textual data (feedback/comments)
documents = df['feedback'].astype(str).tolist()

# ----------------------------------
# Step 2: Define preprocessing function
# ----------------------------------
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r"[^a-z ]", "", text)  # remove special characters/numbers
    tokens = text.split()  # tokenize
    tokens = [t for t in tokens if t not in stop_words]  # remove stopwords
    return " ".join(tokens)

# Clean all feedback
cleaned_docs = [clean_text(doc) for doc in documents]

# ----------------------------------
# Step 3: Convert text to TF-IDF features
# ----------------------------------
vectorizer = TfidfVectorizer()
X_text = vectorizer.fit_transform(cleaned_docs)

print("TF-IDF Shape:", X_text.shape)

# ----------------------------------
# Step 4 (Optional): Combine with other numerical features
# ----------------------------------
# Example: numeric features like 'attendance', 'assignments_score'
# from scipy.sparse import hstack
# X_numeric = df[['attendance', 'assignments_score']].values
# X = hstack([X_text, X_numeric])

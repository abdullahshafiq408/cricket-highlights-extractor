"""
models/ml_classifier.py
────────────────────────
Baseline Text Classifier: Uses TF-IDF and Logistic Regression.
Trained strictly on FOUR, SIX, and OUT from the Cricsheet dataset.
"""

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DATA_PATH = os.path.join(CURRENT_DIR, "commentary_data.csv")
MODEL_PATH = os.path.join(CURRENT_DIR, "commentary_ml_model.pkl")

_classifier = None

def _train_and_save_model():
    print(
        f"\n[ml_classifier] No saved model found. Training the baseline ML model on {CSV_DATA_PATH}...")

    if not os.path.exists(CSV_DATA_PATH):
        raise FileNotFoundError(
            f"Could not find {CSV_DATA_PATH}! Please put your dataset in the folder.")

    df = pd.read_csv(CSV_DATA_PATH)
    df = df.dropna(subset=['Commentary', 'score'])

    def determine_excitement(row):
        score_val = str(row['score']).strip().upper()
        if score_val in ['FOUR', 'SIX', 'OUT']:
            return 1
        return 0

    df['label'] = df.apply(determine_excitement, axis=1)

    X_train = df['Commentary']
    y_train = df['label']

    print(f"[ml_classifier] Processing {len(df)} rows of commentary...")

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english',
         max_features=5000, ngram_range=(1, 2))),
        ('clf', LogisticRegression(random_state=42, class_weight='balanced'))
    ])

    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, MODEL_PATH)
    print(
        f"[ml_classifier] Success! Baseline model trained and saved to {MODEL_PATH}\n")

    return pipeline


def _get_classifier():
    global _classifier
    if _classifier is None:
        if not os.path.exists(MODEL_PATH):
            _classifier = _train_and_save_model()
        else:
            print("[ml_classifier] Loading baseline ML model from disk...")
            _classifier = joblib.load(MODEL_PATH)
    return _classifier

def classify_segment(text: str) -> dict:
    if not text or len(text.strip()) < 5:
        return {"text": text, "excitement_score": 0.0, "label": "boring"}

    clf = _get_classifier()

    probabilities = clf.predict_proba([text])[0]

    excitement_score = float(probabilities[1])
    is_exciting = excitement_score >= 0.8

    return {
        "text":             text,
        "excitement_score": round(excitement_score, 4),
        "label":            "exciting" if is_exciting else "boring",
    }


def classify_all_segments(segments: list[dict]) -> list[dict]:
    print(
        f"[ml_classifier] Classifying {len(segments)} transcript segments...")
    results = []

    for seg in tqdm(segments, desc="ML classification"):
        classification = classify_segment(seg["text"])
        results.append({
            **seg,
            "excitement_score": classification["excitement_score"],
            "label":            classification["label"],
        })

    exciting_count = sum(1 for r in results if r["label"] == "exciting")
    print(
        f"[ml_classifier] {exciting_count}/{len(results)} segments classified as exciting")
    return results


def get_ml_score_at(classified_segments: list[dict], timestamp_sec: float, window_sec: float = 10.0) -> float:
    relevant = [
        seg["excitement_score"]
        for seg in classified_segments
        if seg["start"] - window_sec <= timestamp_sec <= seg["end"] + window_sec
    ]
    return max(relevant) if relevant else 0.0


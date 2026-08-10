"""
model.py — Train and evaluate text classifiers for support ticket categorisation.

Pipeline:
  1. Load ``data/tickets.csv``
  2. Preprocess each ticket (clean + combine subject & body)
  3. Vectorise with TF-IDF (unigrams + bigrams, max 5 000 features)
  4. Stratified 80 / 20 train / test split
  5. Train **Multinomial Naive Bayes** and **Logistic Regression**
  6. Print accuracy, classification report, and confusion matrix for both
  7. Save the best model + vectoriser to ``models/``

Run:
    python -m src.model
"""

import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from src.preprocessing import build_feature_text, download_nltk_data

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_PATH = _PROJECT_ROOT / "data" / "tickets.csv"
_MODEL_DIR = _PROJECT_ROOT / "models"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_data(path: Path = _DATA_PATH) -> pd.DataFrame:
    """Load the ticket dataset and add a ``text`` feature column."""
    df = pd.read_csv(path)
    # Drop rows with missing critical fields
    df.dropna(subset=["subject", "body", "category"], inplace=True)
    df["category"] = df["category"].str.strip().str.lower()
    df["text"] = df.apply(
        lambda row: build_feature_text(str(row["subject"]), str(row["body"])),
        axis=1,
    )
    return df


def train_and_evaluate() -> None:
    """End-to-end training and evaluation pipeline."""

    # ------------------------------------------------------------------
    # 1. Download NLTK data & load dataset
    # ------------------------------------------------------------------
    print("=" * 60)
    print("SUPPORT TICKET CLASSIFIER — TRAINING PIPELINE")
    print("=" * 60)

    download_nltk_data()
    df = load_data()
    print(f"\n[DATA] Dataset loaded: {len(df)} tickets")
    print(f"   Category distribution:\n{df['category'].value_counts().to_string()}\n")

    # ------------------------------------------------------------------
    # 2. TF-IDF Vectorisation
    # ------------------------------------------------------------------
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),     # unigrams + bigrams
        sublinear_tf=True,      # apply 1 + log(tf) dampening
        min_df=2,               # ignore very rare terms
    )

    X = vectorizer.fit_transform(df["text"])
    y = df["category"]

    print(f"[TFIDF] TF-IDF matrix: {X.shape[0]} samples x {X.shape[1]} features")

    # ------------------------------------------------------------------
    # 3. Train / test split (stratified)
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y,
    )
    print(f"   Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}\n")

    # ------------------------------------------------------------------
    # 4. Train classifiers
    # ------------------------------------------------------------------
    models = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=5.0,
            solver="lbfgs",
            random_state=42,
        ),
    }

    results: dict[str, float] = {}

    for name, clf in models.items():
        print("-" * 60)
        print(f"[MODEL] {name}")
        print("-" * 60)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        results[name] = acc

        print(f"\n   Accuracy: {acc:.2%}\n")
        print("   Classification Report:")
        print(classification_report(y_test, y_pred, digits=3, zero_division=0))

        cm = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))
        print("   Confusion Matrix:")
        labels = sorted(y.unique())
        header = "            " + "  ".join(f"{lb:>10s}" for lb in labels)
        print(header)
        for i, row in enumerate(cm):
            row_str = "  ".join(f"{v:>10d}" for v in row)
            print(f"   {labels[i]:>10s}  {row_str}")
        print()

    # ------------------------------------------------------------------
    # 5. Select best model & save
    # ------------------------------------------------------------------
    best_name = max(results, key=results.get)  # type: ignore[arg-type]
    best_model = models[best_name]
    best_acc = results[best_name]

    print("=" * 60)
    print(f"[BEST] Best model: {best_name}  (accuracy {best_acc:.2%})")
    print("=" * 60)

    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = _MODEL_DIR / "classifier.joblib"
    vectorizer_path = _MODEL_DIR / "vectorizer.joblib"
    metadata_path = _MODEL_DIR / "metadata.joblib"

    joblib.dump(best_model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(
        {
            "model_name": best_name,
            "accuracy": best_acc,
            "categories": sorted(y.unique().tolist()),
        },
        metadata_path,
    )

    print(f"\n[SAVED] Saved to {_MODEL_DIR}/")
    print(f"   - classifier.joblib   ({best_name})")
    print(f"   - vectorizer.joblib   (TF-IDF)")
    print(f"   - metadata.joblib     (model info)\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_and_evaluate()

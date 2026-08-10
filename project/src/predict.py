"""
predict.py — Single-ticket prediction engine.

Loads the trained model + vectoriser from ``models/`` and exposes a
``predict_ticket()`` function that returns:
  - predicted category
  - confidence percentage
  - priority tag (urgent / normal)
  - human-review flag (confidence < 60 %)

Also runnable as a CLI for quick testing:
    python -m src.predict "Subject here" "Body text here"
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from src.preprocessing import build_feature_text, clean_text, detect_priority, download_nltk_data

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MODEL_DIR = _PROJECT_ROOT / "models"

# ---------------------------------------------------------------------------
# Prediction result container
# ---------------------------------------------------------------------------

CONFIDENCE_THRESHOLD = 0.60  # Below this → route to human review


@dataclass
class TicketPrediction:
    """Structured prediction result for a single ticket."""

    category: str
    confidence: float          # 0.0 - 1.0
    priority: str              # "urgent" | "normal"
    needs_human_review: bool   # True if confidence < threshold

    @property
    def confidence_pct(self) -> str:
        return f"{self.confidence:.1%}"

    def __str__(self) -> str:
        review = " [!] NEEDS HUMAN REVIEW" if self.needs_human_review else ""
        return (
            f"Category:   {self.category.upper()}\n"
            f"Confidence: {self.confidence_pct}\n"
            f"Priority:   {self.priority}\n"
            f"{review}"
        ).strip()


# ---------------------------------------------------------------------------
# Model loading (lazy singleton)
# ---------------------------------------------------------------------------

_model = None
_vectorizer = None
_metadata = None


def _load_model():
    """Load model artifacts from disk (once)."""
    global _model, _vectorizer, _metadata
    if _model is None:
        _model = joblib.load(_MODEL_DIR / "classifier.joblib")
        _vectorizer = joblib.load(_MODEL_DIR / "vectorizer.joblib")
        _metadata = joblib.load(_MODEL_DIR / "metadata.joblib")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def predict_ticket(subject: str, body: str) -> TicketPrediction:
    """
    Predict the category of a single support ticket.

    Parameters
    ----------
    subject : str
        Ticket subject line.
    body : str
        Ticket body text.

    Returns
    -------
    TicketPrediction
        Structured result with category, confidence, priority, and review flag.
    """
    download_nltk_data()
    _load_model()

    # Preprocess
    feature_text = build_feature_text(subject, body)

    # Vectorise
    X = _vectorizer.transform([feature_text])

    # Predict with probabilities
    proba = _model.predict_proba(X)[0]
    best_idx = np.argmax(proba)
    category = _model.classes_[best_idx]
    confidence = float(proba[best_idx])

    # Priority from raw text (before cleaning strips keywords)
    raw_text = f"{subject} {body}"
    priority = detect_priority(raw_text)

    # Human-review flag
    needs_review = confidence < CONFIDENCE_THRESHOLD

    return TicketPrediction(
        category=category,
        confidence=confidence,
        priority=priority,
        needs_human_review=needs_review,
    )


def get_all_probabilities(subject: str, body: str) -> dict[str, float]:
    """Return probability distribution across all categories."""
    download_nltk_data()
    _load_model()

    feature_text = build_feature_text(subject, body)
    X = _vectorizer.transform([feature_text])
    proba = _model.predict_proba(X)[0]

    return {cls: float(p) for cls, p in zip(_model.classes_, proba)}


def get_metadata() -> dict:
    """Return saved model metadata (name, accuracy, categories)."""
    _load_model()
    return _metadata


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python -m src.predict \"<subject>\" \"<body>\"")
        sys.exit(1)

    subj = sys.argv[1]
    bod = sys.argv[2]

    print("\n" + "=" * 50)
    print("TICKET CLASSIFICATION")
    print("=" * 50)
    print(f"Subject: {subj}")
    print(f"Body:    {bod[:80]}{'...' if len(bod) > 80 else ''}")
    print("-" * 50)

    result = predict_ticket(subj, bod)
    print(result)

    print("\nAll category probabilities:")
    for cat, prob in sorted(get_all_probabilities(subj, bod).items()):
        bar = "#" * int(prob * 30)
        print(f"  {cat:>10s}: {prob:6.1%}  {bar}")
    print()

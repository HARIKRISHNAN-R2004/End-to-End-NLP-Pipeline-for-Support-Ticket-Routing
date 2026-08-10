"""
evaluate.py — End-to-end evaluation and demo of the ticket classifier.

Runs:
  1. Full test-set evaluation (classification report + confusion matrix)
  2. Prediction on 5+ custom new sample tickets
  3. Prints category, confidence, priority, and human-review flag for each

Usage:
    python evaluate.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.model import load_data, _MODEL_DIR
from src.predict import predict_ticket, get_all_probabilities, get_metadata
from src.preprocessing import download_nltk_data

import joblib


def print_section(title: str) -> None:
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


def evaluate_test_set() -> None:
    """Re-evaluate the saved model on the held-out test set."""
    print_section("TEST SET EVALUATION")

    download_nltk_data()

    # Load model artifacts
    model = joblib.load(_MODEL_DIR / "classifier.joblib")
    vectorizer = joblib.load(_MODEL_DIR / "vectorizer.joblib")
    meta = get_metadata()

    print(f"\n  Model:    {meta['model_name']}")
    print(f"  Train accuracy: {meta['accuracy']:.2%}")
    print(f"  Categories: {', '.join(meta['categories'])}")

    # Load and preprocess dataset
    df = load_data()
    from sklearn.model_selection import train_test_split

    X = vectorizer.transform(df["text"])
    y = df["category"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y,
    )

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n  Test Accuracy: {acc:.2%}")
    print(f"\n  Classification Report:\n")
    print(classification_report(y_test, y_pred, digits=3, zero_division=0))

    # Confusion matrix
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print("  Confusion Matrix:")
    header = "            " + "  ".join(f"{lb:>10s}" for lb in labels)
    print(header)
    for i, row in enumerate(cm):
        row_str = "  ".join(f"{v:>10d}" for v in row)
        print(f"  {labels[i]:>10s}  {row_str}")


def predict_new_samples() -> None:
    """Classify 8 hand-written sample tickets to demonstrate real-time usability."""
    print_section("NEW SAMPLE TICKET PREDICTIONS")

    samples = [
        # --- Clear-cut tickets ---
        {
            "subject": "Charged twice for the same order",
            "body": "I ordered a single item yesterday and I see two charges of $34.99 on my credit card. Please refund the duplicate.",
        },
        {
            "subject": "Cannot connect to VPN",
            "body": "Since the latest update I can't connect to the company VPN. It keeps timing out after 30 seconds. I've tried reinstalling the client.",
        },
        {
            "subject": "Requesting parental leave",
            "body": "My partner and I are expecting a baby in October. I'd like to understand the parental leave options and how to formally submit my request.",
        },
        {
            "subject": "Broken microwave in kitchen",
            "body": "The microwave on the 4th floor kitchen has stopped heating. It turns on but food stays cold. Can facilities replace it?",
        },
        # --- Edge-case / ambiguous tickets ---
        {
            "subject": "Help",
            "body": "Nothing is working. I need someone to call me back urgently.",
        },
        {
            "subject": "Question about my account",
            "body": "Hi, I have a question about my account. Can someone reach out?",
        },
        {
            "subject": "Server down and billing issue",
            "body": "Our production server is completely down and we're also getting wrong invoices. This is critical and needs immediate attention.",
        },
        {
            "subject": "Laptop and desk setup for new hire",
            "body": "We have a new team member starting Monday. They need a laptop configured with dev tools and a desk in building B.",
        },
    ]

    for i, sample in enumerate(samples, 1):
        print(f"\n{'-' * 65}")
        print(f"  Sample #{i}")
        print(f"  Subject: {sample['subject']}")
        body_preview = sample["body"][:80] + ("..." if len(sample["body"]) > 80 else "")
        print(f"  Body:    {body_preview}")
        print(f"{'-' * 65}")

        result = predict_ticket(sample["subject"], sample["body"])
        print(f"\n  [>] Category:    {result.category.upper()}")
        print(f"  [>] Confidence:  {result.confidence_pct}")
        print(f"  [>] Priority:    {result.priority}")
        if result.needs_human_review:
            print(f"  [!] FLAGGED FOR HUMAN REVIEW (confidence < 60%)")

        # Show probability distribution
        probs = get_all_probabilities(sample["subject"], sample["body"])
        print(f"\n  Probability breakdown:")
        for cat in sorted(probs.keys()):
            bar = "#" * int(probs[cat] * 30)
            marker = " <--" if cat == result.category else ""
            print(f"    {cat:>10s}: {probs[cat]:6.1%}  {bar}{marker}")


def main() -> None:
    print("\n" + "+" + "=" * 63 + "+")
    print("|" + " SUPPORT TICKET CLASSIFIER - EVALUATION REPORT ".center(63) + "|")
    print("+" + "=" * 63 + "+")

    evaluate_test_set()
    predict_new_samples()

    print_section("EVALUATION COMPLETE")
    print("\n  [OK] Test set evaluated with full metrics")
    print("  [OK] 8 new sample tickets classified (including edge cases)")
    print("  [OK] Confidence scores and priority tags generated")
    print("  [OK] Low-confidence tickets flagged for human review\n")


if __name__ == "__main__":
    main()

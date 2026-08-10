# Support Ticket Classifier

An ML-powered tool that automatically categorizes incoming support tickets into **Billing**, **Technical**, **HR**, or **General** — with confidence scoring, priority tagging, and a human-review fallback for ambiguous cases.

## Architecture

```
ticket text → preprocessing → TF-IDF vectorizer → classifier → prediction
                                                        ↓
                                             confidence < 60%?
                                             ├─ YES → "Needs Human Review"
                                             └─ NO  → auto-assign category
                                                        ↓
                                             keyword scan → priority tag
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python -m src.model
```

This will:
- Load and preprocess the 120-ticket dataset
- Train Multinomial Naive Bayes and Logistic Regression
- Evaluate both on a held-out test set
- Save the best model to `models/`

### 3. Run evaluation

```bash
python evaluate.py
```

Outputs full classification report, confusion matrix, and predictions on 8 new sample tickets (including edge cases).

### 4. Launch the Streamlit demo

```bash
streamlit run app.py
```

Interactive web UI for classifying individual tickets or uploading a CSV for batch processing.

### 5. CLI prediction

```bash
python -m src.predict "Invoice problem" "I was charged twice for my subscription this month"
```

## Project Structure

```
project/
├── data/
│   └── tickets.csv          # 120 labeled support tickets
├── models/                   # Saved model artifacts (after training)
│   ├── classifier.joblib
│   ├── vectorizer.joblib
│   └── metadata.joblib
├── src/
│   ├── __init__.py
│   ├── preprocessing.py      # Text cleaning pipeline
│   ├── model.py              # Training & evaluation
│   └── predict.py            # Single-ticket prediction
├── app.py                    # Streamlit live demo
├── evaluate.py               # End-to-end evaluation script
├── requirements.txt
├── REFLECTION.md
└── README.md
```

## Features

| Feature | Description |
|---------|-------------|
| **Text Preprocessing** | Lowercasing, HTML/URL/email removal, stopwords, lemmatization |
| **TF-IDF Vectorization** | Unigrams + bigrams, sublinear TF, max 5000 features |
| **Dual Model Training** | Naive Bayes and Logistic Regression with automatic best-model selection |
| **Confidence Scoring** | Probability-based confidence % for every prediction |
| **Human Review Routing** | Tickets below 60% confidence flagged for manual triage |
| **Priority Tagging** | Keyword-based urgent/normal detection |
| **Streamlit Demo** | Single ticket + batch CSV classification UI |
| **Edge Case Handling** | Ambiguous tickets get low confidence → human review |

## Model Choice Rationale

- **Multinomial Naive Bayes**: Designed for word-frequency features; extremely fast; strong baseline for text classification. Assumes feature independence, which works well with TF-IDF where features are relatively sparse.

- **Logistic Regression**: Handles feature correlations; outputs well-calibrated probabilities; interpretable via coefficients. Used as a comparison model — often edges out NB when categories share vocabulary.

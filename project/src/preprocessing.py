"""
preprocessing.py — Text cleaning pipeline for support ticket classification.

Handles:
  - Lowercasing
  - HTML tag / URL / email removal
  - Punctuation and digit stripping
  - Stopword removal (NLTK English stopwords)
  - Lemmatization (WordNet)
  - Subject weighting (repeat subject tokens to give them extra signal)
"""

import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ---------------------------------------------------------------------------
# Ensure required NLTK data is available
# ---------------------------------------------------------------------------

_NLTK_PACKAGES = ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]


def download_nltk_data() -> None:
    """Download NLTK data packages if not already present."""
    for pkg in _NLTK_PACKAGES:
        nltk.download(pkg, quiet=True)


# ---------------------------------------------------------------------------
# Core cleaning utilities
# ---------------------------------------------------------------------------

# Pre-compile regex patterns for speed
_RE_HTML = re.compile(r"<[^>]+>")
_RE_URL = re.compile(r"https?://\S+|www\.\S+")
_RE_EMAIL = re.compile(r"\S+@\S+\.\S+")
_RE_DIGITS = re.compile(r"\d+")
_RE_WHITESPACE = re.compile(r"\s+")


def _strip_noise(text: str) -> str:
    """Remove HTML tags, URLs, emails, digits, and punctuation."""
    text = _RE_HTML.sub(" ", text)
    text = _RE_URL.sub(" ", text)
    text = _RE_EMAIL.sub(" ", text)
    text = _RE_DIGITS.sub(" ", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    text = _RE_WHITESPACE.sub(" ", text).strip()
    return text


def clean_text(text: str) -> str:
    """
    Full cleaning pipeline for a single text string.

    Steps:
      1. Lowercase
      2. Remove HTML, URLs, emails, digits, punctuation
      3. Tokenize
      4. Remove stopwords
      5. Lemmatize
      6. Rejoin into a single string

    Returns the cleaned string (may be empty for very noisy inputs).
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()
    text = _strip_noise(text)

    tokens = word_tokenize(text)

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    cleaned_tokens = [
        lemmatizer.lemmatize(tok)
        for tok in tokens
        if tok not in stop_words and len(tok) > 1
    ]

    return " ".join(cleaned_tokens)


# ---------------------------------------------------------------------------
# Row-level feature builder
# ---------------------------------------------------------------------------

_SUBJECT_WEIGHT = 2  # Repeat subject tokens to amplify their signal


def build_feature_text(subject: str, body: str) -> str:
    """
    Combine subject and body into a single feature string.

    The subject is repeated ``_SUBJECT_WEIGHT`` times so that its tokens
    receive proportionally more weight in the TF-IDF matrix — subject lines
    are typically the most informative part of a ticket.
    """
    clean_subject = clean_text(subject)
    clean_body = clean_text(body)

    # Repeat subject tokens
    combined_parts = [clean_subject] * _SUBJECT_WEIGHT + [clean_body]
    return " ".join(part for part in combined_parts if part)


# ---------------------------------------------------------------------------
# Priority tagging (keyword-rule based)
# ---------------------------------------------------------------------------

_URGENT_KEYWORDS = {
    "urgent", "asap", "emergency", "critical", "down", "broken",
    "outage", "crashed", "immediately", "not working", "blocker",
    "production", "severity", "escalate", "p0", "p1",
}


def detect_priority(text: str) -> str:
    """
    Return 'urgent' if the raw ticket text contains any urgency keywords,
    otherwise 'normal'.

    This operates on the *original* (uncleaned) text so that phrases like
    "not working" aren't split by the preprocessing pipeline.
    """
    text_lower = text.lower()
    for keyword in _URGENT_KEYWORDS:
        if keyword in text_lower:
            return "urgent"
    return "normal"

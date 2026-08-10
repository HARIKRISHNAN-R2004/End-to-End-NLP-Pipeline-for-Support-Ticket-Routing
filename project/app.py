"""
app.py — Streamlit live demo for the Support Ticket Classifier.

Features:
  • Single ticket classification with subject + body input
  • Real-time prediction with category, confidence, priority, and review flag
  • Probability distribution bar chart
  • Batch mode — upload a CSV and classify all tickets at once
  • Model stats sidebar

Run:
    streamlit run app.py
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.predict import predict_ticket, get_all_probabilities, get_metadata
from src.preprocessing import download_nltk_data

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Ticket Classifier",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global font */
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main title gradient */
    .main-title {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0;
        letter-spacing: -0.02em;
    }

    .subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        font-weight: 400;
        margin-top: -8px;
        margin-bottom: 2rem;
    }

    /* Result card */
    .result-card {
        background: linear-gradient(145deg, #1e1e2e 0%, #2d2d44 100%);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 16px;
        padding: 1.8rem 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }

    .result-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #94a3b8;
        margin-bottom: 4px;
        font-weight: 600;
    }

    .result-value {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    /* Category badges */
    .badge-billing { color: #4ade80; }
    .badge-technical { color: #60a5fa; }
    .badge-hr { color: #f472b6; }
    .badge-general { color: #fbbf24; }

    /* Priority badges */
    .priority-urgent {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        display: inline-block;
    }

    .priority-normal {
        background: linear-gradient(135deg, #22c55e, #16a34a);
        color: white;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        display: inline-block;
    }

    /* Human review warning */
    .review-warning {
        background: linear-gradient(135deg, rgba(251, 191, 36, 0.15), rgba(245, 158, 11, 0.1));
        border: 1px solid rgba(251, 191, 36, 0.4);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin-top: 1rem;
        color: #fbbf24;
        font-weight: 600;
    }

    /* Confidence meter */
    .confidence-bar-bg {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        height: 12px;
        overflow: hidden;
        margin-top: 6px;
    }

    .confidence-bar-fill {
        height: 100%;
        border-radius: 8px;
        transition: width 0.5s ease;
    }

    /* Sidebar styling */
    .sidebar-stat {
        background: linear-gradient(145deg, #1e1e2e 0%, #2d2d44 100%);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }

    .sidebar-stat-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        font-weight: 600;
    }

    .sidebar-stat-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #e2e8f0;
    }

    /* Streamlit overrides */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: #1e1e2e !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 10px !important;
        color: #e2e8f0 !important;
        font-family: 'Inter', sans-serif !important;
    }

    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2) !important;
    }

    div.stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        letter-spacing: 0.02em;
        transition: all 0.3s ease !important;
    }

    div.stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

download_nltk_data()

CATEGORY_COLORS = {
    "billing": "#4ade80",
    "technical": "#60a5fa",
    "hr": "#f472b6",
    "general": "#fbbf24",
}

CATEGORY_ICONS = {
    "billing": "💳",
    "technical": "🔧",
    "hr": "👥",
    "general": "📋",
}

# ---------------------------------------------------------------------------
# Sidebar — Model stats
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 📊 Model Info")

    try:
        meta = get_metadata()
        st.markdown(f"""
        <div class="sidebar-stat">
            <div class="sidebar-stat-label">Algorithm</div>
            <div class="sidebar-stat-value">{meta['model_name'].split()[-2]} {meta['model_name'].split()[-1]}</div>
        </div>
        <div class="sidebar-stat">
            <div class="sidebar-stat-label">Training Accuracy</div>
            <div class="sidebar-stat-value">{meta['accuracy']:.1%}</div>
        </div>
        <div class="sidebar-stat">
            <div class="sidebar-stat-label">Categories</div>
            <div class="sidebar-stat-value">{len(meta['categories'])}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🏷️ Categories")
        for cat in meta["categories"]:
            icon = CATEGORY_ICONS.get(cat, "📌")
            color = CATEGORY_COLORS.get(cat, "#e2e8f0")
            st.markdown(
                f'<span style="color:{color}; font-weight:600;">{icon} {cat.title()}</span>',
                unsafe_allow_html=True,
            )

    except Exception:
        st.warning("Model not trained yet. Run `python -m src.model` first.")

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    confidence_threshold = st.slider(
        "Human review threshold",
        min_value=0.30,
        max_value=0.90,
        value=0.60,
        step=0.05,
        format="%.0f%%",
        help="Tickets with confidence below this threshold are flagged for human review.",
    )

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.markdown('<h1 class="main-title">🎫 Ticket Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Instantly categorize support tickets with ML-powered classification</p>', unsafe_allow_html=True)

tab_single, tab_batch = st.tabs(["✉️ Single Ticket", "📂 Batch Upload"])

# ---------------------------------------------------------------------------
# Tab 1: Single ticket
# ---------------------------------------------------------------------------

with tab_single:
    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown("#### Enter Ticket Details")
        subject = st.text_input(
            "Subject",
            placeholder="e.g., Cannot access my account",
            key="single_subject",
        )
        body = st.text_area(
            "Body",
            placeholder="Describe the issue in detail…",
            height=200,
            key="single_body",
        )
        classify_btn = st.button("🔍 Classify Ticket", use_container_width=True, key="classify_single")

    with col_result:
        if classify_btn and (subject.strip() or body.strip()):
            result = predict_ticket(subject, body)
            probs = get_all_probabilities(subject, body)

            cat_color = CATEGORY_COLORS.get(result.category, "#e2e8f0")
            cat_icon = CATEGORY_ICONS.get(result.category, "📌")
            priority_class = "priority-urgent" if result.priority == "urgent" else "priority-normal"

            # Confidence bar color
            if result.confidence >= 0.8:
                conf_color = "#4ade80"
            elif result.confidence >= 0.6:
                conf_color = "#fbbf24"
            else:
                conf_color = "#ef4444"

            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Predicted Category</div>
                <div class="result-value" style="color: {cat_color};">
                    {cat_icon} {result.category.upper()}
                </div>

                <div class="result-label" style="margin-top: 1rem;">Confidence</div>
                <div class="result-value" style="color: {conf_color}; font-size: 1.3rem;">
                    {result.confidence_pct}
                </div>
                <div class="confidence-bar-bg">
                    <div class="confidence-bar-fill" style="width: {result.confidence*100:.0f}%; background: {conf_color};"></div>
                </div>

                <div style="margin-top: 1.2rem;">
                    <div class="result-label">Priority</div>
                    <span class="{priority_class}">{result.priority.upper()}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Human review warning
            if result.confidence < confidence_threshold:
                st.markdown(f"""
                <div class="review-warning">
                    ⚠️ <strong>FLAGGED FOR HUMAN REVIEW</strong><br>
                    <span style="font-weight: 400; font-size: 0.9rem;">
                        Confidence ({result.confidence_pct}) is below the {confidence_threshold:.0%} threshold.
                        This ticket should be routed to a human agent for manual triage.
                    </span>
                </div>
                """, unsafe_allow_html=True)

            # Probability breakdown
            st.markdown("#### 📊 Probability Breakdown")
            prob_df = pd.DataFrame(
                [{"Category": cat.title(), "Probability": prob} for cat, prob in sorted(probs.items())],
            )
            prob_df = prob_df.set_index("Category")

            # Horizontal bar chart
            chart_data = pd.DataFrame(
                {"Probability": [probs.get(cat, 0) for cat in sorted(CATEGORY_COLORS.keys())]},
                index=[cat.title() for cat in sorted(CATEGORY_COLORS.keys())],
            )
            st.bar_chart(chart_data, horizontal=True, color="#667eea")

        elif classify_btn:
            st.warning("Please enter a subject or body text.")
        else:
            st.markdown("""
            <div class="result-card" style="text-align: center; padding: 3rem 2rem;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">🎫</div>
                <div style="color: #94a3b8; font-size: 1rem;">
                    Enter a ticket and click <strong>Classify</strong> to see results
                </div>
            </div>
            """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Tab 2: Batch upload
# ---------------------------------------------------------------------------

with tab_batch:
    st.markdown("#### Upload a CSV of Tickets")
    st.markdown(
        "CSV must have `subject` and `body` columns. "
        "Results will include predicted category, confidence, priority, and review flag."
    )

    uploaded = st.file_uploader("Choose a CSV file", type="csv", key="batch_upload")

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)

            if "subject" not in df.columns or "body" not in df.columns:
                st.error("CSV must contain `subject` and `body` columns.")
            else:
                with st.spinner("Classifying tickets…"):
                    results = []
                    for _, row in df.iterrows():
                        subj = str(row.get("subject", ""))
                        bod = str(row.get("body", ""))
                        pred = predict_ticket(subj, bod)
                        results.append({
                            "Subject": subj,
                            "Category": pred.category.upper(),
                            "Confidence": pred.confidence_pct,
                            "Priority": pred.priority.upper(),
                            "Needs Review": "⚠️ YES" if pred.confidence < confidence_threshold else "No",
                        })

                    result_df = pd.DataFrame(results)

                    # Stats
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Tickets", len(result_df))
                    with col2:
                        review_count = sum(1 for r in results if "YES" in r["Needs Review"])
                        st.metric("Needs Review", review_count)
                    with col3:
                        urgent_count = sum(1 for r in results if r["Priority"] == "URGENT")
                        st.metric("Urgent", urgent_count)

                    st.dataframe(result_df, use_container_width=True, hide_index=True)

                    # Download button
                    csv_data = result_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results CSV",
                        data=csv_data,
                        file_name="classified_tickets.csv",
                        mime="text/csv",
                    )

        except Exception as e:
            st.error(f"Error processing file: {e}")

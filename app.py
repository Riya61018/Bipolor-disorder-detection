"""Streamlit application for Bipolar Disorder EEG Detection and Model Comparison."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import BipolarEEGPredictor

MODEL_PATH = PROJECT_ROOT / "models" / "best_bipolar_model.joblib"
META_PATH = PROJECT_ROOT / "models" / "best_model_metadata.json"
COMPARISON_CSV = PROJECT_ROOT / "results" / "model_comparison.csv"
SAMPLE_TEST_CSV = PROJECT_ROOT / "data" / "processed" / "sample_test_patients.csv"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"

st.set_page_config(
    page_title="Bipolar Disorder EEG Detection",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Bipolar Disorder Detection Using EEG & Machine Learning")
st.caption(
    "Research & educational screening prototype trained on the BRMH EEG Psychiatric Disorders dataset. "
    "Not a substitute for clinical diagnosis."
)

if not MODEL_PATH.exists():
    st.error(
        "No trained model found yet. Please run `python src/train_and_compare.py` first."
    )
    st.stop()


@st.cache_resource
def load_predictor() -> BipolarEEGPredictor:
    return BipolarEEGPredictor(MODEL_PATH)


predictor = load_predictor()
metadata = json.loads(META_PATH.read_text(encoding="utf-8")) if META_PATH.exists() else {}

with st.sidebar:
    st.header("🏆 Selected Final Model")
    st.success(f"**{predictor.model_name}**")
    st.write(f"**Task:** `{predictor.task}`")
    st.write(f"**Classes:** `{predictor.class_names[0]}` vs `{predictor.class_names[1]}`")
    if predictor.metrics:
        st.metric("5-Fold CV ROC-AUC", f"{predictor.metrics.get('cv_roc_auc_mean', 0):.3f}")
        st.metric("Holdout Test ROC-AUC", f"{predictor.metrics.get('test_roc_auc', 0):.3f}")
        st.metric("Holdout Balanced Acc", f"{predictor.metrics.get('test_balanced_acc', 0):.3f}")
        st.metric("Holdout F1-Score", f"{predictor.metrics.get('test_f1', 0):.3f}")

tab_predict, tab_compare, tab_biomarkers = st.tabs(
    ["🔍 Final Model Prediction", "📊 10-Model Comparison", "🧬 EEG Biomarker Insights"]
)

with tab_predict:
    st.subheader(f"Predict with Winning Model: {predictor.model_name}")
    mode = st.radio(
        "Choose input source:",
        ["Test on Held-Out Unseen Patient Sample", "Upload Patient EEG CSV"],
        horizontal=True,
    )

    if mode == "Test on Held-Out Unseen Patient Sample":
        if not SAMPLE_TEST_CSV.exists():
            st.warning("Sample holdout file not found. Run training first.")
        else:
            df_samples = pd.read_csv(SAMPLE_TEST_CSV)
            row_idx = st.slider(
                "Select held-out test patient index:",
                min_value=0,
                max_value=len(df_samples) - 1,
                value=0,
            )
            patient_row = df_samples.iloc[[row_idx]]
            res = predictor.predict(patient_row).iloc[0]

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Predicted Class", res["predicted_label"])
            col2.metric("Bipolar Probability", f"{res['bipolar_probability'] * 100:.1f}%")
            col3.metric("Risk Tier", res["risk_tier"])
            if "true_label" in patient_row.columns:
                col4.metric("Actual Ground Truth", str(patient_row["true_label"].iloc[0]))

            st.progress(float(res["bipolar_probability"]))

            with st.expander("Batch evaluate all held-out test patients"):
                batch_res = predictor.predict(df_samples)
                st.dataframe(batch_res, use_container_width=True)
    else:
        uploaded = st.file_uploader("Upload a CSV with EEG features (matching BRMH columns)", type=["csv"])
        if uploaded is not None:
            df_up = pd.read_csv(uploaded)
            preds = predictor.predict(df_up)
            st.dataframe(preds, use_container_width=True)
            st.download_button(
                "Download Predictions CSV",
                data=preds.to_csv(index=False).encode("utf-8"),
                file_name="bipolar_eeg_predictions.csv",
                mime="text/csv",
            )

with tab_compare:
    st.subheader("Benchmark Comparison Across 10 Machine Learning Models")
    if COMPARISON_CSV.exists():
        df_comp = pd.read_csv(COMPARISON_CSV)
        st.dataframe(
            df_comp.style.highlight_max(
                subset=["cv_roc_auc_mean", "test_roc_auc", "test_balanced_acc", "test_f1"],
                color="#d1fae5",
            ),
            use_container_width=True,
        )

    c1, c2 = st.columns(2)
    comp_plot = PLOTS_DIR / "model_comparison.png"
    roc_plot = PLOTS_DIR / "roc_curves.png"
    if comp_plot.exists():
        c1.image(str(comp_plot), caption="CV & Holdout Performance Comparison", use_container_width=True)
    if roc_plot.exists():
        c2.image(str(roc_plot), caption="Holdout ROC Curves", use_container_width=True)

with tab_biomarkers:
    st.subheader("Confusion Matrix & Top Discriminative EEG Features")
    c1, c2 = st.columns(2)
    cm_plot = PLOTS_DIR / "best_model_confusion_matrix.png"
    feat_plot = PLOTS_DIR / "top_eeg_features.png"
    if cm_plot.exists():
        c1.image(str(cm_plot), caption="Best Model Confusion Matrix", use_container_width=True)
    if feat_plot.exists():
        c2.image(str(feat_plot), caption="Top EEG Features (Permutation Importance)", use_container_width=True)

    if metadata.get("top_features"):
        st.markdown("#### Top Ranked EEG Features")
        st.dataframe(pd.DataFrame(metadata["top_features"]), use_container_width=True)

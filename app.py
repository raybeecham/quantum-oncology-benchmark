"""Streamlit dashboard for completed benchmark artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Quantum Oncology Benchmark", layout="wide")
st.title("Quantum Oncology Benchmark")
st.error(
    "Research use only. This dashboard is not a diagnostic tool, medical device, "
    "treatment recommendation, or proof of quantum advantage."
)

artifact_dir = Path(st.sidebar.text_input("Artifact directory", "reports/latest"))
summary_path = artifact_dir / "summary.csv"
experiment_path = artifact_dir / "experiment.json"

if not summary_path.exists() or not experiment_path.exists():
    st.info("Run a benchmark first: qob benchmark --config configs/baseline.yaml")
    st.stop()

summary = pd.read_csv(summary_path)
experiment = json.loads(experiment_path.read_text(encoding="utf-8"))

st.subheader("Experiment Context")
left, middle, right = st.columns(3)
left.metric("Dataset", experiment["dataset"]["name"])
middle.metric("Samples", experiment["dataset"]["samples_used"])
right.metric("Selected Features", len(experiment["selected_features"]))

st.subheader("Model Comparison")
metric_columns = [
    "model",
    "model_family",
    "balanced_accuracy_mean",
    "sensitivity_mean",
    "specificity_mean",
    "f1_mean",
    "roc_auc_mean",
    "elapsed_seconds_mean",
]
st.dataframe(summary[metric_columns], use_container_width=True, hide_index=True)
st.bar_chart(summary.set_index("model")["balanced_accuracy_mean"])

st.subheader("Selected Features")
st.write(experiment["selected_features"])

if experiment.get("quantum_resources"):
    st.subheader("Quantum Resource Record")
    st.json(experiment["quantum_resources"])

st.subheader("Evidence Boundary")
st.markdown(
    "Quantum and classical models are evaluated on shared splits. A favorable quantum result "
    "in this small benchmark is hypothesis-generating only and requires independent replication."
)

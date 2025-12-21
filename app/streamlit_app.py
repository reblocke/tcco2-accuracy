"""Streamlit UI for TcCO2 → PaCO2 inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from tcco2_accuracy.data import load_paco2_distribution, prepare_paco2_distribution
from tcco2_accuracy.ui_api import predict_paco2_from_tcco2


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BOOTSTRAP_PATH = REPO_ROOT / "artifacts" / "bootstrap_params.csv"
DEFAULT_PACO2_PATH = REPO_ROOT / "Data" / "In Silico TCCO2 Database.dta"
DEFAULT_BINNED_PRIOR_PATH = REPO_ROOT / "artifacts" / "paco2_prior_bins.csv"

SUBGROUP_LABELS = {
    "Ambulatory / PFT": "pft",
    "ED / Inpatient": "ed_inp",
    "ICU": "icu",
}


@st.cache_data(show_spinner=False)
def _load_bootstrap_params(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def _load_paco2_distribution(path: str) -> pd.DataFrame:
    return prepare_paco2_distribution(load_paco2_distribution(Path(path)))


@st.cache_data(show_spinner=False)
def _load_binned_prior(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _prior_values_for_subgroup(
    subgroup: str,
    paco2_path: str,
    binned_path: str,
) -> np.ndarray:
    if Path(paco2_path).exists():
        prepared = _load_paco2_distribution(paco2_path)
        values = prepared.loc[prepared["subgroup"] == subgroup, "paco2"].to_numpy(dtype=float)
        if values.size == 0:
            raise ValueError(f"No PaCO2 values found for subgroup '{subgroup}'.")
        return values
    if Path(binned_path).exists():
        binned = _load_binned_prior(binned_path)
        required = {"subgroup", "paco2_bin", "count"}
        missing = required - set(binned.columns)
        if missing:
            raise ValueError(f"Binned prior missing columns: {sorted(missing)}")
        subset = binned.loc[binned["subgroup"] == subgroup]
        if subset.empty:
            raise ValueError(f"No binned priors available for subgroup '{subgroup}'.")
        return np.repeat(
            subset["paco2_bin"].to_numpy(dtype=float),
            subset["count"].to_numpy(dtype=int),
        )
    raise FileNotFoundError(
        "PaCO2 prior data not found. Provide the in-silico database or a binned prior CSV."
    )


def _format_interval(result) -> str:
    return (
        f"{result.paco2_median:.1f} "
        f"[{result.paco2_q_low:.1f}, {result.paco2_q_high:.1f}]"
    )


def _build_posterior_plot(result) -> go.Figure:
    highlight_mask = result.paco2_bin >= result.threshold
    # Highlight posterior mass above the hypercapnia threshold region.
    colors = np.where(highlight_mask, "rgba(220, 90, 70, 0.75)", "rgba(90, 140, 200, 0.75)")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=result.paco2_bin,
            y=result.posterior_prob,
            marker_color=colors,
            name="Posterior",
        )
    )
    if result.prior_prob is not None:
        # Prior curve shows the pretest PaCO2 distribution for comparison.
        fig.add_trace(
            go.Scatter(
                x=result.paco2_bin,
                y=result.prior_prob,
                mode="lines",
                line=dict(color="rgba(120, 120, 120, 0.7)", width=2),
                name="Prior",
            )
        )

    fig.add_vline(
        x=result.threshold,
        line_color="rgba(200, 60, 50, 0.9)",
        line_dash="dash",
        annotation_text=f"Threshold {result.threshold:g}",
    )
    fig.add_vline(
        x=result.paco2_q_low,
        line_color="rgba(30, 30, 30, 0.6)",
        line_dash="dot",
        annotation_text="PI low",
    )
    fig.add_vline(
        x=result.paco2_median,
        line_color="rgba(30, 30, 30, 0.9)",
        line_dash="solid",
        annotation_text="Median",
    )
    fig.add_vline(
        x=result.paco2_q_high,
        line_color="rgba(30, 30, 30, 0.6)",
        line_dash="dot",
        annotation_text="PI high",
    )

    fig.update_layout(
        title="Posterior PaCO2 distribution (conditioned on observed TcCO2)",
        xaxis_title="PaCO2 (mmHg)",
        yaxis_title="Posterior probability",
        legend_title="Distribution",
        bargap=0.05,
    )
    return fig


def main() -> None:
    st.set_page_config(page_title="TcCO2 → PaCO2 inference", layout="centered")
    st.title("TcCO2 → PaCO2 inference")
    st.warning("Research tool. Not for clinical decision-making.")

    st.sidebar.header("Inputs")
    tcco2 = st.sidebar.number_input("TcCO2 (mmHg)", min_value=0.0, value=45.0, step=0.1)
    subgroup_label = st.sidebar.selectbox("Setting", list(SUBGROUP_LABELS.keys()))
    threshold = st.sidebar.number_input("Hypercapnia threshold (mmHg)", value=45.0, step=0.5)
    mode_label = st.sidebar.radio(
        "Inference mode",
        ["Prior-weighted", "Likelihood-only"],
        help="Prior-weighted uses the empirical PaCO2 distribution as a pretest prior.",
    )
    interval = st.sidebar.select_slider(
        "Prediction interval",
        options=[0.90, 0.95, 0.99],
        value=0.95,
        format_func=lambda value: f"{value:.0%}",
    )

    with st.sidebar.expander("Advanced"):
        params_path = st.text_input("Bootstrap params path", value=str(DEFAULT_BOOTSTRAP_PATH))
        paco2_path = st.text_input("PaCO2 prior path", value=str(DEFAULT_PACO2_PATH))
        binned_path = st.text_input("Fallback binned prior path", value=str(DEFAULT_BINNED_PRIOR_PATH))
        n_param_draws_input = st.number_input(
            "Parameter draws (0 = all)",
            min_value=0,
            value=1000,
            step=100,
        )
        seed_input = st.number_input("Seed (0 = random)", min_value=0, value=202401, step=1)
        bin_width = st.number_input("Histogram bin width (mmHg)", min_value=0.1, value=1.0, step=0.1)

    mode = "prior_weighted" if mode_label == "Prior-weighted" else "likelihood_only"
    subgroup = SUBGROUP_LABELS[subgroup_label]
    n_param_draws = int(n_param_draws_input) if n_param_draws_input > 0 else None
    seed = int(seed_input) if seed_input > 0 else None

    try:
        params_draws = _load_bootstrap_params(params_path)
    except Exception as exc:
        st.error(f"Failed to load bootstrap parameter draws: {exc}")
        st.stop()

    paco2_prior_values = None
    if mode == "prior_weighted":
        try:
            paco2_prior_values = _prior_values_for_subgroup(subgroup, paco2_path, binned_path)
        except Exception as exc:
            st.error(str(exc))
            st.stop()

    try:
        result = predict_paco2_from_tcco2(
            tcco2=float(tcco2),
            subgroup=subgroup,
            threshold=float(threshold),
            mode=mode,
            interval=float(interval),
            params_draws=params_draws,
            paco2_prior_values=paco2_prior_values,
            n_param_draws=n_param_draws,
            seed=seed,
            bin_width=float(bin_width),
        )
    except Exception as exc:
        st.error(f"Inference failed: {exc}")
        st.stop()

    st.caption("Prediction intervals (PI) quantify expected PaCO2, not confidence intervals (CI).")
    col1, col2, col3 = st.columns(3)
    col1.metric("PaCO2 median [PI]", _format_interval(result))
    col2.metric(f"P(PaCO2 ≥ {threshold:g})", f"{result.p_ge_threshold:.3f}")
    col3.metric("TcCO2 decision", result.decision_label.title())

    st.subheader("Decision correctness")
    if result.decision_label == "positive":
        st.write(
            f"True positive: {result.p_true_positive:.3f} · "
            f"False positive: {result.p_false_positive:.3f}"
        )
    else:
        st.write(
            f"True negative: {result.p_true_negative:.3f} · "
            f"False negative: {result.p_false_negative:.3f}"
        )

    st.subheader("Posterior distribution")
    st.plotly_chart(_build_posterior_plot(result), use_container_width=True)
    st.caption(f"Posterior mass above threshold: {result.p_ge_threshold:.1%}.")


if __name__ == "__main__":
    main()

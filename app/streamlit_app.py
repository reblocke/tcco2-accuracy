"""Streamlit UI for TcCO2 → PaCO2 inference."""

from __future__ import annotations

from pathlib import Path
import hashlib
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from tcco2_accuracy.data import (
    INSILICO_PACO2_PATH,
    PACO2_PRIOR_BINS_PATH,
    REPO_ROOT,
    PriorLoadResult,
    load_paco2_prior,
    prepare_conway_meta_inputs,
)
from tcco2_accuracy.bootstrap import BOOTSTRAP_MODES, bootstrap_conway_parameters
from tcco2_accuracy.simulation import PACO2_TO_CONWAY_GROUP
from tcco2_accuracy.validate_inputs import validate_conway_studies_df
from tcco2_accuracy.ui_api import predict_paco2_from_tcco2


DEFAULT_STUDY_TABLE_PATH = REPO_ROOT / "Data" / "conway_studies.csv"
DEFAULT_PACO2_PATH = INSILICO_PACO2_PATH
DEFAULT_BINNED_PRIOR_PATH = PACO2_PRIOR_BINS_PATH

SUBGROUP_LABELS = {
    "All": "all",
    "Ambulatory / PFT": "pft",
    "ED / Inpatient": "ed_inp",
    "ICU": "icu",
}


@st.cache_data(show_spinner=False)
def _load_conway_studies(
    file_bytes: bytes | None,
    filename: str,
    default_path: str,
) -> pd.DataFrame:
    if file_bytes is None:
        data = pd.read_csv(default_path)
    else:
        buffer = io.BytesIO(file_bytes)
        if filename.lower().endswith(".csv"):
            data = pd.read_csv(buffer)
        elif filename.lower().endswith((".xlsx", ".xls")):
            data = pd.read_excel(buffer)
        else:
            raise ValueError("Uploaded study table must be CSV or XLSX.")
    validate_conway_studies_df(data)
    return data


@st.cache_data(show_spinner=False)
def _load_prior_values(
    subgroup: str,
    uploaded_bytes: bytes | None,
    uploaded_name: str,
    default_bins_path: str,
    insilico_path: str,
) -> PriorLoadResult:
    # Prior weights encode the empirical PaCO2 pretest distribution for inference.
    return load_paco2_prior(
        subgroup=subgroup,
        uploaded_bytes=uploaded_bytes,
        uploaded_name=uploaded_name,
        default_bins_path=Path(default_bins_path),
        insilico_path=Path(insilico_path),
    )


def _hash_study_table(studies: pd.DataFrame) -> str:
    stable = studies.sort_values("study_id").reset_index(drop=True)
    hashed = pd.util.hash_pandas_object(stable, index=True).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()


@st.cache_data(show_spinner=False)
def _bootstrap_draws(
    table_hash: str,
    subgroup: str,
    n_boot: int,
    seed: int | None,
    bootstrap_mode: str,
    studies: pd.DataFrame,
) -> pd.DataFrame:
    # "All" uses Conway main-analysis parameters (all studies).
    group_key = PACO2_TO_CONWAY_GROUP.get(subgroup, subgroup)
    if group_key == "icu":
        subset = studies[studies["is_icu"].astype(bool)]
    elif group_key == "arf":
        subset = studies[studies["is_arf"].astype(bool)]
    elif group_key == "lft":
        subset = studies[studies["is_lft"].astype(bool)]
    else:
        subset = studies
    if subset.empty:
        raise ValueError(f"No studies available for Conway group '{group_key}'.")
    conway_inputs = prepare_conway_meta_inputs(subset)
    draws = bootstrap_conway_parameters(
        conway_inputs,
        n_boot=n_boot,
        seed=seed,
        bootstrap_mode=bootstrap_mode,
        truncate_tau2=True,
    )
    _ = table_hash
    return draws


def _format_prior_source(result: PriorLoadResult | None, mode: str) -> str:
    if mode != "prior_weighted":
        return "Likelihood-only (prior not used)"
    if result is None:
        return "Prior not loaded"
    return {
        "uploaded": "Uploaded CSV",
        "default_bins": "Repo-shipped binned prior",
        "insilico_dta": "In-silico .dta fallback",
    }.get(result.source or "", "Unknown")


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
    st.warning("Research tool, not for clinical decision-making.")

    st.sidebar.header("Inputs")
    tcco2 = st.sidebar.number_input("TcCO2 (mmHg)", min_value=0.0, value=50.0, step=0.1)
    subgroup_labels = list(SUBGROUP_LABELS.keys())
    subgroup_label = st.sidebar.selectbox("Setting", subgroup_labels, index=0)
    threshold = st.sidebar.number_input("Hypercapnia threshold (mmHg)", value=45.0, step=0.5)
    mode_label = st.sidebar.radio(
        "Inference mode",
        ["Prior-weighted", "Likelihood-only"],
        index=0,
        help="Prior-weighted uses the empirical PaCO2 distribution as a pretest prior.",
    )
    interval = st.sidebar.selectbox(
        "Prediction interval (PI)",
        [0.95],
        format_func=lambda value: f"{value:.0%}",
        help="PI is the expected range for a new PaCO2 value; CI is for the mean.",
    )

    with st.sidebar.expander("Advanced"):
        study_table_upload = st.file_uploader(
            "Upload study table (CSV/XLSX)",
            type=["csv", "xlsx", "xls"],
        )
        st.caption(f"Default study table: `{DEFAULT_STUDY_TABLE_PATH}`")
        prior_upload = st.file_uploader(
            "Upload binned PaCO2 prior (CSV/XLSX)",
            type=["csv", "xlsx", "xls"],
        )
        st.caption(f"Default binned prior: `{DEFAULT_BINNED_PRIOR_PATH}`")
        paco2_path = st.text_input("PaCO2 prior path", value=str(DEFAULT_PACO2_PATH))
        binned_path = st.text_input("Fallback binned prior path", value=str(DEFAULT_BINNED_PRIOR_PATH))
        n_boot = st.number_input(
            "Bootstrap draws per subgroup",
            min_value=100,
            value=1000,
            step=100,
        )
        bootstrap_mode = st.selectbox(
            "Bootstrap mode",
            list(BOOTSTRAP_MODES),
            index=1,
        )
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

    upload_bytes = study_table_upload.getvalue() if study_table_upload else None
    upload_name = study_table_upload.name if study_table_upload else ""
    try:
        studies = _load_conway_studies(upload_bytes, upload_name, str(DEFAULT_STUDY_TABLE_PATH))
    except Exception as exc:
        st.error(f"Study table validation failed: {exc}")
        st.stop()
    table_hash = _hash_study_table(studies)
    study_source = study_table_upload.name if study_table_upload else DEFAULT_STUDY_TABLE_PATH.name
    st.caption(f"Study table source: `{study_source}` ({studies.shape[0]} studies).")

    try:
        params_draws = _bootstrap_draws(
            table_hash=table_hash,
            subgroup=subgroup,
            n_boot=int(n_boot),
            seed=seed,
            bootstrap_mode=str(bootstrap_mode),
            studies=studies,
        )
    except Exception as exc:
        st.error(f"Failed to generate bootstrap parameter draws: {exc}")
        st.stop()

    paco2_prior_values = None
    prior_result = None
    if mode == "prior_weighted":
        try:
            upload_bytes = prior_upload.getvalue() if prior_upload else None
            upload_name = prior_upload.name if prior_upload else None
            # "All" uses the pooled prior across subgroups, matching Conway main-analysis parameters.
            prior_result = _load_prior_values(
                subgroup=subgroup,
                uploaded_bytes=upload_bytes,
                uploaded_name=upload_name or "prior.csv",
                default_bins_path=binned_path,
                insilico_path=paco2_path,
            )
            if prior_result.error is not None:
                st.error(prior_result.error.message)
                st.stop()
            paco2_prior_values = prior_result.values
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

    with st.sidebar.expander("Debug"):
        st.write(f"Prior source: {_format_prior_source(prior_result, mode)}")
        if prior_result is not None and prior_result.paths_checked:
            checked_paths = [str(path) for path in prior_result.paths_checked]
        else:
            checked_paths = [str(Path(binned_path)), str(Path(paco2_path))]
        st.write("Paths checked:")
        for path in checked_paths:
            st.write(f"- `{path}`")


if __name__ == "__main__":
    main()

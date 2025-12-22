"""Conway meta-analysis calculations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class MetaEstimate:
    studies: int
    mean: float
    tau2: float
    var_model: float
    var_robust: float


@dataclass(frozen=True)
class LoaSummary:
    studies: int
    bias: float
    sd: float
    tau2: float
    loa_l: float
    loa_u: float
    ci_l_mod: float
    ci_u_mod: float
    ci_l_rve: float
    ci_u_rve: float


@dataclass(frozen=True)
class ConwayGroupSummary:
    studies: int
    n_pairs: int
    n_participants: int
    bias: float
    sd: float
    tau2: float
    loa_l: float
    loa_u: float
    ci_l: float
    ci_u: float


def repeated_measures_variance(s2: np.ndarray, n: np.ndarray, c: np.ndarray) -> np.ndarray:
    s2 = np.asarray(s2, dtype=float)
    n = np.asarray(n, dtype=float)
    c = np.asarray(c, dtype=float)
    return s2 * (1 + (c - 1) / (n - c))


def prepare_conway_inputs(data: pd.DataFrame) -> pd.DataFrame:
    """Prepare Conway meta-analysis inputs.

    Args:
        data: Study-level Conway data with columns ``n``, ``n_2``, ``bias``, and ``s2``.
            Optional column ``c`` (repeated measures per participant) is imputed as ``n / n_2``.

    Returns:
        DataFrame with additional columns:
        - ``c``: repeated measures count per participant.
        - ``s2_adj``: within-study variance adjusted for repeated measures.
        - ``v_bias``: variance of study-level bias estimates.
        - ``logs2``: log10-adjusted variance proxy used in Conway meta-analysis.
        - ``v_logs2``: variance for ``logs2``.

    Assumptions:
        ``n`` and ``n_2`` are positive and represent paired measurements and participants.
    """
    inputs = data.copy()
    if "c" not in inputs:
        inputs["c"] = inputs["n"] / inputs["n_2"]
    inputs["c"] = inputs["c"].fillna(inputs["n"] / inputs["n_2"])
    inputs["s2_adj"] = repeated_measures_variance(inputs["s2"], inputs["n"], inputs["c"])
    # v_bias captures finite-sample uncertainty in each study's bias estimate.
    inputs["v_bias"] = inputs["s2_adj"] / inputs["n_2"]
    inputs["logs2"] = np.log10(inputs["s2_adj"]) + 1 / (inputs["n_2"] - 1)
    # v_logs2 captures finite-sample uncertainty in each study's log-variance proxy.
    inputs["v_logs2"] = 2 / (inputs["n_2"] - 1)
    return inputs


def random_effects_meta(effect, variance, truncate_tau2: bool = False) -> MetaEstimate:
    """Compute a random-effects meta-analysis estimate.

    Args:
        effect: Array-like study effects (e.g., bias or log-variance).
        variance: Array-like within-study variances aligned with ``effect``.
        truncate_tau2: If True, truncate between-study variance at zero.

    Returns:
        ``MetaEstimate`` with pooled mean (delta when ``effect`` is bias),
        between-study variance (``tau2``), model-based variance, and robust variance.

    Assumptions:
        Non-finite effect/variance entries are dropped. When ``studies <= 1``,
        heterogeneity is not estimable, so ``tau2`` is set to 0 and robust
        variance equals model variance.
    """

    effect = np.asarray(effect, dtype=float)
    variance = np.asarray(variance, dtype=float)
    mask = np.isfinite(effect) & np.isfinite(variance)
    effect = effect[mask]
    variance = variance[mask]
    studies = effect.size
    if studies == 0:
        return MetaEstimate(studies=0, mean=0.0, tau2=0.0, var_model=0.0, var_robust=0.0)

    weights_fe = 1 / variance
    sum_wt = np.sum(weights_fe)
    mean_fe = np.sum(effect * weights_fe) / sum_wt
    if studies <= 1:
        tau2 = 0.0
        weights_re = 1 / (variance + tau2)
        sum_wt_re = np.sum(weights_re)
        var_model = float(1 / sum_wt_re) if sum_wt_re > 0 else 0.0
        return MetaEstimate(
            studies=studies,
            mean=float(mean_fe),
            tau2=tau2,
            var_model=var_model,
            var_robust=var_model,
        )

    q_stat = np.sum(weights_fe * (effect - mean_fe) ** 2)
    sum_wt_sq = np.sum(weights_fe**2)
    denominator = sum_wt - sum_wt_sq / sum_wt
    if not np.isfinite(denominator) or denominator <= 0:
        tau2 = 0.0
    else:
        tau2 = (q_stat - (studies - 1)) / denominator
        if not np.isfinite(tau2):
            tau2 = 0.0
    if truncate_tau2:
        tau2 = max(0.0, tau2)

    weights_re = 1 / (variance + tau2)
    sum_wt_re = np.sum(weights_re)
    if not np.isfinite(sum_wt_re) or sum_wt_re <= 0:
        mean_re = float(np.mean(effect))
        var_model = 0.0
    else:
        mean_re = np.sum(effect * weights_re) / sum_wt_re
        var_model = 1 / sum_wt_re
    var_robust = (
        (studies / (studies - 1))
        * np.sum(weights_re**2 * (effect - mean_re) ** 2)
        / (sum_wt_re**2)
    )
    if not np.isfinite(var_robust):
        var_robust = var_model
    return MetaEstimate(
        studies=studies,
        mean=float(mean_re),
        tau2=float(tau2),
        var_model=float(var_model),
        var_robust=float(var_robust),
    )


def loa_summary(bias, v_bias, logs2, v_logs2, truncate_tau2: bool = False) -> LoaSummary:
    """Summarize Conway bias/LoA estimates.

    Args:
        bias: Study-level bias estimates (PaCO2 - TcCO2).
        v_bias: Variance of bias estimates.
        logs2: Log-transformed variance proxy for within-study SD.
        v_logs2: Variance of ``logs2``.
        truncate_tau2: If True, truncate between-study variance at zero.

    Returns:
        ``LoaSummary`` containing pooled bias (delta), within-study SD
        (sqrt(sigma2)), between-study variance (tau2), LoA bounds, and 95%
        model/robust LoA confidence intervals.

    Notes:
        When there is fewer than two studies, LoA CIs are undefined and
        returned as NaN to avoid invalid t critical values.
    """

    bias_row = random_effects_meta(bias, v_bias, truncate_tau2=truncate_tau2)
    logs2_row = random_effects_meta(logs2, v_logs2, truncate_tau2=truncate_tau2)
    bias_mean = bias_row.mean
    sd2_est = float(np.exp(logs2_row.mean))
    tau_est = bias_row.tau2
    loa_l = bias_mean - 2 * np.sqrt(sd2_est + tau_est)
    loa_u = bias_mean + 2 * np.sqrt(sd2_est + tau_est)
    b1 = sd2_est**2 / (sd2_est + tau_est)
    b2 = tau_est**2 / (sd2_est + tau_est)
    v_logt2 = 2 / np.sum((np.asarray(v_bias) + tau_est) ** -2)
    v_loa_mod = bias_row.var_model + b1 * logs2_row.var_model + b2 * v_logt2
    v_loa_rve = bias_row.var_robust + b1 * logs2_row.var_robust + b2 * v_logt2
    df = bias_row.studies - 1
    if df <= 0:
        ci_l_mod = float("nan")
        ci_u_mod = float("nan")
        ci_l_rve = float("nan")
        ci_u_rve = float("nan")
    else:
        tcrit = stats.t.ppf(1 - 0.05 / 2, df)
        if not np.isfinite(tcrit):
            ci_l_mod = float("nan")
            ci_u_mod = float("nan")
            ci_l_rve = float("nan")
            ci_u_rve = float("nan")
        else:
            ci_l_mod = loa_l - tcrit * np.sqrt(v_loa_mod)
            ci_u_mod = loa_u + tcrit * np.sqrt(v_loa_mod)
            ci_l_rve = loa_l - tcrit * np.sqrt(v_loa_rve)
            ci_u_rve = loa_u + tcrit * np.sqrt(v_loa_rve)
    return LoaSummary(
        studies=bias_row.studies,
        bias=bias_mean,
        sd=float(np.sqrt(sd2_est)),
        tau2=tau_est,
        loa_l=loa_l,
        loa_u=loa_u,
        ci_l_mod=ci_l_mod,
        ci_u_mod=ci_u_mod,
        ci_l_rve=ci_l_rve,
        ci_u_rve=ci_u_rve,
    )


def _conway_descriptive_counts(data: pd.DataFrame) -> tuple[int, int, int]:
    studies = int(data.shape[0])
    n_pairs = float(pd.to_numeric(data["n"], errors="coerce").sum())
    n_participants = float(pd.to_numeric(data["n_2"], errors="coerce").sum())
    if data.attrs.get("group") not in {"main", "all"} or "study_base" not in data.columns:
        return studies, int(round(n_pairs)), int(round(n_participants))

    # Main-analysis counts group multi-row citations by base study ID to mirror Table 1.
    # Identical bias entries imply overlapping cohorts, so we keep max counts for that base.
    # This only changes descriptive totals; pooled estimates still use all rows.
    grouped = data.groupby("study_base", dropna=False)
    studies = int(grouped.ngroups)
    n_pairs = 0.0
    n_participants = 0.0
    for _, group in grouped:
        if group.shape[0] == 1:
            n_pairs += float(group["n"].iloc[0])
            n_participants += float(group["n_2"].iloc[0])
            continue
        bias_values = pd.to_numeric(group["bias"], errors="coerce").to_numpy(dtype=float)
        bias_span = np.nanmax(bias_values) - np.nanmin(bias_values)
        if np.isfinite(bias_span) and bias_span <= 1e-8:
            n_pairs += float(group["n"].max())
            n_participants += float(group["n_2"].max())
        else:
            n_pairs += float(group["n"].sum())
            n_participants += float(group["n_2"].sum())
    return studies, int(round(n_pairs)), int(round(n_participants))


def conway_group_summary(data, truncate_tau2: bool = False) -> ConwayGroupSummary:
    """Summarize Conway meta-analysis outputs for a subgroup.

    Args:
        data: Study-level Conway data for a subgroup.
        truncate_tau2: If True, truncate between-study variance at zero.

    Returns:
        ``ConwayGroupSummary`` with study counts, pooled bias (delta), within-study
        SD (sqrt(sigma2)), between-study variance (tau2), LoA bounds, and robust
        LoA confidence interval bounds.
    """

    inputs = prepare_conway_inputs(data)
    loa = loa_summary(
        inputs["bias"],
        inputs["v_bias"],
        inputs["logs2"],
        inputs["v_logs2"],
        truncate_tau2=truncate_tau2,
    )
    studies, n_pairs, n_participants = _conway_descriptive_counts(data)
    return ConwayGroupSummary(
        studies=studies,
        n_pairs=n_pairs,
        n_participants=n_participants,
        bias=loa.bias,
        sd=loa.sd,
        tau2=loa.tau2,
        loa_l=loa.loa_l,
        loa_u=loa.loa_u,
        ci_l=loa.ci_l_rve,
        ci_u=loa.ci_u_rve,
    )

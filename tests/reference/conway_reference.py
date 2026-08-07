"""Standalone reference equations for the Conway agreement meta-analysis.

This module is deliberately independent of the application package. It follows
Tipton and Shuster equations 4.4-4.5, 4.13-4.16, and 5.5 directly so tests can
compare the production implementation against a separately expressed reference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def repeated_measures_adjusted_variance(
    s2: np.ndarray,
    n_pairs: np.ndarray,
    replications: np.ndarray,
) -> np.ndarray:
    """Apply the conservative repeated-measures correction from equation 5.5."""

    s2 = np.asarray(s2, dtype=float)
    n_pairs = np.asarray(n_pairs, dtype=float)
    replications = np.asarray(replications, dtype=float)
    return s2 * (n_pairs - 1) / (n_pairs - replications)


def prepare_reference_inputs(data: pd.DataFrame) -> dict[str, np.ndarray]:
    """Derive agreement inputs from an analysis-format Conway study table."""

    n_pairs = data["n"].to_numpy(dtype=float)
    n_participants = data["n_2"].to_numpy(dtype=float)
    if "c" in data:
        replications = data["c"].fillna(data["n"] / data["n_2"]).to_numpy(dtype=float)
    else:
        replications = n_pairs / n_participants

    s2_adjusted = repeated_measures_adjusted_variance(
        data["s2"].to_numpy(dtype=float),
        n_pairs,
        replications,
    )
    return {
        "bias": data["bias"].to_numpy(dtype=float),
        "v_bias": s2_adjusted / n_participants,
        "log_sigma2": np.log(s2_adjusted) + 1 / (n_participants - 1),
        "var_log_sigma2": 2 / (n_participants - 1),
        "s2_adjusted": s2_adjusted,
    }


def dersimonian_laird_tau2(effect: np.ndarray, variance: np.ndarray) -> float:
    """Return the raw, potentially negative DerSimonian-Laird estimate."""

    effect = np.asarray(effect, dtype=float)
    variance = np.asarray(variance, dtype=float)
    if effect.ndim != 1 or variance.ndim != 1 or effect.shape != variance.shape:
        raise ValueError("effect and variance must be aligned one-dimensional arrays")
    if effect.size == 0 or not np.isfinite(effect).all():
        raise ValueError("effect must contain at least one finite value")
    if not np.isfinite(variance).all() or np.any(variance <= 0):
        raise ValueError("variance must contain finite positive values")
    if effect.size == 1:
        return 0.0

    weights_fixed = 1 / variance
    sum_weights_fixed = np.sum(weights_fixed)
    mean_fixed = np.sum(effect * weights_fixed) / sum_weights_fixed
    q_stat = np.sum(weights_fixed * (effect - mean_fixed) ** 2)
    denominator = sum_weights_fixed - np.sum(weights_fixed**2) / sum_weights_fixed
    return float((q_stat - (effect.size - 1)) / denominator)


def random_effects_reference(
    effect: np.ndarray,
    variance: np.ndarray,
    *,
    truncate_tau2: bool,
) -> dict[str, float]:
    """Pool effects using independently expressed random-effects equations."""

    effect = np.asarray(effect, dtype=float)
    variance = np.asarray(variance, dtype=float)
    tau2_raw = dersimonian_laird_tau2(effect, variance)
    tau2 = max(0.0, tau2_raw) if truncate_tau2 else tau2_raw
    adjusted_variance = variance + tau2
    if np.any(adjusted_variance <= 0):
        raise ValueError("raw tau2 does not define positive random-effects variances")

    weights_random = 1 / adjusted_variance
    sum_weights_random = np.sum(weights_random)
    mean_random = np.sum(effect * weights_random) / sum_weights_random
    var_model = 1 / sum_weights_random
    if effect.size == 1:
        var_robust = var_model
    else:
        var_robust = (
            (effect.size / (effect.size - 1))
            * np.sum(weights_random**2 * (effect - mean_random) ** 2)
            / sum_weights_random**2
        )
    return {
        "mean": float(mean_random),
        "tau2_raw": tau2_raw,
        "tau2": float(tau2),
        "var_model": float(var_model),
        "var_robust": float(var_robust),
    }


def agreement_reference_from_inputs(
    bias: np.ndarray,
    v_bias: np.ndarray,
    log_sigma2: np.ndarray,
    var_log_sigma2: np.ndarray,
    *,
    truncate_tau2: bool = True,
) -> dict[str, float]:
    """Evaluate the corrected agreement equations from prepared study inputs."""

    bias_meta = random_effects_reference(
        bias,
        v_bias,
        truncate_tau2=truncate_tau2,
    )
    log_sigma2_meta = random_effects_reference(
        log_sigma2,
        var_log_sigma2,
        truncate_tau2=truncate_tau2,
    )
    sigma2 = float(np.exp(log_sigma2_meta["mean"]))
    tau2 = bias_meta["tau2"]
    total_variance = sigma2 + tau2

    v_bias = np.asarray(v_bias, dtype=float)
    var_tau2 = float(2 / np.sum((v_bias + tau2) ** -2))
    b_sigma2 = sigma2**2 / total_variance
    b_tau2 = 1 / total_variance
    var_loa_model = (
        bias_meta["var_model"] + b_sigma2 * log_sigma2_meta["var_model"] + b_tau2 * var_tau2
    )
    var_loa_robust = (
        bias_meta["var_robust"] + b_sigma2 * log_sigma2_meta["var_robust"] + b_tau2 * var_tau2
    )
    half_width = 2 * np.sqrt(total_variance)
    loa_l = bias_meta["mean"] - half_width
    loa_u = bias_meta["mean"] + half_width

    studies = np.asarray(bias).size
    if studies <= 1:
        ci_l_mod = ci_u_mod = ci_l = ci_u = float("nan")
    else:
        tcrit = stats.t.ppf(0.975, studies - 1)
        ci_l_mod = loa_l - tcrit * np.sqrt(var_loa_model)
        ci_u_mod = loa_u + tcrit * np.sqrt(var_loa_model)
        ci_l = loa_l - tcrit * np.sqrt(var_loa_robust)
        ci_u = loa_u + tcrit * np.sqrt(var_loa_robust)

    return {
        "bias": bias_meta["mean"],
        "sd": float(np.sqrt(sigma2)),
        "sigma2": sigma2,
        "tau2_raw": bias_meta["tau2_raw"],
        "tau2": tau2,
        "var_tau2": var_tau2,
        "var_loa_model": float(var_loa_model),
        "var_loa_robust": float(var_loa_robust),
        "loa_l": float(loa_l),
        "loa_u": float(loa_u),
        "ci_l_mod": float(ci_l_mod),
        "ci_u_mod": float(ci_u_mod),
        "ci_l": float(ci_l),
        "ci_u": float(ci_u),
    }


def corrected_reference(data: pd.DataFrame) -> dict[str, float]:
    """Evaluate the corrected agreement equations for a Conway study table."""

    inputs = prepare_reference_inputs(data)
    return agreement_reference_from_inputs(
        inputs["bias"],
        inputs["v_bias"],
        inputs["log_sigma2"],
        inputs["var_log_sigma2"],
    )

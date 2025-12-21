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
    inputs = data.copy()
    if "c" not in inputs:
        inputs["c"] = inputs["n"] / inputs["n_2"]
    inputs["c"] = inputs["c"].fillna(inputs["n"] / inputs["n_2"])
    inputs["s2_adj"] = repeated_measures_variance(inputs["s2"], inputs["n"], inputs["c"])
    inputs["v_bias"] = inputs["s2_adj"] / inputs["n_2"]
    inputs["logs2"] = np.log10(inputs["s2_adj"]) + 1 / (inputs["n_2"] - 1)
    inputs["v_logs2"] = 2 / (inputs["n_2"] - 1)
    return inputs


def random_effects_meta(effect, variance) -> MetaEstimate:
    effect = np.asarray(effect, dtype=float)
    variance = np.asarray(variance, dtype=float)
    mask = np.isfinite(effect) & np.isfinite(variance)
    effect = effect[mask]
    variance = variance[mask]
    studies = effect.size
    weights_fe = 1 / variance
    mean_fe = np.sum(effect * weights_fe) / np.sum(weights_fe)
    q_stat = np.sum(weights_fe * (effect - mean_fe) ** 2)
    sum_wt = np.sum(weights_fe)
    sum_wt_sq = np.sum(weights_fe**2)
    tau2 = (q_stat - (studies - 1)) / (sum_wt - sum_wt_sq / sum_wt)
    weights_re = 1 / (variance + tau2)
    mean_re = np.sum(effect * weights_re) / np.sum(weights_re)
    var_model = 1 / np.sum(weights_re)
    var_robust = (
        (studies / (studies - 1))
        * np.sum(weights_re**2 * (effect - mean_re) ** 2)
        / (np.sum(weights_re) ** 2)
    )
    return MetaEstimate(
        studies=studies,
        mean=mean_re,
        tau2=tau2,
        var_model=var_model,
        var_robust=var_robust,
    )


def loa_summary(bias, v_bias, logs2, v_logs2) -> LoaSummary:
    bias_row = random_effects_meta(bias, v_bias)
    logs2_row = random_effects_meta(logs2, v_logs2)
    bias_mean = bias_row.mean
    sd2_est = float(np.exp(logs2_row.mean))
    tau_est = bias_row.tau2
    loa_l = bias_mean - 2 * np.sqrt(sd2_est + tau_est)
    loa_u = bias_mean + 2 * np.sqrt(sd2_est + tau_est)
    tcrit = stats.t.ppf(1 - 0.05 / 2, bias_row.studies - 1)
    b1 = sd2_est**2 / (sd2_est + tau_est)
    b2 = tau_est**2 / (sd2_est + tau_est)
    v_logt2 = 2 / np.sum((np.asarray(v_bias) + tau_est) ** -2)
    v_loa_mod = bias_row.var_model + b1 * logs2_row.var_model + b2 * v_logt2
    v_loa_rve = bias_row.var_robust + b1 * logs2_row.var_robust + b2 * v_logt2
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


def conway_group_summary(data) -> ConwayGroupSummary:
    inputs = prepare_conway_inputs(data)
    loa = loa_summary(inputs["bias"], inputs["v_bias"], inputs["logs2"], inputs["v_logs2"])
    return ConwayGroupSummary(
        studies=int(inputs.shape[0]),
        n_pairs=int(round(float(inputs["n"].sum()))),
        n_participants=int(round(float(inputs["n_2"].sum()))),
        bias=loa.bias,
        sd=loa.sd,
        tau2=loa.tau2,
        loa_l=loa.loa_l,
        loa_u=loa.loa_u,
        ci_l=loa.ci_l_rve,
        ci_u=loa.ci_u_rve,
    )

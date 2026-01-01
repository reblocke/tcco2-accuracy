"""Forward simulation utilities for TcCO2 accuracy."""

from __future__ import annotations

import math
import warnings
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .bland_altman import loa_bounds, total_sd
from .data import PACO2_SUBGROUP_ORDER, prepare_paco2_distribution
from .utils import quantile_key, safe_ratio, safe_ratio_inf, validate_params_df


DEFAULT_CLASSIFICATION_THRESHOLDS: tuple[float, ...] = (45.0,)
DEFAULT_D_QUANTILES: tuple[float, ...] = (0.025, 0.975)
DEFAULT_SUMMARY_QUANTILES: tuple[float, ...] = (0.025, 0.5, 0.975)
PACO2_TO_CONWAY_GROUP: dict[str, str] = {
    "pft": "lft",
    "ed_inp": "arf",
    "icu": "icu",
    # "All" uses Conway main-analysis parameters (all studies).
    "all": "main",
}


def simulate_forward(
    paco2_data: pd.DataFrame,
    params: pd.DataFrame,
    thresholds: Sequence[float] = DEFAULT_CLASSIFICATION_THRESHOLDS,
    mode: str = "analytic",
    seed: int | None = None,
    n_draws: int | None = None,
    n_mc: int | None = None,
) -> pd.DataFrame:
    prepared = paco2_data if "subgroup" in paco2_data.columns else prepare_paco2_distribution(paco2_data)
    random_state = np.random.default_rng(seed)
    available_groups = set(params["group"]) if "group" in params.columns else set()
    frames: list[pd.DataFrame] = []
    for subgroup in PACO2_SUBGROUP_ORDER:
        paco2_values = prepared.loc[prepared["subgroup"] == subgroup, "paco2"].to_numpy(dtype=float)
        if "group" in params.columns:
            group_key = subgroup if subgroup in available_groups else PACO2_TO_CONWAY_GROUP.get(subgroup, subgroup)
            group_params = params[params["group"] == group_key]
        else:
            group_params = params
        if group_params.empty:
            warnings.warn(
                f"No parameters found for subgroup '{subgroup}'; falling back to all params.",
                UserWarning,
            )
            group_params = params
        if n_draws is not None and n_draws < group_params.shape[0]:
            chosen = random_state.choice(group_params.index.to_numpy(), size=n_draws, replace=True)
            group_params = group_params.loc[chosen].reset_index(drop=True)
        group_seed = int(random_state.integers(0, np.iinfo(np.uint32).max))
        metrics = simulate_forward_metrics(
            paco2_values,
            group_params,
            thresholds=thresholds,
            mode=mode,
            seed=group_seed,
            n_mc=n_mc,
        )
        if "group" in metrics.columns:
            metrics = metrics.rename(columns={"group": "param_group"})
        metrics.insert(0, "group", subgroup)
        frames.append(metrics)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def simulate_forward_metrics(
    paco2_values: np.ndarray,
    params: pd.DataFrame,
    thresholds: Sequence[float] = DEFAULT_CLASSIFICATION_THRESHOLDS,
    mode: str = "analytic",
    seed: int | None = None,
    n_mc: int | None = None,
) -> pd.DataFrame:
    random_state = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []
    params = validate_params_df(params)
    for index, params_row in enumerate(params.itertuples(index=False)):
        delta = float(getattr(params_row, "delta"))
        sigma2 = float(getattr(params_row, "sigma2"))
        tau2 = float(getattr(params_row, "tau2"))
        sd_within = math.sqrt(sigma2)
        # Bootstrap draws encode epistemic uncertainty; sd_total captures measurement error + heterogeneity.
        sd_total = total_sd(sd_within, tau2)
        base_row = _base_row_from_params(params_row, index, sd_total)
        if mode == "analytic":
            difference_summary = difference_moments(delta, sd_total, quantiles=DEFAULT_D_QUANTILES)
            loa_lower, loa_upper = loa_bounds(delta, sd_within, tau2)
            for threshold_value in thresholds:
                classification = expected_classification_metrics(paco2_values, delta, sd_total, threshold_value)
                row = {
                    **base_row,
                    **difference_summary,
                    "threshold": float(threshold_value),
                    "loa_l": loa_lower,
                    "loa_u": loa_upper,
                    **classification,
                }
                rows.append(row)
        elif mode == "monte_carlo":
            rows.extend(
                _monte_carlo_rows(
                    paco2_values,
                    delta,
                    sd_total,
                    sd_within,
                    tau2,
                    thresholds,
                    base_row,
                    random_state,
                    n_mc,
                )
            )
        else:
            raise ValueError(f"Unknown simulation mode: {mode}")
    return pd.DataFrame(rows)


def difference_moments(
    delta: float,
    sd_total: float,
    quantiles: Sequence[float] = DEFAULT_D_QUANTILES,
) -> dict[str, float]:
    if sd_total <= 0:
        raise ValueError("Total SD must be positive.")
    quantile_values = stats.norm.ppf(quantiles, loc=delta, scale=sd_total)
    moment_summary: dict[str, float] = {"d_mean": delta, "d_sd": sd_total}
    for quantile_value, quantile in zip(quantile_values, quantiles):
        moment_summary[quantile_key("d", quantile)] = float(quantile_value)
    return moment_summary


def expected_classification_metrics(
    paco2_values: np.ndarray,
    delta: float,
    sd_total: float,
    threshold_value: float,
    population_n: int | None = None,
) -> dict[str, float]:
    paco2_values = np.asarray(paco2_values, dtype=float)
    if paco2_values.size == 0:
        raise ValueError("PaCO2 distribution must be non-empty.")
    threshold_value = float(threshold_value)
    mean_tcco2 = paco2_values - delta
    z_scores = (threshold_value - mean_tcco2) / sd_total
    prob_positive = 1 - stats.norm.cdf(z_scores)
    positive_mask = paco2_values >= threshold_value
    negative_mask = ~positive_mask
    total_count = paco2_values.size
    population_n = int(population_n) if population_n is not None else int(total_count)
    if population_n <= 0:
        raise ValueError("Population size must be positive for expected counts.")
    # Expected fractions across the subgroup distribution integrate measurement error analytically.
    true_positive = prob_positive[positive_mask].sum() / total_count
    false_negative = (1 - prob_positive[positive_mask]).sum() / total_count
    false_positive = prob_positive[negative_mask].sum() / total_count
    true_negative = (1 - prob_positive[negative_mask]).sum() / total_count
    prevalence = positive_mask.mean()
    sensitivity = safe_ratio(true_positive, true_positive + false_negative)
    specificity = safe_ratio(true_negative, true_negative + false_positive)
    ppv = safe_ratio(true_positive, true_positive + false_positive)
    npv = safe_ratio(true_negative, true_negative + false_negative)
    accuracy = true_positive + true_negative
    tp_rate = true_positive
    fp_rate = false_positive
    tn_rate = true_negative
    fn_rate = false_negative
    misclass_rate = fp_rate + fn_rate
    lr_pos = safe_ratio_inf(sensitivity, 1 - specificity)
    lr_neg = safe_ratio_inf(1 - sensitivity, specificity)
    tp_count = tp_rate * population_n
    fp_count = fp_rate * population_n
    tn_count = tn_rate * population_n
    fn_count = fn_rate * population_n
    return {
        "prevalence": float(prevalence),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv": float(ppv),
        "npv": float(npv),
        "accuracy": float(accuracy),
        "tp_rate": float(tp_rate),
        "fp_rate": float(fp_rate),
        "tn_rate": float(tn_rate),
        "fn_rate": float(fn_rate),
        "misclass_rate": float(misclass_rate),
        "lr_pos": float(lr_pos),
        "lr_neg": float(lr_neg),
        "tp_count": float(tp_count),
        "fp_count": float(fp_count),
        "tn_count": float(tn_count),
        "fn_count": float(fn_count),
        "tp_per_1000": float(tp_rate * 1000),
        "fp_per_1000": float(fp_rate * 1000),
        "tn_per_1000": float(tn_rate * 1000),
        "fn_per_1000": float(fn_rate * 1000),
        "misclass_per_1000": float(misclass_rate * 1000),
    }


def summarize_simulation_metrics(
    metrics: pd.DataFrame,
    quantiles: Sequence[float] = DEFAULT_SUMMARY_QUANTILES,
) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    group_columns = ["group"] if "group" in metrics.columns else []
    if "threshold" in metrics.columns:
        group_columns.append("threshold")
    value_columns = [
        column
        for column in (
            "d_mean",
            "d_sd",
            "d_q025",
            "d_q975",
            "loa_l",
            "loa_u",
            "prevalence",
            "sensitivity",
            "specificity",
            "ppv",
            "npv",
            "accuracy",
            "tp_rate",
            "fp_rate",
            "tn_rate",
            "fn_rate",
            "misclass_rate",
            "lr_pos",
            "lr_neg",
            "tp_count",
            "fp_count",
            "tn_count",
            "fn_count",
            "tp_per_1000",
            "fp_per_1000",
            "tn_per_1000",
            "fn_per_1000",
            "misclass_per_1000",
        )
        if column in metrics.columns
    ]
    summary = metrics.groupby(group_columns)[value_columns].quantile(quantiles).unstack(-1)
    summary.columns = [
        f"{metric}_{quantile_key('', quantile).lstrip('_')}" for metric, quantile in summary.columns
    ]
    return summary.reset_index()


def _monte_carlo_rows(
    paco2_values: np.ndarray,
    delta: float,
    sd_total: float,
    sd_within: float,
    tau2: float,
    thresholds: Sequence[float],
    base_row: dict[str, float | int],
    random_state: np.random.Generator,
    n_mc: int | None,
) -> list[dict[str, float | int]]:
    sample_size = int(n_mc) if n_mc is not None else int(paco2_values.size)
    if sample_size <= 0:
        raise ValueError("Monte Carlo sample size must be positive.")
    paco2_sample = random_state.choice(paco2_values, size=sample_size, replace=True)
    difference_sample = random_state.normal(delta, sd_total, size=sample_size)
    tcco2_sample = paco2_sample - difference_sample
    difference_summary = _difference_sample_summary(difference_sample)
    loa_lower, loa_upper = loa_bounds(delta, sd_within, tau2)
    rows: list[dict[str, float | int]] = []
    for threshold_value in thresholds:
        classification = _sample_classification_metrics(paco2_sample, tcco2_sample, threshold_value)
        row = {
            **base_row,
            **difference_summary,
            "threshold": float(threshold_value),
            "loa_l": loa_lower,
            "loa_u": loa_upper,
            **classification,
        }
        rows.append(row)
    return rows


def _difference_sample_summary(difference_sample: np.ndarray) -> dict[str, float]:
    if difference_sample.size == 0:
        raise ValueError("Difference sample must be non-empty.")
    quantiles = np.quantile(difference_sample, DEFAULT_D_QUANTILES)
    return {
        "d_mean": float(np.mean(difference_sample)),
        "d_sd": float(np.std(difference_sample, ddof=0)),
        quantile_key("d", DEFAULT_D_QUANTILES[0]): float(quantiles[0]),
        quantile_key("d", DEFAULT_D_QUANTILES[1]): float(quantiles[1]),
    }


def _sample_classification_metrics(
    paco2_sample: np.ndarray,
    tcco2_sample: np.ndarray,
    threshold_value: float,
) -> dict[str, float]:
    if paco2_sample.size == 0:
        raise ValueError("Samples must be non-empty.")
    threshold_value = float(threshold_value)
    positive_mask = paco2_sample >= threshold_value
    predicted_positive = tcco2_sample >= threshold_value
    true_positive = np.sum(positive_mask & predicted_positive)
    false_negative = np.sum(positive_mask & ~predicted_positive)
    false_positive = np.sum(~positive_mask & predicted_positive)
    true_negative = np.sum(~positive_mask & ~predicted_positive)
    total_count = paco2_sample.size
    prevalence = true_positive + false_negative
    negative_total = true_negative + false_positive
    sensitivity = safe_ratio(true_positive, prevalence)
    specificity = safe_ratio(true_negative, negative_total)
    ppv = safe_ratio(true_positive, true_positive + false_positive)
    npv = safe_ratio(true_negative, true_negative + false_negative)
    accuracy = safe_ratio(true_positive + true_negative, total_count)
    tp_rate = safe_ratio(true_positive, total_count)
    fp_rate = safe_ratio(false_positive, total_count)
    tn_rate = safe_ratio(true_negative, total_count)
    fn_rate = safe_ratio(false_negative, total_count)
    misclass_rate = fp_rate + fn_rate
    lr_pos = safe_ratio_inf(sensitivity, 1 - specificity)
    lr_neg = safe_ratio_inf(1 - sensitivity, specificity)
    return {
        "prevalence": float(prevalence / total_count),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv": float(ppv),
        "npv": float(npv),
        "accuracy": float(accuracy),
        "tp_rate": float(tp_rate),
        "fp_rate": float(fp_rate),
        "tn_rate": float(tn_rate),
        "fn_rate": float(fn_rate),
        "misclass_rate": float(misclass_rate),
        "lr_pos": float(lr_pos),
        "lr_neg": float(lr_neg),
        "tp_count": float(true_positive),
        "fp_count": float(false_positive),
        "tn_count": float(true_negative),
        "fn_count": float(false_negative),
        "tp_per_1000": float(tp_rate * 1000),
        "fp_per_1000": float(fp_rate * 1000),
        "tn_per_1000": float(tn_rate * 1000),
        "fn_per_1000": float(fn_rate * 1000),
        "misclass_per_1000": float(misclass_rate * 1000),
    }


def _base_row_from_params(
    params_row: tuple,
    index: int,
    sd_total: float,
) -> dict[str, float | int]:
    base_row: dict[str, float | int] = {
        "replicate": int(getattr(params_row, "replicate", index)),
        "delta": float(getattr(params_row, "delta")),
        "sigma2": float(getattr(params_row, "sigma2")),
        "tau2": float(getattr(params_row, "tau2")),
        "sd_total": float(sd_total),
    }
    if hasattr(params_row, "group"):
        base_row["group"] = getattr(params_row, "group")
    return base_row

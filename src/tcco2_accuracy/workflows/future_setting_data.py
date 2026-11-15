"""Prepare versioned public summary data for the offline future-setting model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import block_diag

from ..core.conway_meta import repeated_measures_variance

ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT / "models/future_setting"
SPEC = json.loads((MODEL_DIR / "specification.json").read_text())
GROUPS = ("main", "icu", "arf", "lft")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_ledger():
    ledger = pd.read_csv(MODEL_DIR / "source_ledger.csv", keep_default_na=False)
    source = pd.read_csv(ROOT / "Data/conway_studies.csv").set_index("study_id")
    if (
        ledger.analysis_unit_id.duplicated().any()
        or ledger.study_id.duplicated().any()
        or set(ledger.study_id) != set(source.index)
    ):
        raise ValueError("Ledger must account for each public source row once")
    if not np.allclose(
        ledger[["archive_bias", "archive_s2"]],
        source.loc[ledger.study_id, ["bias", "s2"]],
        rtol=1e-10,
    ):
        raise ValueError("Ledger and public source revisions disagree")
    if not set(ledger.status) <= {"cleared", "held"} or not set(ledger.difference_orientation) == {
        "arterial_minus_transcutaneous"
    }:
        raise ValueError("Invalid adjudication or difference direction")
    for labels in ledger.strata:
        if not labels or not set(labels.split("|")) <= set(GROUPS):
            raise ValueError("Unknown reference class")
    return ledger


def select_units(ledger, group):
    if group not in GROUPS:
        raise ValueError("Unknown source stratum")
    selection = ledger.loc[ledger.strata.map(lambda labels: group in labels.split("|"))]
    if selection.empty:
        raise ValueError("Missing source stratum")
    return selection


def prepare_summaries(frame, blocks=()):
    if frame.empty or not frame.status.eq("cleared").all():
        raise ValueError("Unresolved source unit blocks this fit")
    ids = frame.analysis_unit_id.tolist()
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate analysis unit")
    m = frame.participants.to_numpy(float)
    n = frame.n_pairs.to_numpy(float)
    s2 = frame.variance_input.to_numpy(float).copy()
    if not (
        np.all(m == np.floor(m))
        and np.all(n == np.floor(n))
        and np.all(m > 1)
        and np.all(n >= m)
        and np.all(np.isfinite(s2))
        and np.all(s2 > 0)
    ):
        raise ValueError("Invalid source counts or variance")
    kinds = set(frame.repeat_action)
    if not kinds <= {"retain", "adjust_once"}:
        raise ValueError("Unresolved repeated-measure adjustment")
    needs = frame.repeat_action.eq("adjust_once").to_numpy()
    if frame.loc[needs, "variance_kind"].str.contains("adjusted").any():
        raise ValueError("Cannot adjust an already adjusted variance")
    s2[needs] = repeated_measures_variance(s2[needs], n[needs], n[needs] / m[needs])
    y = np.column_stack([frame.bias, np.log(s2) + 1 / (m - 1)])
    v = np.zeros((len(m), 2, 2))
    v[:, 0, 0], v[:, 1, 1] = s2 / m, 2 / (m - 1)
    v[:, 0, 1] = v[:, 1, 0] = frame.sampling_cov
    return stan_data(y, v, ids, blocks)


def stan_data(y, v, ids, blocks=()):
    """Blocks carry a full sampling covariance ordered (bias, log variance) per unit."""
    y, v = np.asarray(y, float), np.asarray(v, float)
    if (
        len(set(ids)) != len(ids)
        or y.shape != (len(ids), 2)
        or not len(y)
        or v.shape != (len(y), 2, 2)
        or not np.isfinite(y).all()
        or not np.isfinite(v).all()
        or not np.allclose(v, v.transpose(0, 2, 1))
        or np.min(np.linalg.eigvalsh(v)) <= 0
    ):
        raise ValueError("Invalid summary observations or sampling covariance")
    dependent = []
    covariances = []
    latent_ids = []
    sizes = []
    for b in blocks:
        units = b["analysis_units"]
        if (
            len(units) < 2
            or len(set(units)) != len(units)
            or set(units) & set(dependent)
            or not set(units) <= set(ids)
        ):
            raise ValueError("Invalid or overlapping covariance block")
        c = np.asarray(b["sampling_covariance"], float)
        if (
            c.shape != (2 * len(units), 2 * len(units))
            or not np.isfinite(c).all()
            or not np.allclose(c, c.T)
            or np.linalg.eigvalsh(c).min() <= 0
        ):
            raise ValueError("Invalid joint sampling covariance")
        for j, unit in enumerate(units):
            if not np.allclose(c[2 * j : 2 * j + 2, 2 * j : 2 * j + 2], v[ids.index(unit)]):
                raise ValueError("Block marginal sampling covariance disagrees with its ledger row")
        setting = b["setting_id"]
        if len(setting) != len(units) or any(not isinstance(s, int) or s < 1 for s in setting):
            raise ValueError("Invalid latent setting assignment")
        dependent.extend(units)
        sizes.append(len(units))
        covariances.append(c)
        latent_ids.extend(setting)
    independent = [j for j, key in enumerate(ids) if key not in dependent]
    order = independent + [ids.index(key) for key in dependent]
    return dict(
        N=len(y),
        I=len(independent),
        bias=y[order, 0],
        ell=y[order, 1],
        v_bias=v[independent, 0, 0],
        v_ell=v[independent, 1, 1],
        cov_bias_ell=v[independent, 0, 1],
        B=len(sizes),
        block_size=sizes,
        V_dep=block_diag(*covariances) if sizes else np.empty((0, 0)),
        setting_id=latent_ids,
        **SPEC["prior"],
    )


def recovery_data():
    rng = np.random.default_rng(88031)
    n = 160
    latent = rng.multivariate_normal([1, np.log(9)], [[4, -0.4], [-0.4, 0.25]], size=n)
    m = rng.integers(15, 100, n)
    v = np.zeros((n, 2, 2))
    v[:, 0, 0] = 9 / m
    v[:, 1, 1] = 2 / (m - 1)
    v[:, 0, 1] = v[:, 1, 0] = 0.15 * np.sqrt(v[:, 0, 0] * v[:, 1, 1])
    y = latent + np.einsum("nij,nj->ni", np.linalg.cholesky(v), rng.normal(size=(n, 2)))
    return stan_data(y, v, list(range(n)))

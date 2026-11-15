"""Fixed empirical inverse inference in the caller's approved private process."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import special

from ..core.downstream import PatientInputColumns, _normalize_patient_schema, _select_index_records
from ..core.future_setting import LEVELS, METHOD, PARAMETERS, PredictiveKernel, inverse_summary
from ..core.paco2 import prepare_paco2_distribution
from ..reporting.future_setting import GRID, batch_mcse
from .downstream import _validate_target_data_revision
from .future_setting_data import SPEC, digest


@dataclass(frozen=True)
class EmpiricalPriorMetadata:
    """Opaque caller-managed references to the approved cohort, window and eligibility."""

    cohort_revision: str
    eligibility_revision: str
    window_revision: str

    def __post_init__(self):
        for value in (self.cohort_revision, self.eligibility_revision, self.window_revision):
            _validate_target_data_revision(value)


def load_candidate_kernel(run_dir, target):
    """Read only a public posterior with passed sampler, precision and identity checks."""
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    group = SPEC["target_mapping"].get(target)
    info = manifest.get("strata", {}).get(group, {})
    if (
        not group
        or manifest.get("mode") != "public"
        or manifest.get("run_status") != "complete"
        or manifest.get("method_version") != METHOD
        or info.get("status") != "candidate"
        or info.get("target_group") != target
    ):
        raise ValueError("No cleared public posterior for this target")
    attempt = run_dir / group / info["attempt"]
    receipt = json.loads((attempt / "sampler.json").read_text())
    precision = json.loads((attempt / "precision.json").read_text())
    path = attempt / "posterior.npz"
    if not (
        receipt["sampler_pass"]
        and precision["passed"]
        and info.get("posterior_sha256") == receipt["posterior_sha256"] == digest(path)
        and info.get("precision_sha256") == digest(attempt / "precision.json")
    ):
        raise ValueError("Invalid posterior receipt")
    with np.load(path) as data:
        theta = data["theta"]
        if (
            list(data["parameters"]) != list(PARAMETERS)
            or data["method_version"].item() != METHOD
            or not np.array_equal(data["chain_ids"], [1, 2, 3, 4])
            or theta.shape != (4, receipt["retained_per_chain"], 5)
            or not np.array_equal(data["draw_ids"], np.arange(1, theta.shape[1] + 1))
        ):
            raise ValueError("Invalid posterior type, chain or draw identity")
    return PredictiveKernel(theta, order=precision["quadrature_order"]), manifest


def index_support(records, target, columns=PatientInputColumns()):
    if target not in SPEC["target_mapping"]:
        raise ValueError("Unknown target group")
    records = _normalize_patient_schema(prepare_paco2_distribution(records), columns=columns)
    if target != "all":
        records = records.loc[records.subgroup == target]
    if records.empty:
        raise ValueError("No index records for the requested target")
    index = _select_index_records(records, columns=columns)
    return np.unique(index.paco2.to_numpy(float), return_counts=True)


def _empirical_precision(kernel, a, w, t, summary):
    # Stream across support to bound memory even for unbinned private priors.
    quantiles = summary["quantiles"]
    accum = np.zeros((8, *kernel.theta.shape[:2]))
    logmass = np.log(w / w.sum())
    log_evidence = special.logsumexp(logmass + kernel.logpdf(a - t))
    for support, prior in zip(a, logmass, strict=True):
        joint = np.exp(kernel.draw_logpdf(support - t) + prior - log_evidence)
        accum[0] += joint
        if support >= 45:
            accum[1] += joint
        for j, q in enumerate(quantiles):
            if support < q:
                accum[2 + 2 * j] += joint
            if support <= q:
                accum[3 + 2 * j] += joint
    den = accum[0]
    ratios = [v.mean() / den.mean() for v in accum[1:]]
    se = [batch_mcse((v - r * den) / den.mean()) for v, r in zip(accum[1:], ratios, strict=True)]
    resolved = all(
        ratios[1 + 2 * i] + 2 * se[1 + 2 * i] < q <= ratios[2 + 2 * i] - 2 * se[2 + 2 * i]
        for i, q in enumerate(LEVELS)
    )
    return se[0], resolved


def infer_index_cohort(
    run_dir, records, *, target, metadata: EmpiricalPriorMetadata, columns=PatientInputColumns()
):
    """Return private aggregates for caller-supplied eligible index records; no file I/O on patient data."""
    if not isinstance(metadata, EmpiricalPriorMetadata):
        raise TypeError("Supply explicit cohort, eligibility and window revisions")
    kernel, manifest = load_candidate_kernel(run_dir, target)
    a, w = index_support(records, target, columns)
    rows = []
    # Recheck quadrature for the new support. The public gate used a synthetic prior.
    coarser = PredictiveKernel(kernel.theta, order=kernel.order // 2)
    for t in GRID:
        summary = inverse_summary(kernel, a, w, t)
        low = inverse_summary(coarser, a, w, t)
        se, resolved = _empirical_precision(kernel, a, w, t, summary)
        error = abs(low["probability_ge"] - summary["probability_ge"])
        if (
            se > 0.00025
            or not resolved
            or error > 0.0001
            or not np.array_equal(low["quantiles"], summary["quantiles"])
        ):
            raise ValueError("Numerical precision is insufficient for the supplied fixed prior")
        lower, median, upper = summary["quantiles"]
        rows.append(
            dict(
                quantity_type="posterior_paco2_given_tcco2",
                method_version=METHOD,
                agreement_run=manifest["run_id"],
                source_revision=manifest["input_sha256"]["models/future_setting/source_ledger.csv"],
                prior_revision=metadata.cohort_revision,
                eligibility_revision=metadata.eligibility_revision,
                window_revision=metadata.window_revision,
                target_stratum=target,
                stratum=SPEC["target_mapping"][target],
                prior_uncertainty="fixed_empirical",
                reference_standard="arterial PaCO2",
                support_representation="unique observed index values",
                sampling_unit="one index measurement per patient",
                index_policy="earliest eligible measurement in earliest eligible encounter",
                tcco2=t,
                unit="mmHg",
                interval_level=0.95,
                lower=lower,
                median=median,
                upper=upper,
                threshold=45,
                operator=">=",
                probability_ge=summary["probability_ge"],
                probability_mcse=se,
                probability_quadrature_error=error,
                result_status="private_candidate",
            )
        )
    return pd.DataFrame(rows)

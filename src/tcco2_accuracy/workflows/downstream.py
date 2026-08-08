"""In-memory draw-aligned downstream analysis workflow.

This workflow is intentionally separate from legacy artifact rebuilding and
manuscript reporting. It performs no filesystem I/O and has no publication or
authorization gate; callers supply in-memory patient and Conway study data.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from ..core._params import select_conway_studies_for_subgroup
from ..core.bootstrap import BOOTSTRAP_MODES, bootstrap_group_draws
from ..core.conway_meta import AGREEMENT_METHOD_VERSION, RESULTS_STATUS
from ..core.downstream import (
    MAX_DEGENERATE_PROPOSAL_FRACTION,
    MAX_TARGET_RESAMPLE_ATTEMPTS,
    TARGET_RESAMPLING_POLICY,
    DownstreamAnalysisConfig,
    PatientInputColumns,
    _is_stable,
    _JointDraws,
    _monte_carlo_stability,
    _run_draw_aligned_downstream,
    _summarize_joint_draws,
)
from ..data import prepare_conway_meta_inputs

MINIMUM_DOWNSTREAM_DRAWS = 10_000


@dataclass(frozen=True)
class DownstreamWorkflowConfig:
    """Execution settings for the specified draw-aligned downstream workflow."""

    analysis: DownstreamAnalysisConfig = DownstreamAnalysisConfig()
    n_boot: int = MINIMUM_DOWNSTREAM_DRAWS
    seed: int = 202401
    repeat_seed: int = 202402
    agreement_clustering: Literal["publication", "effect_row"] = "publication"
    bootstrap_mode: str = "cluster_plus_withinstudy"
    enforce_minimum_draws: bool = True
    assess_stability: bool = True
    require_stability: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.n_boot, bool) or not isinstance(self.n_boot, int) or self.n_boot <= 0:
            raise ValueError("n_boot must be a positive integer.")
        if self.enforce_minimum_draws and self.n_boot < MINIMUM_DOWNSTREAM_DRAWS:
            raise ValueError(
                f"n_boot must be at least {MINIMUM_DOWNSTREAM_DRAWS} for this workflow."
            )
        for name, value in (("seed", self.seed), ("repeat_seed", self.repeat_seed)):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer.")
        if self.assess_stability and self.seed == self.repeat_seed:
            raise ValueError("seed and repeat_seed must be distinct for stability assessment.")
        if self.agreement_clustering not in {"publication", "effect_row"}:
            raise ValueError(f"Unknown agreement clustering: {self.agreement_clustering}")
        if self.bootstrap_mode not in BOOTSTRAP_MODES:
            raise ValueError(f"Unknown bootstrap mode: {self.bootstrap_mode}")
        if self.require_stability and not self.assess_stability:
            raise ValueError("require_stability=True requires assess_stability=True.")
        deviations: list[str] = []
        if self.agreement_clustering != "publication":
            deviations.append("effect_row_clustering")
        if self.analysis.parameter_mapping != "setting_specific":
            deviations.append("pooled_main_parameters")
        if self.analysis.measurement_policy != "index":
            deviations.append("all_measurements")
        if self.analysis.support != "all":
            deviations.append("central_95_support")
        if len(deviations) > 1:
            raise ValueError(
                f"Downstream sensitivities must be run one at a time; requested {deviations}."
            )


@dataclass(frozen=True)
class DownstreamWorkflowResult:
    """Aggregate summaries and non-sensitive reproducibility metadata only."""

    core: pd.DataFrame
    prediction: pd.DataFrame
    two_stage: pd.DataFrame
    stability: pd.DataFrame
    manifest: dict[str, object]


class MonteCarloStabilityError(ValueError):
    """Expose aggregate diagnostics when the configured stability gate fails."""

    def __init__(self, stability: pd.DataFrame) -> None:
        super().__init__(
            "Monte Carlo error exceeded one tenth of reporting precision in the predetermined "
            "independent repeat; increase n_boot and inspect the attached aggregate diagnostics."
        )
        self.stability = stability


def run_downstream_analysis(
    patient_data: pd.DataFrame,
    conway_studies: pd.DataFrame,
    *,
    target_data_revision: str,
    config: DownstreamWorkflowConfig = DownstreamWorkflowConfig(),
    columns: PatientInputColumns = PatientInputColumns(),
) -> DownstreamWorkflowResult:
    """Run the in-memory, draw-aligned downstream analysis.

    The caller must supply patient-level data with the columns described by
    ``PatientInputColumns`` plus ``paco2`` and either ``subgroup`` or the raw
    subgroup flags. ``target_data_revision`` is a caller-supplied non-sensitive
    extract/version label. No paths are
    accepted, no output is written, and returned tables contain aggregate
    estimates only. The default configuration uses 10,000 publication-cluster
    agreement draws and an independent Monte Carlo stability repeat.
    """

    target_data_revision = _validate_target_data_revision(target_data_revision)
    subgroup_input_mode = "prepared_subgroup" if "subgroup" in patient_data.columns else "raw_flags"
    conway_data = prepare_conway_meta_inputs(conway_studies)
    group_data = _conway_groups(conway_data)
    primary = _run_independent_joint_draws(
        patient_data,
        group_data,
        config=config,
        columns=columns,
        seed=config.seed,
    )
    core, prediction, two_stage = _summarize_joint_draws(primary, config=config.analysis)

    stability = pd.DataFrame()
    repeat_seeds: list[int] = []
    stable = False
    repeat: _JointDraws | None = None
    if config.assess_stability:
        repeat = _run_independent_joint_draws(
            patient_data,
            group_data,
            config=config,
            columns=columns,
            seed=config.repeat_seed,
        )
        stability = _monte_carlo_stability(
            primary,
            repeat,
            repeat_seed=config.repeat_seed,
        )
        repeat_seeds.append(config.repeat_seed)
        stable = _is_stable(stability)
        if config.require_stability and not stable:
            raise MonteCarloStabilityError(stability)

    manifest = _build_manifest(
        conway_studies,
        target_data_revision=target_data_revision,
        subgroup_input_mode=subgroup_input_mode,
        primary=primary,
        repeat=repeat,
        config=config,
        columns=columns,
        repeat_seeds=repeat_seeds,
        stability_passed=stable if config.assess_stability else None,
    )
    return DownstreamWorkflowResult(
        core=core,
        prediction=prediction,
        two_stage=two_stage,
        stability=stability,
        manifest=manifest,
    )


def _conway_groups(conway_data: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    return [
        ("main", select_conway_studies_for_subgroup(conway_data, "all")),
        ("icu", select_conway_studies_for_subgroup(conway_data, "icu")),
        ("arf", select_conway_studies_for_subgroup(conway_data, "ed_inp")),
        ("lft", select_conway_studies_for_subgroup(conway_data, "pft")),
    ]


def _run_independent_joint_draws(
    patient_data: pd.DataFrame,
    group_data: list[tuple[str, pd.DataFrame]],
    *,
    config: DownstreamWorkflowConfig,
    columns: PatientInputColumns,
    seed: int,
) -> _JointDraws:
    agreement_seed, target_seed = _child_seeds(seed)
    cluster_column = "study_base" if config.agreement_clustering == "publication" else "study"
    if cluster_column not in group_data[0][1].columns:
        raise ValueError(
            f"Conway inputs are missing requested clustering column '{cluster_column}'."
        )
    params = bootstrap_group_draws(
        group_data,
        n_boot=config.n_boot,
        seed=agreement_seed,
        study_id=cluster_column,
        truncate_tau2=True,
        bootstrap_mode=config.bootstrap_mode,
    )
    return _run_draw_aligned_downstream(
        patient_data,
        params,
        config=config.analysis,
        columns=columns,
        seed=target_seed,
    )


def _child_seeds(seed: int) -> tuple[int, int]:
    sequence = np.random.SeedSequence(seed)
    agreement, target = sequence.spawn(2)
    return (
        int(agreement.generate_state(1, dtype=np.uint32)[0]),
        int(target.generate_state(1, dtype=np.uint32)[0]),
    )


def _build_manifest(
    conway_studies: pd.DataFrame,
    *,
    target_data_revision: str,
    subgroup_input_mode: str,
    primary: _JointDraws,
    repeat: _JointDraws | None,
    config: DownstreamWorkflowConfig,
    columns: PatientInputColumns,
    repeat_seeds: list[int],
    stability_passed: bool | None,
) -> dict[str, object]:
    """Return JSON-safe metadata without source paths or target-derived details."""

    return {
        "schema_version": "tcco2_downstream_joint_bootstrap_v2",
        "agreement_method_version": AGREEMENT_METHOD_VERSION,
        "results_status": RESULTS_STATUS,
        "conway_table_sha256": _conway_digest(conway_studies),
        "target_data_revision": target_data_revision,
        "n_boot": config.n_boot,
        "seeds": {
            "primary": config.seed,
            "independent_repeats": repeat_seeds,
        },
        "agreement": {
            "clustering": config.agreement_clustering,
            "cluster_column": "study_base"
            if config.agreement_clustering == "publication"
            else "study",
            "bootstrap_mode": config.bootstrap_mode,
            "parameter_mapping": config.analysis.parameter_mapping,
            "sensitivity": _sensitivity_name(config),
        },
        "target": {
            "resampling_unit": "patient_cluster",
            "resampling_policy": TARGET_RESAMPLING_POLICY,
            "subgroup_input_mode": subgroup_input_mode,
            "measurement_policy": config.analysis.measurement_policy,
            "support": config.analysis.support,
            "required_columns": {
                "patient_id": columns.patient_id,
                "encounter_id": columns.encounter_id,
                "encounter_order": columns.encounter_order,
                "measurement_order": columns.measurement_order,
                "paco2": "paco2",
                "subgroup_input": (
                    ["subgroup"]
                    if subgroup_input_mode == "prepared_subgroup"
                    else ["is_amb", "is_emer", "is_inp", "cc_time"]
                ),
            },
            "index_rule": (
                "earliest eligible encounter then earliest eligible PaCO2 value within each "
                "setting; pooled All selects the earliest eligible record overall"
            ),
            "degenerate_redraw": {
                "maximum_attempts_per_replicate": MAX_TARGET_RESAMPLE_ATTEMPTS,
                "maximum_rejected_proposal_fraction_per_setting": (
                    MAX_DEGENERATE_PROPOSAL_FRACTION
                ),
                "primary_rejected_proposal_fraction": _resampling_fractions(primary),
                "repeat_rejected_proposal_fraction": (
                    _resampling_fractions(repeat) if repeat is not None else None
                ),
            },
        },
        "outputs": {
            "true_threshold": config.analysis.true_threshold,
            "two_stage_bounds": [
                config.analysis.two_stage_lower,
                config.analysis.two_stage_upper,
            ],
            "tcco2_values": list(config.analysis.tcco2_values),
            "primary_prediction": "prior_weighted",
            "prediction_comparator": "likelihood_only",
            "interval_method": "percentile_2.5_50_97.5",
        },
        "stability": {
            "assessed": config.assess_stability,
            "required": config.require_stability,
            "passed": stability_passed,
            "hard_gate": "combined batch-quantile MCSE <= one tenth reporting precision",
            "descriptive_check": "independent runs within 2 combined MCSE",
        },
        "contract_compliance": _contract_compliance(config),
    }


def _validate_target_data_revision(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("target_data_revision must be a nonblank non-sensitive label.")
    return value.strip()


def _resampling_fractions(draws: _JointDraws) -> dict[str, float]:
    return {
        group: rejected / proposals
        for group, (rejected, proposals) in sorted(draws.target_resampling.items())
    }


def _sensitivity_deviations(config: DownstreamWorkflowConfig) -> list[str]:
    deviations: list[str] = []
    if config.agreement_clustering != "publication":
        deviations.append("effect_row_clustering")
    if config.analysis.parameter_mapping != "setting_specific":
        deviations.append("pooled_main_parameters")
    if config.analysis.measurement_policy != "index":
        deviations.append("all_measurements")
    if config.analysis.support != "all":
        deviations.append("central_95_support")
    return deviations


def _sensitivity_name(config: DownstreamWorkflowConfig) -> str:
    deviations = _sensitivity_deviations(config)
    return deviations[0] if deviations else "primary"


def _contract_compliance(config: DownstreamWorkflowConfig) -> dict[str, object]:
    canonical = DownstreamWorkflowConfig()
    canonical_analysis = canonical.analysis
    reasons: list[str] = []
    if config.n_boot < MINIMUM_DOWNSTREAM_DRAWS:
        reasons.append("fewer_than_10000_draws")
    if not config.enforce_minimum_draws:
        reasons.append("minimum_draw_enforcement_disabled")
    if not config.assess_stability:
        reasons.append("stability_assessment_disabled")
    if not config.require_stability:
        reasons.append("stability_gate_not_required")
    if config.bootstrap_mode != "cluster_plus_withinstudy":
        reasons.append("noncanonical_bootstrap_mode")
    if config.analysis.true_threshold != canonical_analysis.true_threshold:
        reasons.append("noncanonical_true_threshold")
    if (
        config.analysis.two_stage_lower,
        config.analysis.two_stage_upper,
    ) != (
        canonical_analysis.two_stage_lower,
        canonical_analysis.two_stage_upper,
    ):
        reasons.append("noncanonical_two_stage_boundaries")
    if config.analysis.tcco2_values != canonical_analysis.tcco2_values:
        reasons.append("noncanonical_prediction_grid")
    if config.seed != canonical.seed:
        reasons.append("noncanonical_primary_seed")
    if config.repeat_seed != canonical.repeat_seed:
        reasons.append("noncanonical_repeat_seed")
    return {"compliant": not reasons, "reasons": reasons}


def _conway_digest(data: pd.DataFrame) -> str:
    canonical = data.copy().reindex(sorted(data.columns), axis=1)
    if "study_id" in canonical.columns:
        canonical = canonical.sort_values("study_id", kind="stable")
    payload = canonical.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

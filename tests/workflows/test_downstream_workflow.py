from __future__ import annotations

import inspect
import json

import pandas as pd
import pytest

from tcco2_accuracy.core.downstream import DownstreamAnalysisConfig, PatientInputColumns
from tcco2_accuracy.workflows.downstream import (
    MINIMUM_DOWNSTREAM_DRAWS,
    DownstreamWorkflowConfig,
    MonteCarloStabilityError,
    _contract_compliance,
    run_downstream_analysis,
)


def test_downstream_workflow_is_in_memory_and_uses_publication_clustering() -> None:
    result = run_downstream_analysis(
        _synthetic_patient_data(),
        _synthetic_conway_studies(),
        target_data_revision="synthetic-v1",
        config=DownstreamWorkflowConfig(
            n_boot=25,
            enforce_minimum_draws=False,
            assess_stability=False,
            require_stability=False,
        ),
    )

    assert set(inspect.signature(run_downstream_analysis).parameters) == {
        "patient_data",
        "conway_studies",
        "target_data_revision",
        "config",
        "columns",
    }
    assert result.manifest["agreement"] == {
        "clustering": "publication",
        "cluster_column": "study_base",
        "bootstrap_mode": "cluster_plus_withinstudy",
        "parameter_mapping": "setting_specific",
        "sensitivity": "primary",
    }
    assert result.manifest["target"]["resampling_unit"] == "patient_cluster"
    assert (
        result.manifest["target"]["resampling_policy"]
        == "ordinary_patient_cluster_with_bounded_degenerate_redraw"
    )
    assert result.manifest["stability"]["passed"] is None
    assert result.manifest["target_data_revision"] == "synthetic-v1"
    assert result.manifest["contract_compliance"]["compliant"] is False
    json.dumps(result.manifest)
    assert list(result.core.columns) == [
        "requested_group",
        "parameter_group_used",
        "true_threshold",
        "metric",
        "bootstrap_q025",
        "bootstrap_q500",
        "bootstrap_q975",
    ]
    assert list(result.prediction.columns) == [
        "requested_group",
        "parameter_group_used",
        "true_threshold",
        "tcco2",
        "mode",
        "metric",
        "bootstrap_q025",
        "bootstrap_q500",
        "bootstrap_q975",
    ]
    assert list(result.two_stage.columns) == [
        "requested_group",
        "parameter_group_used",
        "true_threshold",
        "zone_lower",
        "zone_upper",
        "metric",
        "bootstrap_q025",
        "bootstrap_q500",
        "bootstrap_q975",
    ]
    probability_rows = result.core.loc[~result.core["metric"].str.startswith("lr_")]
    assert (
        probability_rows[["bootstrap_q025", "bootstrap_q500", "bootstrap_q975"]].ge(0).all().all()
    )
    assert (
        probability_rows[["bootstrap_q025", "bootstrap_q500", "bootstrap_q975"]].le(1).all().all()
    )
    assert set(result.core["parameter_group_used"]) == {"main", "icu", "arf", "lft"}
    for frame in (result.core, result.prediction, result.two_stage):
        _assert_aggregate_only(frame)


def test_downstream_workflow_exposes_effect_row_sensitivity_only_when_requested() -> None:
    result = run_downstream_analysis(
        _synthetic_patient_data(),
        _synthetic_conway_studies(),
        target_data_revision="synthetic-v1",
        config=DownstreamWorkflowConfig(
            n_boot=25,
            agreement_clustering="effect_row",
            enforce_minimum_draws=False,
            assess_stability=False,
            require_stability=False,
        ),
    )

    assert result.manifest["agreement"]["clustering"] == "effect_row"
    assert result.manifest["agreement"]["cluster_column"] == "study"


def test_downstream_workflow_enforces_minimum_draws_by_default() -> None:
    with pytest.raises(ValueError, match=str(MINIMUM_DOWNSTREAM_DRAWS)):
        DownstreamWorkflowConfig(n_boot=MINIMUM_DOWNSTREAM_DRAWS - 1)


def test_downstream_workflow_uses_one_predetermined_independent_repeat() -> None:
    result = run_downstream_analysis(
        _synthetic_patient_data(),
        _synthetic_conway_studies(),
        target_data_revision="synthetic-v1",
        config=DownstreamWorkflowConfig(
            n_boot=25,
            seed=101,
            repeat_seed=202,
            enforce_minimum_draws=False,
            require_stability=False,
        ),
    )

    assert result.manifest["seeds"]["independent_repeats"] == [202]
    assert set(result.stability["repeat_seed"]) == {202}
    assert {"combined_mcse", "within_2_mcse", "mcse_passed"}.issubset(result.stability.columns)
    assert set(result.stability.columns) == {
        "analysis",
        "requested_group",
        "parameter_group_used",
        "metric",
        "repeat_seed",
        "component",
        "primary",
        "repeat",
        "primary_mcse",
        "repeat_mcse",
        "combined_mcse",
        "difference",
        "reporting_precision",
        "within_2_mcse",
        "mcse_passed",
        "tcco2",
        "mode",
    }


def test_downstream_workflow_exposes_complete_stability_failure_diagnostics() -> None:
    with pytest.raises(MonteCarloStabilityError) as exc_info:
        run_downstream_analysis(
            _synthetic_patient_data(),
            _synthetic_conway_studies(),
            target_data_revision="synthetic-stability-failure-v1",
            config=DownstreamWorkflowConfig(
                n_boot=25,
                enforce_minimum_draws=False,
            ),
        )

    stability = exc_info.value.stability
    assert not stability.empty
    assert (~stability["mcse_passed"]).any()
    assert set(stability["repeat_seed"]) == {202402}


@pytest.mark.parametrize(("field", "value"), [("seed", True), ("repeat_seed", 2.5)])
def test_downstream_workflow_requires_integer_seeds(field: str, value: object) -> None:
    kwargs = {field: value}
    with pytest.raises(ValueError, match=f"{field} must be an integer"):
        DownstreamWorkflowConfig(**kwargs)  # type: ignore[arg-type]


def test_downstream_workflow_requires_distinct_stability_seeds() -> None:
    with pytest.raises(ValueError, match="must be distinct"):
        DownstreamWorkflowConfig(seed=1, repeat_seed=1)


def test_downstream_workflow_rejects_factorial_sensitivities() -> None:
    with pytest.raises(ValueError, match="one at a time"):
        DownstreamWorkflowConfig(
            agreement_clustering="effect_row",
            analysis=DownstreamAnalysisConfig(parameter_mapping="pooled_main"),
        )


def test_canonical_configuration_is_contract_compliant() -> None:
    assert _contract_compliance(DownstreamWorkflowConfig()) == {
        "compliant": True,
        "reasons": [],
    }


@pytest.mark.parametrize(
    ("config", "reason"),
    [
        (
            DownstreamWorkflowConfig(analysis=DownstreamAnalysisConfig(true_threshold=46.0)),
            "noncanonical_true_threshold",
        ),
        (
            DownstreamWorkflowConfig(analysis=DownstreamAnalysisConfig(two_stage_lower=39.0)),
            "noncanonical_two_stage_boundaries",
        ),
        (
            DownstreamWorkflowConfig(
                analysis=DownstreamAnalysisConfig(tcco2_values=(35.0, 45.0, 55.0))
            ),
            "noncanonical_prediction_grid",
        ),
        (DownstreamWorkflowConfig(seed=202403), "noncanonical_primary_seed"),
        (DownstreamWorkflowConfig(repeat_seed=202403), "noncanonical_repeat_seed"),
    ],
)
def test_contract_compliance_rejects_noncanonical_scientific_settings(
    config: DownstreamWorkflowConfig, reason: str
) -> None:
    assert _contract_compliance(config) == {"compliant": False, "reasons": [reason]}


@pytest.mark.parametrize(
    "config",
    [
        DownstreamWorkflowConfig(n_boot=MINIMUM_DOWNSTREAM_DRAWS + 1),
        DownstreamWorkflowConfig(agreement_clustering="effect_row"),
        DownstreamWorkflowConfig(
            analysis=DownstreamAnalysisConfig(parameter_mapping="pooled_main")
        ),
        DownstreamWorkflowConfig(
            analysis=DownstreamAnalysisConfig(measurement_policy="all_measurements")
        ),
        DownstreamWorkflowConfig(analysis=DownstreamAnalysisConfig(support="central_95")),
    ],
)
def test_contract_compliance_accepts_prespecified_configurations(
    config: DownstreamWorkflowConfig,
) -> None:
    assert _contract_compliance(config) == {"compliant": True, "reasons": []}


@pytest.mark.parametrize("target_data_revision", ["", "   "])
def test_downstream_workflow_requires_target_data_revision(
    target_data_revision: str,
) -> None:
    with pytest.raises(ValueError, match="target_data_revision must be a nonblank"):
        run_downstream_analysis(
            _synthetic_patient_data(),
            _synthetic_conway_studies(),
            target_data_revision=target_data_revision,
            config=_development_config(),
        )


def test_downstream_workflow_accepts_custom_columns_and_datetime_ordering() -> None:
    data = _synthetic_patient_data().rename(
        columns={
            "patient_id": "pid",
            "encounter_id": "eid",
            "encounter_order": "encounter_time",
            "measurement_order": "measurement_time",
        }
    )
    data["encounter_time"] = "2026-01-01T00:00:00Z"
    data["measurement_time"] = "2026-01-01T00:05:00Z"

    result = run_downstream_analysis(
        data,
        _synthetic_conway_studies(),
        target_data_revision="synthetic-custom-columns-v1",
        config=_development_config(),
        columns=PatientInputColumns(
            patient_id="pid",
            encounter_id="eid",
            encounter_order="encounter_time",
            measurement_order="measurement_time",
        ),
    )

    assert result.manifest["target"]["required_columns"]["patient_id"] == "pid"


def test_downstream_workflow_accepts_raw_subgroup_flags() -> None:
    data = _synthetic_patient_data()
    subgroup = data.pop("subgroup")
    data["is_amb"] = (subgroup == "pft").astype(int)
    data["is_emer"] = (subgroup == "ed_inp").astype(int)
    data["is_inp"] = subgroup.isin(["ed_inp", "icu"]).astype(int)
    data["cc_time"] = (subgroup == "icu").astype(int)

    result = run_downstream_analysis(
        data,
        _synthetic_conway_studies(),
        target_data_revision="synthetic-raw-flags-v1",
        config=_development_config(),
    )

    assert result.manifest["target"]["subgroup_input_mode"] == "raw_flags"
    assert result.manifest["target"]["required_columns"]["subgroup_input"] == [
        "is_amb",
        "is_emer",
        "is_inp",
        "cc_time",
    ]


def test_downstream_workflow_fails_closed_for_bad_source_schema() -> None:
    data = _synthetic_patient_data().drop(columns="patient_id")
    with pytest.raises(ValueError, match="Missing required patient-level columns"):
        run_downstream_analysis(
            data,
            _synthetic_conway_studies(),
            target_data_revision="synthetic-v1",
            config=DownstreamWorkflowConfig(
                n_boot=2,
                enforce_minimum_draws=False,
                assess_stability=False,
                require_stability=False,
            ),
            columns=PatientInputColumns(),
        )


def _synthetic_patient_data() -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for group, (low, high) in {
        "pft": (35.0, 52.0),
        "ed_inp": (37.0, 55.0),
        "icu": (39.0, 58.0),
    }.items():
        for number in range(20):
            patient_id = f"synthetic-{group}-{number:02d}"
            rows.append(
                {
                    "patient_id": patient_id,
                    "encounter_id": f"enc-{patient_id}",
                    "encounter_order": 1,
                    "measurement_order": 1,
                    "paco2": low if number % 2 == 0 else high,
                    "subgroup": group,
                }
            )
    return pd.DataFrame(rows)


def _development_config() -> DownstreamWorkflowConfig:
    return DownstreamWorkflowConfig(
        n_boot=25,
        enforce_minimum_draws=False,
        assess_stability=False,
        require_stability=False,
    )


def _synthetic_conway_studies() -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    definitions = [
        ("Shared 2001 (a)", -0.3, 1, 0, 0),
        ("Shared 2001 (b)", -0.1, 1, 0, 0),
        ("ICU 2002", -0.5, 1, 0, 0),
        ("ARF 2003", 0.6, 0, 1, 0),
        ("LFT 2004", -0.2, 0, 0, 1),
        ("All 2005", 0.1, 0, 0, 0),
    ]
    for study_id, bias, is_icu, is_arf, is_lft in definitions:
        rows.append(
            {
                "study_id": study_id,
                "bias": bias,
                "s2": 4.0,
                "n_pairs": 30,
                "n_participants": 30,
                "c": 1.0,
                "is_icu": is_icu,
                "is_arf": is_arf,
                "is_lft": is_lft,
            }
        )
    return pd.DataFrame(rows)


def _assert_aggregate_only(frame: pd.DataFrame) -> None:
    forbidden = {
        "patient_id",
        "encounter_id",
        "encounter_order",
        "measurement_order",
        "count",
        "weight",
    }
    assert not forbidden.intersection(frame.columns)
    assert not any(
        column.startswith("n_")
        or "per_1000" in column
        or column.endswith("_bin")
        or "_bin_" in column
        for column in frame.columns
    )
    assert "synthetic-pft-00" not in frame.to_csv(index=False)

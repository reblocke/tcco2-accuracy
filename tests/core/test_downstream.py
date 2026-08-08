from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from tcco2_accuracy.core import downstream
from tcco2_accuracy.core.downstream import (
    DownstreamAnalysisConfig,
    PatientInputColumns,
    _monte_carlo_stability,
    _order_values,
    _prepare_target_groups,
    _resample_patient_clusters,
    _run_draw_aligned_downstream,
    _summarize_joint_draws,
    _TargetPopulation,
)


def test_downstream_public_surface_exposes_no_raw_draw_runner() -> None:
    assert downstream.__all__ == ["DownstreamAnalysisConfig", "PatientInputColumns"]
    assert not hasattr(downstream, "run_draw_aligned_downstream")
    assert not hasattr(downstream, "JointDrawResult")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"patient_id": ""},
        {"patient_id": "id", "encounter_id": "id"},
        {"measurement_order": "paco2"},
    ],
)
def test_patient_input_column_roles_are_nonblank_distinct_and_unreserved(
    kwargs: dict[str, str],
) -> None:
    with pytest.raises(ValueError, match="column roles"):
        PatientInputColumns(**kwargs)


def test_joint_downstream_summaries_are_deterministic_and_aggregate_only() -> None:
    patient_data = _synthetic_patient_data()
    params = _synthetic_params(n_boot=25)

    first = _summaries(patient_data, params, seed=123)
    second = _summaries(patient_data, params, seed=123)

    for first_frame, second_frame in zip(first, second, strict=True):
        pdt.assert_frame_equal(first_frame, second_frame)
        _assert_aggregate_only(first_frame)
    core, prediction, two_stage = first
    assert set(core["requested_group"]) == {"pft", "ed_inp", "icu", "all"}
    assert {
        "prevalence",
        "tp_probability",
        "fp_probability",
        "tn_probability",
        "fn_probability",
        "misclassification_probability",
    }.issubset(set(core["metric"]))
    assert set(prediction["mode"]) == {"likelihood_only", "prior_weighted"}
    assert set(prediction["tcco2"]) == {35.0, 40.0, 45.0, 50.0, 55.0}
    assert {"bootstrap_q025", "bootstrap_q500", "bootstrap_q975"}.issubset(two_stage.columns)


def test_index_policy_ignores_later_measurements_but_all_measurements_uses_them() -> None:
    patient_data = _synthetic_patient_data()
    changed_later = patient_data.copy()
    changed_later.loc[changed_later["measurement_order"] > 1, "paco2"] = 150.0
    params = _synthetic_params(n_boot=25)

    index_config = DownstreamAnalysisConfig(measurement_policy="index")
    index_original = _summaries(patient_data, params, config=index_config, seed=321)
    index_changed = _summaries(changed_later, params, config=index_config, seed=321)
    for original, changed in zip(index_original, index_changed, strict=True):
        pdt.assert_frame_equal(original, changed)

    all_config = DownstreamAnalysisConfig(measurement_policy="all_measurements")
    all_original = _summaries(patient_data, params, config=all_config, seed=321)
    all_changed = _summaries(changed_later, params, config=all_config, seed=321)
    assert not all_original[0].equals(all_changed[0])


@pytest.mark.parametrize("measurement_policy", ["index", "all_measurements"])
def test_seeded_patient_resampling_is_row_order_invariant(measurement_policy: str) -> None:
    patient_data = _synthetic_patient_data()
    params = _synthetic_params(n_boot=25)
    config = DownstreamAnalysisConfig(measurement_policy=measurement_policy)

    original = _run(patient_data, params, config=config, seed=321)
    reversed_rows = _run(
        patient_data.iloc[::-1].reset_index(drop=True),
        params,
        config=config,
        seed=321,
    )

    for name in ("core", "prediction", "two_stage"):
        original_values = getattr(original, name)
        reversed_values = getattr(reversed_rows, name)
        assert original_values.keys() == reversed_values.keys()
        for key in original_values:
            np.testing.assert_array_equal(original_values[key], reversed_values[key])
    assert original.target_resampling == reversed_rows.target_resampling


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("patient_id", "", "patient_id.*nonblank"),
        ("encounter_id", None, "encounter_id.*nonblank"),
        ("encounter_order", None, "encounter_order.*numeric or valid datetime"),
        ("measurement_order", None, "measurement_order.*numeric or valid datetime"),
    ],
)
def test_joint_downstream_rejects_missing_patient_schema(
    column: str, value: object, message: str
) -> None:
    patient_data = _synthetic_patient_data()
    patient_data.loc[0, column] = value

    with pytest.raises(ValueError, match=message):
        _run(patient_data, _synthetic_params(n_boot=2), seed=1)


def test_joint_downstream_rejects_mixed_ordering_types() -> None:
    patient_data = _synthetic_patient_data()
    patient_data["encounter_order"] = patient_data["encounter_order"].astype(object)
    patient_data.loc[0, "encounter_order"] = "2020-01-01"

    with pytest.raises(ValueError, match="only finite numeric values or only valid datetimes"):
        _run(patient_data, _synthetic_params(n_boot=2), seed=1)


def test_joint_downstream_rejects_duplicate_measurement_key() -> None:
    patient_data = _synthetic_patient_data()
    duplicate = patient_data.iloc[[0]].copy()
    duplicate["paco2"] = 60.0
    patient_data = pd.concat([patient_data, duplicate], ignore_index=True)

    with pytest.raises(ValueError, match="unique patient/encounter/measurement-order keys"):
        _run(patient_data, _synthetic_params(n_boot=2), seed=1)


@pytest.mark.parametrize(
    ("original_order", "duplicate_order", "use_datetimes"),
    [
        (1, "01", False),
        ("2026-01-01T00:00:00Z", "2025-12-31T19:00:00-05:00", True),
    ],
)
def test_all_measurements_rejects_equivalent_normalized_measurement_orders(
    original_order: int | str, duplicate_order: str, use_datetimes: bool
) -> None:
    patient_data = _synthetic_patient_data()
    if use_datetimes:
        patient_data["measurement_order"] = (
            pd.Timestamp("2026-01-01T00:00:00Z")
            + pd.to_timedelta(patient_data["measurement_order"], unit="h")
        ).astype(str)
    patient_data["measurement_order"] = patient_data["measurement_order"].astype(object)
    patient_data.loc[0, "measurement_order"] = original_order
    duplicate = patient_data.iloc[[0]].copy()
    duplicate["measurement_order"] = duplicate_order
    duplicate["paco2"] = 60.0
    patient_data = pd.concat([patient_data, duplicate], ignore_index=True)

    with pytest.raises(ValueError, match="unique patient/encounter/measurement-order keys"):
        _run(
            patient_data,
            _synthetic_params(n_boot=2),
            config=DownstreamAnalysisConfig(measurement_policy="all_measurements"),
            seed=1,
        )


def test_order_normalization_preserves_adjacent_large_integers() -> None:
    orders = pd.Series([2**53, 2**53 + 1], dtype="int64")

    normalized = _order_values(orders, "measurement_order")

    assert normalized.tolist() == [2**53, 2**53 + 1]
    assert normalized.nunique() == 2


def test_joint_downstream_rejects_ambiguous_earliest_encounter_order() -> None:
    patient_data = _synthetic_patient_data()
    duplicate = patient_data.iloc[[0]].copy()
    duplicate["encounter_id"] = "different-earliest-encounter"
    patient_data = pd.concat([patient_data, duplicate], ignore_index=True)

    with pytest.raises(ValueError, match="Encounter order does not uniquely identify"):
        _run(patient_data, _synthetic_params(n_boot=2), seed=1)


def test_joint_downstream_requires_each_group_to_cover_both_truth_classes() -> None:
    patient_data = _synthetic_patient_data()
    patient_data.loc[patient_data["subgroup"] == "pft", "paco2"] = 40.0

    with pytest.raises(ValueError, match="both below and at/above"):
        _run(patient_data, _synthetic_params(n_boot=2), seed=1)


def test_ordinary_patient_bootstrap_propagates_prevalence_uncertainty() -> None:
    patient_data = _synthetic_patient_data()
    for group in ("pft", "ed_inp", "icu"):
        group_index = patient_data.loc[
            (patient_data["subgroup"] == group) & (patient_data["measurement_order"] == 1)
        ].index
        patient_data.loc[group_index[:14], "paco2"] = 40.0
        patient_data.loc[group_index[14:], "paco2"] = 50.0

    draws = _run(patient_data, _synthetic_params(n_boot=100), seed=3)

    for group, parameter_group in (("pft", "lft"), ("ed_inp", "arf"), ("icu", "icu")):
        prevalence = draws.core[(group, parameter_group, "prevalence")]
        assert prevalence.max() > prevalence.min()
    assert draws.target_resampling


def test_sparse_truth_classes_fail_redraw_fraction_gate() -> None:
    with pytest.raises(ValueError, match="exceeded the allowed 1%"):
        _run(_two_patient_data_per_group(), _synthetic_params(n_boot=20), seed=0)


def test_degenerate_patient_sample_is_redrawn_within_attempt_limit() -> None:
    population = _TargetPopulation(clusters=(np.array([40.0]), np.array([50.0])))

    class _StubRng:
        def __init__(self) -> None:
            self.calls = 0

        def integers(self, *_args: object, **_kwargs: object):
            self.calls += 1
            return np.array([0, 0] if self.calls == 1 else [0, 1])

    values, rejected, proposals = _resample_patient_clusters(
        population,
        rng=_StubRng(),  # type: ignore[arg-type]
        threshold=45.0,
        group="pft",
        replicate=0,
    )

    assert values.tolist() == [40.0, 50.0]
    assert (rejected, proposals) == (1, 2)


def test_index_selection_is_within_setting_and_pooled_all_is_global() -> None:
    patient_data = _synthetic_patient_data()
    pft_patient = "synthetic-pft-00"
    icu_patient = "synthetic-icu-00"
    patient_data.loc[patient_data["patient_id"] == icu_patient, "patient_id"] = pft_patient
    icu_rows = (patient_data["subgroup"] == "icu") & (patient_data["patient_id"] == pft_patient)
    patient_data.loc[icu_rows, "encounter_order"] += 2

    populations = _prepare_target_groups(
        patient_data,
        columns=PatientInputColumns(),
        config=DownstreamAnalysisConfig(),
    )

    assert _population_cluster_count(populations["pft"]) == 20
    assert _population_cluster_count(populations["icu"]) == 20
    assert _population_cluster_count(populations["all"]) == 59


def test_patient_and_encounter_identifiers_are_stripped_before_grouping() -> None:
    patient_data = _synthetic_patient_data()
    later_rows = (patient_data["patient_id"] == "synthetic-pft-00") & (
        patient_data["measurement_order"] > 1
    )
    patient_data.loc[later_rows, "patient_id"] = " synthetic-pft-00 "
    patient_data.loc[later_rows, "encounter_id"] = (
        " " + patient_data.loc[later_rows, "encounter_id"].astype(str) + " "
    )

    populations = _prepare_target_groups(
        patient_data,
        columns=PatientInputColumns(),
        config=DownstreamAnalysisConfig(),
    )

    assert _population_cluster_count(populations["pft"]) == 20


@pytest.mark.parametrize("variance", [(0.0, 0.0), (1e308, 1e308)])
def test_joint_downstream_rejects_nonpositive_or_overflowing_total_variance(
    variance: tuple[float, float],
) -> None:
    params = _synthetic_params(n_boot=2)
    params["sigma2"] = variance[0]
    params["tau2"] = variance[1]

    with pytest.raises(ValueError, match="Total downstream variance"):
        _run(_synthetic_patient_data(), params, seed=1)


def test_joint_downstream_support_and_mapping_sensitivities_are_explicit() -> None:
    patient_data = _synthetic_patient_data()
    patient_data.loc[0, "paco2"] = 500.0
    params = _synthetic_params(n_boot=25)

    primary = _summaries(patient_data, params, seed=777)
    central = _summaries(
        patient_data,
        params,
        config=DownstreamAnalysisConfig(support="central_95"),
        seed=777,
    )
    pooled = _summaries(
        patient_data,
        params,
        config=DownstreamAnalysisConfig(parameter_mapping="pooled_main"),
        seed=777,
    )

    assert not primary[0].equals(central[0])
    assert set(pooled[0]["parameter_group_used"]) == {"main"}
    assert set(primary[0].loc[primary[0]["requested_group"] == "pft", "parameter_group_used"]) == {
        "lft"
    }


def test_central_support_uses_the_pooled_all_distribution_for_all() -> None:
    data = _support_sensitive_patient_data()
    populations = _prepare_target_groups(
        data,
        columns=PatientInputColumns(),
        config=DownstreamAnalysisConfig(support="central_95"),
    )

    pft_cluster_count = _population_cluster_count(populations["pft"])
    all_cluster_count = _population_cluster_count(populations["all"])
    assert pft_cluster_count == 2
    assert all_cluster_count == 80


def test_joint_downstream_requires_complete_aligned_parameter_groups() -> None:
    params = _synthetic_params(n_boot=2)
    missing_group = params.loc[params["group"] != "lft"].copy()

    with pytest.raises(ValueError, match="No parameter draws"):
        _run(_synthetic_patient_data(), missing_group, seed=1)


def test_joint_downstream_stability_is_aggregate_only() -> None:
    primary = _run(_synthetic_patient_data(), _synthetic_params(n_boot=25), seed=123)
    repeat = _run(_synthetic_patient_data(), _synthetic_params(n_boot=25), seed=456)
    stability = _monte_carlo_stability(primary, repeat, repeat_seed=456)

    _assert_aggregate_only(stability)
    assert set(stability["component"]) == {
        "bootstrap_q025",
        "bootstrap_q500",
        "bootstrap_q975",
    }
    assert set(stability["repeat_seed"]) == {456}
    assert (stability["combined_mcse"] >= 0).all()
    assert stability["within_2_mcse"].dtype == bool
    assert stability["mcse_passed"].dtype == bool


def test_single_cluster_end_to_end_matches_hand_calculation() -> None:
    config = DownstreamAnalysisConfig(
        measurement_policy="all_measurements",
        tcco2_values=(45.0,),
    )
    core, prediction, two_stage = _summaries(
        _single_mixed_cluster_per_group(),
        _synthetic_params(n_boot=1).assign(delta=0.0, sigma2=1.0, tau2=0.0),
        config=config,
        seed=11,
    )
    pft_core = core.loc[core["requested_group"] == "pft"].set_index("metric")
    tail = 1.0 - NormalDist().cdf(1.0)

    assert pft_core.loc["prevalence", "bootstrap_q500"] == pytest.approx(0.5)
    assert pft_core.loc["sensitivity", "bootstrap_q500"] == pytest.approx(1.0 - tail)
    assert pft_core.loc["specificity", "bootstrap_q500"] == pytest.approx(1.0 - tail)
    assert pft_core.loc["tp_probability", "bootstrap_q500"] == pytest.approx((1.0 - tail) / 2)
    confusion = pft_core.loc[
        ["tp_probability", "fp_probability", "tn_probability", "fn_probability"],
        "bootstrap_q500",
    ].sum()
    assert confusion == pytest.approx(1.0)

    prior_probability = prediction.loc[
        (prediction["requested_group"] == "pft")
        & (prediction["mode"] == "prior_weighted")
        & (prediction["metric"] == "paco2_ge_threshold_probability"),
        "bootstrap_q500",
    ].item()
    likelihood_median = prediction.loc[
        (prediction["requested_group"] == "pft")
        & (prediction["mode"] == "likelihood_only")
        & (prediction["metric"] == "paco2_pi_median"),
        "bootstrap_q500",
    ].item()
    assert prior_probability == pytest.approx(0.5)
    assert likelihood_median == pytest.approx(45.0)

    pft_two_stage = two_stage.loc[two_stage["requested_group"] == "pft"].set_index("metric")
    zone_mass = pft_two_stage.loc[
        ["zone1_probability", "zone2_probability", "zone3_probability"],
        "bootstrap_q500",
    ].sum()
    assert zone_mass == pytest.approx(1.0)


def _run(
    patient_data: pd.DataFrame,
    params: pd.DataFrame,
    *,
    config: DownstreamAnalysisConfig = DownstreamAnalysisConfig(),
    seed: int,
):
    return _run_draw_aligned_downstream(
        patient_data,
        params,
        config=config,
        columns=PatientInputColumns(),
        seed=seed,
    )


def _summaries(
    patient_data: pd.DataFrame,
    params: pd.DataFrame,
    *,
    config: DownstreamAnalysisConfig = DownstreamAnalysisConfig(),
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return _summarize_joint_draws(
        _run(patient_data, params, config=config, seed=seed),
        config=config,
    )


def _synthetic_patient_data() -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    values_by_group = {
        "pft": (36.0, 52.0),
        "ed_inp": (38.0, 56.0),
        "icu": (40.0, 60.0),
    }
    for group, (low, high) in values_by_group.items():
        for number in range(20):
            patient_id = f"synthetic-{group}-{number:02d}"
            index_value = low if number % 2 == 0 else high
            rows.extend(
                [
                    {
                        "patient_id": patient_id,
                        "encounter_id": f"{patient_id}-first",
                        "encounter_order": 1,
                        "measurement_order": 1,
                        "paco2": index_value,
                        "subgroup": group,
                    },
                    {
                        "patient_id": patient_id,
                        "encounter_id": f"{patient_id}-first",
                        "encounter_order": 1,
                        "measurement_order": 2,
                        "paco2": index_value + 1,
                        "subgroup": group,
                    },
                    {
                        "patient_id": patient_id,
                        "encounter_id": f"{patient_id}-later",
                        "encounter_order": 2,
                        "measurement_order": 3,
                        "paco2": index_value + 10,
                        "subgroup": group,
                    },
                ]
            )
    return pd.DataFrame(rows)


def _two_patient_data_per_group() -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for group in ("pft", "ed_inp", "icu"):
        for number, value in enumerate((40.0, 50.0)):
            patient_id = f"synthetic-small-{group}-{number}"
            rows.append(
                {
                    "patient_id": patient_id,
                    "encounter_id": f"enc-{patient_id}",
                    "encounter_order": 1,
                    "measurement_order": 1,
                    "paco2": value,
                    "subgroup": group,
                }
            )
    return pd.DataFrame(rows)


def _single_mixed_cluster_per_group() -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for group in ("pft", "ed_inp", "icu"):
        for order, value in enumerate((44.0, 46.0), start=1):
            rows.append(
                {
                    "patient_id": f"synthetic-mixed-{group}",
                    "encounter_id": f"enc-synthetic-mixed-{group}",
                    "encounter_order": 1,
                    "measurement_order": order,
                    "paco2": value,
                    "subgroup": group,
                }
            )
    return pd.DataFrame(rows)


def _support_sensitive_patient_data() -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for number, value in enumerate((1.0, 40.0, 50.0, 100.0)):
        rows.append(
            {
                "patient_id": f"synthetic-pft-{number}",
                "encounter_id": f"enc-pft-{number}",
                "encounter_order": 1,
                "measurement_order": 1,
                "paco2": value,
                "subgroup": "pft",
            }
        )
    for group in ("ed_inp", "icu"):
        for number in range(40):
            rows.append(
                {
                    "patient_id": f"synthetic-{group}-{number}",
                    "encounter_id": f"enc-{group}-{number}",
                    "encounter_order": 1,
                    "measurement_order": 1,
                    "paco2": 44.0 if number % 2 == 0 else 46.0,
                    "subgroup": group,
                }
            )
    return pd.DataFrame(rows)


def _synthetic_params(n_boot: int) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    offsets = {"main": 0.0, "icu": -0.4, "arf": 0.7, "lft": -0.2}
    for group, offset in offsets.items():
        for replicate in range(n_boot):
            rows.append(
                {
                    "group": group,
                    "replicate": replicate,
                    "delta": offset + replicate * 0.01,
                    "sigma2": 4.0,
                    "tau2": 0.5,
                }
            )
    return pd.DataFrame(rows)


def _population_cluster_count(population: object) -> int:
    return len(getattr(population, "clusters"))


def _assert_aggregate_only(frame: pd.DataFrame) -> None:
    forbidden = {
        "patient_id",
        "encounter_id",
        "encounter_order",
        "measurement_order",
        "count",
        "weight",
        "replicate",
        "value",
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

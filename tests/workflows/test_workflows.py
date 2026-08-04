from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from tcco2_accuracy.core.conway_meta import (
    AGREEMENT_METHOD_VERSION,
    RESULTS_STATUS,
    conway_group_summary,
)
from tcco2_accuracy.data import CONWAY_DATA_PATH, INSILICO_PACO2_PATH, load_conway_group
from tcco2_accuracy.workflows import bootstrap, conditional, infer, meta, paco2, sim

PUBLISHED_FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "conway_table1.csv"


def test_workflows_deterministic(tmp_path: Path) -> None:
    seed = 123
    n_boot = 25
    conway_path, conway_groups = _resolve_conway_sources()
    paco2_path, paco2_source = _resolve_paco2_sources()

    out_dir1 = tmp_path / "run1"
    out_dir2 = tmp_path / "run2"

    meta_result1 = meta.run_meta_checks(
        conway_path=conway_path,
        data_by_group=conway_groups,
        published_comparator_path=PUBLISHED_FIXTURE_PATH,
        out_dir=out_dir1,
    )
    meta_result2 = meta.run_meta_checks(
        conway_path=conway_path,
        data_by_group=conway_groups,
        published_comparator_path=PUBLISHED_FIXTURE_PATH,
        out_dir=out_dir2,
    )
    assert (out_dir1 / "meta_loa_check.md").exists()
    assert "Corrected versus published/legacy comparator" in meta_result1.markdown
    assert "| Metric | Corrected | Published/legacy | Delta |" in meta_result1.markdown
    pdt.assert_frame_equal(
        meta_result1.summary, meta_result2.summary, check_exact=False, atol=1e-12
    )

    boot_result1 = bootstrap.run_bootstrap(
        n_boot=n_boot,
        seed=seed,
        conway_path=conway_path,
        data_by_group=conway_groups,
        out_dir=out_dir1,
    )
    boot_result2 = bootstrap.run_bootstrap(
        n_boot=n_boot,
        seed=seed,
        conway_path=conway_path,
        data_by_group=conway_groups,
        out_dir=out_dir2,
    )
    assert (out_dir1 / "bootstrap_params.csv").exists()
    assert (out_dir1 / "bootstrap_summary.md").exists()
    pdt.assert_frame_equal(boot_result1.draws, boot_result2.draws, check_exact=False, atol=1e-12)
    pdt.assert_frame_equal(
        boot_result1.summary, boot_result2.summary, check_exact=False, atol=1e-12
    )
    assert {
        "corrected_analytic_loa_l",
        "corrected_analytic_loa_u",
        "corrected_analytic_ci_l",
        "corrected_analytic_ci_u",
        "corrected_analytic_outer_width",
    }.issubset(boot_result1.summary.columns)
    assert not any(column.startswith("conway_") for column in boot_result1.summary.columns)
    assert boot_result1.draws["agreement_method_version"].unique().tolist() == [
        AGREEMENT_METHOD_VERSION
    ]
    assert boot_result1.draws["results_status"].unique().tolist() == [RESULTS_STATUS]

    paco2_result1 = paco2.run_paco2_summary(
        paco2_path=paco2_path,
        paco2_data=paco2_source,
        out_dir=out_dir1,
    )
    paco2_result2 = paco2.run_paco2_summary(
        paco2_path=paco2_path,
        paco2_data=paco2_source,
        out_dir=out_dir2,
    )
    assert (out_dir1 / "paco2_distribution_summary.md").exists()
    pdt.assert_frame_equal(
        paco2_result1.summary, paco2_result2.summary, check_exact=False, atol=1e-12
    )

    sim_result1 = sim.run_forward_simulation_summary(
        params=boot_result1.draws,
        paco2_data=paco2_result1.data,
        seed=seed,
        n_draws=10,
        out_dir=out_dir1,
    )
    sim_result2 = sim.run_forward_simulation_summary(
        params=boot_result2.draws,
        paco2_data=paco2_result2.data,
        seed=seed,
        n_draws=10,
        out_dir=out_dir2,
    )
    assert (out_dir1 / "simulation_summary.md").exists()
    pdt.assert_frame_equal(sim_result1.summary, sim_result2.summary, check_exact=False, atol=1e-12)

    infer_result1 = infer.run_inference_demo(
        params=boot_result1.draws,
        paco2_data=paco2_result1.data,
        seed=seed,
        n_draws=10,
        out_dir=out_dir1,
    )
    infer_result2 = infer.run_inference_demo(
        params=boot_result2.draws,
        paco2_data=paco2_result2.data,
        seed=seed,
        n_draws=10,
        out_dir=out_dir2,
    )
    assert (out_dir1 / "inference_demo.md").exists()
    pdt.assert_frame_equal(
        infer_result1.summary, infer_result2.summary, check_exact=False, atol=1e-12
    )

    cond_result1 = conditional.run_conditional_classification(
        params=boot_result1.draws,
        paco2_data=paco2_result1.data,
        seed=seed,
        n_draws=10,
        out_dir=out_dir1,
    )
    cond_result2 = conditional.run_conditional_classification(
        params=boot_result2.draws,
        paco2_data=paco2_result2.data,
        seed=seed,
        n_draws=10,
        out_dir=out_dir2,
    )
    assert (out_dir1 / "conditional_classification_t45.csv").exists()
    assert (out_dir1 / "conditional_classification_t45.md").exists()
    pdt.assert_frame_equal(cond_result1.curves, cond_result2.curves, check_exact=False, atol=1e-12)


def test_bootstrap_analytic_comparator_uses_supplied_group_data() -> None:
    group_name = "noncanonical"
    supplied_data = _synthetic_conway_group(group_name, offset=50.0)
    supplied_groups = ((name, data) for name, data in [(group_name, supplied_data)])

    result = bootstrap.run_bootstrap(
        n_boot=10,
        seed=123,
        data_by_group=supplied_groups,
        truncate_tau2=True,
    )

    expected = conway_group_summary(supplied_data, truncate_tau2=True)
    canonical = conway_group_summary(load_conway_group("main"), truncate_tau2=True)
    assert result.draws["group"].unique().tolist() == [group_name]
    assert result.summary["group"].tolist() == [group_name]
    row = result.summary.iloc[0]
    assert row["corrected_analytic_loa_l"] == pytest.approx(expected.loa_l, abs=1e-12)
    assert row["corrected_analytic_loa_u"] == pytest.approx(expected.loa_u, abs=1e-12)
    assert row["corrected_analytic_ci_l"] == pytest.approx(expected.ci_l, abs=1e-12)
    assert row["corrected_analytic_ci_u"] == pytest.approx(expected.ci_u, abs=1e-12)
    assert row["corrected_analytic_loa_l"] != pytest.approx(canonical.loa_l, abs=1e-6)


def test_meta_checks_handles_zero_heterogeneity_without_runtime_warning() -> None:
    homogeneous_data = pd.DataFrame(
        {
            "study": ["equal_a", "equal_b", "equal_c"],
            "n": [20.0, 20.0, 20.0],
            "n_2": [20.0, 20.0, 20.0],
            "bias": [1.0, 1.0, 1.0],
            "s2": [4.0, 4.0, 4.0],
        }
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = meta.run_meta_checks(data_by_group=[("homogeneous", homogeneous_data)])

    runtime_warnings = [
        warning for warning in caught if issubclass(warning.category, RuntimeWarning)
    ]
    assert runtime_warnings == []
    row = result.summary.iloc[0]
    assert row["tau2"] == pytest.approx(0.0, abs=1e-12)
    assert np.isfinite(
        row[["bias", "sd", "tau2", "loa_l", "loa_u", "ci_l", "ci_u"]].to_numpy(dtype=float)
    ).all()
    loa_half_width = 2 * np.sqrt(row["sd"] ** 2 + row["tau2"])
    assert row["loa_l"] == pytest.approx(row["bias"] - loa_half_width, abs=1e-12)
    assert row["loa_u"] == pytest.approx(row["bias"] + loa_half_width, abs=1e-12)
    assert result.invariants["max_loa_abs_error"] == pytest.approx(0.0, abs=1e-12)


def test_format_inference_demo_requires_single_threshold() -> None:
    likelihood = pd.DataFrame(
        {
            "group": ["pft"],
            "tcco2": [40.0],
            "paco2_q025": [30.0],
            "paco2_q500": [40.0],
            "paco2_q975": [50.0],
        }
    )

    with pytest.raises(ValueError, match="format_inference_demo supports one threshold"):
        infer.format_inference_demo(
            likelihood,
            prior=None,
            thresholds=[40.0, 45.0],
            n_boot=1,
            n_draws=None,
            seed=None,
            paco2_data=_synthetic_paco2_data(),
        )


def _resolve_conway_sources() -> tuple[Path | None, list[tuple[str, pd.DataFrame]] | None]:
    if CONWAY_DATA_PATH.exists():
        return CONWAY_DATA_PATH, None
    groups = []
    offsets = {"main": -0.2, "icu": -0.5, "arf": 1.1, "lft": -0.1}
    for group_name, offset in offsets.items():
        groups.append((group_name, _synthetic_conway_group(group_name, offset)))
    return None, groups


def _synthetic_conway_group(group_name: str, offset: float) -> pd.DataFrame:
    bias = np.array([offset - 0.2, offset, offset + 0.15])
    return pd.DataFrame(
        {
            "study": [f"{group_name}_a", f"{group_name}_b", f"{group_name}_c"],
            "n": [20.0, 25.0, 30.0],
            "n_2": [20.0, 25.0, 30.0],
            "bias": bias,
            "s2": [4.0, 5.5, 6.0],
        }
    )


def _resolve_paco2_sources() -> tuple[Path | None, pd.DataFrame | None]:
    if INSILICO_PACO2_PATH.exists():
        return INSILICO_PACO2_PATH, None
    return None, _synthetic_paco2_data()


def _synthetic_paco2_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "paco2": [35.0, 40.0, 50.0, 42.0, 47.0, 55.0, 30.0, 60.0, 38.0],
            "is_amb": [1, 1, 1, 0, 0, 0, 0, 0, 0],
            "is_emer": [0, 0, 0, 1, 1, 1, 0, 0, 0],
            "is_inp": [0, 0, 0, 1, 1, 1, 1, 1, 1],
            "cc_time": [0, 0, 0, 0, 0, 0, 1, 1, 1],
        }
    )

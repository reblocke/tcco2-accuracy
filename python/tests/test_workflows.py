from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt

from tcco2_accuracy.data import CONWAY_DATA_PATH, INSILICO_PACO2_PATH
from tcco2_accuracy.workflows import bootstrap, infer, meta, paco2, sim


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
        out_dir=out_dir1,
    )
    meta_result2 = meta.run_meta_checks(
        conway_path=conway_path,
        data_by_group=conway_groups,
        out_dir=out_dir2,
    )
    assert (out_dir1 / "meta_loa_check.md").exists()
    pdt.assert_frame_equal(meta_result1.summary, meta_result2.summary, check_exact=False, atol=1e-12)

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
    pdt.assert_frame_equal(boot_result1.summary, boot_result2.summary, check_exact=False, atol=1e-12)

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
    pdt.assert_frame_equal(paco2_result1.summary, paco2_result2.summary, check_exact=False, atol=1e-12)

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
    pdt.assert_frame_equal(infer_result1.summary, infer_result2.summary, check_exact=False, atol=1e-12)


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

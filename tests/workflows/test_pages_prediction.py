import json
from pathlib import Path

import numpy as np
import pytest

from tcco2_accuracy.pages_prediction import conditional

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "models/pages"
PRIOR = (ROOT / "tests/fixtures/synthetic_paco2_prior.csv").read_text()
CASES = json.loads((ROOT / "tests/fixtures/future_setting_synthetic_reference.json").read_text())


@pytest.mark.parametrize("case", CASES)
def test_public_kernel_reproduces_native_fit(case):
    target = case["target_stratum"]
    group = {"all": "main", "icu": "icu", "ed_inp": "arf", "pft": "lft"}[target]
    with np.load(SOURCE / f"{group}.npz") as kernel:
        result = conditional(kernel, target, PRIOR, float(case["tcco2"]), 45)
    assert result["quantiles"] == [float(case[k]) for k in ("lower", "median", "upper")]
    assert result["probability"] == pytest.approx(float(case["probability_ge"]), abs=1e-4)


def test_bounds_and_equality():
    with np.load(SOURCE / "main.npz") as kernel:
        with pytest.raises(ValueError, match="offline"):
            conditional(kernel, "all", PRIOR, 999, 45)
        prior = "group,paco2_bin,weight\n" + "".join(
            f"{g},45,1\n" for g in ["all", "icu", "ed_inp", "pft"]
        )
        assert conditional(kernel, "all", prior, 45.25, 45)["probability"] == 1

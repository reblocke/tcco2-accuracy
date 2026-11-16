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


def test_dense_tail_precision_independent_at_47():
    with np.load(SOURCE / "main.npz") as kernel:
        k = dict(kernel)
    r = conditional(k, "all", PRIOR, 47, 52)
    assert r["tails"] == pytest.approx([0.8003835221376633, 0.05302181782790692])
    assert max(r["likelihood"]) == pytest.approx(1)
    assert sum(r["mass"]) == pytest.approx(1)
    assert r["prior"] == pytest.approx([0.1, 0.2, 0.4, 0.2, 0.1])
    # Force coarse-order density error only for the upper two support values.
    prior = "group,paco2_bin,weight\n" + "".join(
        f"{g},{a},0.5\n" for g in ["all", "pft", "icu", "ed_inp"] for a in (45, 52)
    )
    k["logpdf"][1, k["x"] == 5] += 0.2
    r = conditional(k, "all", prior, 47, 45)
    assert r["tails"] == [1, None]

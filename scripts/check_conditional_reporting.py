"""Replay the scientific acceptance cases written before the reporting amendment."""

import json
from pathlib import Path

import numpy as np

from tcco2_accuracy.core.conditional_reporting import display_range, qualify
from tcco2_accuracy.pages_prediction import conditional


def check():
    results = []

    def evaluate(a, p, delta=None, coarse=None):
        a, p = np.array(a, dtype=float), np.array(p, dtype=float)
        delta = np.zeros_like(p) if delta is None else np.array(delta)
        batches = np.tile([p - delta, p + delta], (16, 1))
        return qualify(a, p, p if coarse is None else np.array(coarse), [45, 52], [batches])

    for value, tails in [(45, [1, 0]), (52, [1, 1])]:
        q, p, _, _ = evaluate([value], [1])
        assert q == [value] * 3 and p == tails
    results.append("Inclusive thresholds and exact structural probabilities")
    q, _, report, _ = evaluate([40, 40.3, 50], [0.025, 0.5, 0.475], [0.001, -0.001, 0])
    assert q[0] == 40 and report["quantiles"][0] == [40, 40.3]
    q, _, _, _ = evaluate([40, 41, 50], [0.025, 0.5, 0.475], [0.001, -0.001, 0])
    assert q[0] is None
    results.append("Narrow and wide discrete-quantile brackets")
    q, _, _, _ = evaluate([30, 40, 60, 80], [0.03, 0.47, 0.47, 0.03], [0, 0.001, -0.001, 0])
    assert q == [30, None, 80]
    results.append("Median held independently of interval")
    _, p, _, _ = evaluate([40, 50], [0.5, 0.5], coarse=[0.5002, 0.4998])
    assert p == [None, 0]
    results.append("Separate quadrature threshold gate")
    assert display_range([35.49, 35.51]) == "35–36"
    assert display_range([0.004, 0.006], percent=True) == "<1%"
    assert display_range([0.994, 0.996], percent=True) == ">99%"
    _, _, report, _ = evaluate([40, 52], [1.0, 0.0])
    assert display_range(report["probabilities"][0], percent=True) == "<1%"
    results.append("Whole-unit boundary and supported extreme formatting")

    x = np.arange(-50, 51, dtype=float)
    # Deliberately unequal evidence across two posterior components.
    like = np.stack([np.exp(-0.5 * ((x - m) / s) ** 2) / s for m, s in [(-3, 2), (4, 8)]])
    mean = like.mean(axis=0)
    kernel = dict(
        group="main", x=x, logpdf=np.log([mean, mean]), batch_logpdf=np.tile(np.log(mean), (32, 1))
    )
    a, w = np.array([40.0, 45.0, 52.0]), np.array([0.2, 0.3, 0.5])

    def prior(support):
        return "group,paco2_bin,weight\n" + "".join(
            f"{group},{v},{p}\n"
            for group in ("all", "icu", "ed_inp", "pft")
            for v, p in zip(support, w, strict=True)
        )

    got = conditional(kernel, "all", prior(a), 47, 45)
    expected = w * mean[np.searchsorted(x, a - 47)]
    expected /= expected.sum()
    np.testing.assert_allclose(got["mass"], expected, atol=1e-14)
    wrong = like[:, np.searchsorted(x, a - 47)] * w
    wrong = (wrong / wrong.sum(axis=1)[:, None]).mean(axis=0)
    assert np.max(abs(wrong - expected)) > 0.01
    moved = conditional(kernel, "all", prior(a + 10), 57, 55)
    np.testing.assert_allclose(got["mass"], moved["mass"], atol=1e-14)
    results.append("Signed offset, paired-offset invariance and average before normalization")
    return results


if __name__ == "__main__":
    passed = check()
    out = Path(".pytest_tmp/conditional-reporting-acceptance.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(dict(passed=passed), indent=2) + "\n")
    print(f"Passed {len(passed)} scientific acceptance groups. Receipt at {out}")

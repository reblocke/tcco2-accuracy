"""Analytic acceptance cases for the private grid and secondary strategy integration."""

import sys
from pathlib import Path

import numpy as np
from predictions import inverse
from scipy.stats import norm
from strategy import metrics


def verify(library):
    a = np.array([27.0, 42.0, 46.0, 52.0, 71.0])
    weights = np.array([0.03, 0.22, 0.45, 0.20, 0.10])
    # Equal-count pairs in every contiguous batch give known mixture weights.
    parameters = np.array([[-3.0, np.log(9), 0.0, 0.0, 0.0], [9.0, np.log(36), 0.0, 0.0, 0.0]])
    theta = np.tile(parameters, (4, 32, 1))
    rows, _ = inverse(theta, a, weights, library)
    for t, lower, median, upper, p45, p52 in rows:
        likes = norm.pdf(a[:, None] - t, loc=parameters[:, 0], scale=[3, 6])
        posterior = weights * likes.mean(axis=1)
        posterior /= posterior.sum()
        expected = a[np.searchsorted(np.cumsum(posterior), [0.025, 0.5, 0.975])]
        np.testing.assert_allclose([lower, median, upper], expected)
        np.testing.assert_allclose(
            [p45, p52], [posterior[a >= 45].sum(), posterior[a >= 52].sum()], atol=1e-12
        )
    # Zero measurement error; the middle group receives an ABG and classification is exact.
    with np.errstate(divide="ignore"):
        m = metrics(np.array([0.60, 0.60, 0.60, 0.25]), np.array([0.40, 0.12, 0.0, 0.0]))
    np.testing.assert_allclose(
        [
            m["reflex_ABG"],
            m["sensitivity"],
            m["specificity"],
            m["residual_FN"],
            m["residual_FP"],
            m["classified_accuracy"],
        ],
        [0.47, 1.0, 1.0, 0.0, 0.0, 1.0],
    )


if __name__ == "__main__":
    verify(Path(sys.argv[1]).resolve())

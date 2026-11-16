"""Independent scientific acceptance cases for one new paired observation."""

import numpy as np
import pytest
from scipy import integrate, stats

from tcco2_accuracy.core.future_setting import PredictiveKernel, inverse_summary


def kernel(*rows, order=48):
    return PredictiveKernel(np.asarray(rows, dtype=float)[None, :, :], order=order)


def test_normal_case_pair_sign_and_variance():
    arterial = np.array([30, 58, 92])
    offset = np.array([10, -7, 21])
    for difference in (0, 3):
        transcutaneous = arterial - difference
        d = (arterial + offset) - (transcutaneous + offset)
        np.testing.assert_array_equal(d, arterial - transcutaneous)
        k = kernel([d.mean(), np.log(9), 4, 0, 0.8])
        np.testing.assert_allclose(
            k.quantiles(), stats.norm.ppf([0.025, 0.5, 0.975], difference, 5)
        )
        assert sum(k.variance_components().values()) == pytest.approx(25)


def test_heterogeneous_kernel_against_independent_latent_simulation_and_units():
    rng = np.random.default_rng(5223)
    theta = [1.2, np.log(9), 2.3, 0.75, -0.65]
    k = kernel(theta)
    b, z = rng.multivariate_normal(
        theta[:2], [[2.3**2, -0.65 * 2.3 * 0.75], [-0.65 * 2.3 * 0.75, 0.75**2]], 400000
    ).T
    observations = b + np.exp(z / 2) * rng.standard_normal(len(b))
    np.testing.assert_allclose(
        k.quantiles(), np.quantile(observations, [0.025, 0.5, 0.975]), atol=0.10
    )
    assert integrate.quad(lambda x: np.exp(k.logpdf(x)), -np.inf, np.inf)[0] == pytest.approx(1)
    np.testing.assert_allclose(k.cdf(k.quantiles()), [0.025, 0.5, 0.975], atol=1e-9)
    c = 0.133322
    converted = kernel([c * theta[0], theta[1] + 2 * np.log(c), c * theta[2], *theta[3:]])
    np.testing.assert_allclose(converted.quantiles(), c * k.quantiles())
    assert sum(k.variance_components().values()) == pytest.approx(np.var(observations), rel=0.015)


def test_inverse_normal_and_mixture_order():
    grid = np.arange(1, 100, 0.02)
    prior = stats.norm.pdf(grid, 50, 6)
    posterior = inverse_summary(kernel([3, np.log(4), 0, 0, 0]), grid, prior, 40)
    mean = (50 / 36 + 43 / 4) / (1 / 36 + 1 / 4)
    sd = (1 / 36 + 1 / 4) ** -0.5
    np.testing.assert_allclose(
        posterior["quantiles"], stats.norm.ppf([0.025, 0.5, 0.975], mean, sd), atol=0.021
    )
    assert posterior["probability_ge"] == pytest.approx(
        stats.norm.sf(45 - 0.01, mean, sd), abs=0.002
    )
    support = np.array([40, 52])
    weights = np.array([0.9, 0.1])
    k = kernel([0, 0, 0, 0, 0], [12, np.log(9), 0, 0, 0])
    lik = np.stack((stats.norm.pdf(support - 40), stats.norm.pdf(support - 40, 12, 3)))
    truth = weights * lik.mean(axis=0)
    wrong = (lik * weights) / (lik * weights).sum(axis=1)[:, None]
    result = inverse_summary(k, support, weights, 40)
    assert result["probability_ge"] == pytest.approx(truth[1] / truth.sum())
    assert abs(result["probability_ge"] - wrong[:, 1].mean()) > 0.2
    assert inverse_summary(k, [45], [1], 40)["probability_ge"] == 1

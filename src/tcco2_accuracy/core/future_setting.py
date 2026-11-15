"""Posterior prediction for one paired arterial minus transcutaneous measurement."""

from __future__ import annotations

import numpy as np
from scipy import optimize, special

from ._posterior import normalize_log_weights, weighted_quantile

LEVELS = np.array([0.025, 0.5, 0.975])
PARAMETERS = ("mu_b", "mu_ell", "tau_b", "tau_ell", "rho")
METHOD = "future_setting_joint_v1"


class PredictiveKernel:
    """Five joint hyperparameters, ordered as PARAMETERS, indexed by chain and draw."""

    def __init__(self, theta, *, order: int = 32):
        x = np.asarray(theta, dtype=float)
        if x.ndim != 3 or not all(x.shape) or x.shape[-1] != 5 or not np.isfinite(x).all():
            raise ValueError("Expected finite chain × draw × five joint hyperparameters")
        if (x[..., 2:4] < 0).any() or (np.abs(x[..., 4]) > 1).any():
            raise ValueError("Invalid heterogeneity or correlation")
        if not isinstance(order, int) or order < 2:
            raise ValueError("Quadrature requires at least two nodes")
        self.theta = x
        self.order = order
        # Keep temporary quadrature arrays small for million-draw precision runs.
        chunk = max(1, 500_000 // (x.shape[0] * order))
        self.tiles = []
        if x.shape[1] > chunk:
            self.tiles = [
                PredictiveKernel(x[:, j : j + chunk], order=order)
                for j in range(0, x.shape[1], chunk)
            ]
            return
        nodes, weights = special.roots_hermitenorm(order)
        self.weights = weights / weights.sum()
        mb, ml, tb, tl, rho = (a[..., None] for a in np.moveaxis(x, -1, 0))
        # At exactly zero log-variance heterogeneity integrate all future bias analytically.
        loading = np.where(tl == 0, 0, rho * tb)
        self.mean = mb + loading * nodes
        self.sd = np.sqrt(np.exp(ml + tl * nodes) + (tb**2 - loading**2))
        if not np.isfinite(self.sd).all() or (self.sd <= 0).any():
            raise ValueError("Nonrepresentable predictive scale")
        self.log_factor = np.log(self.weights) - np.log(self.sd) - 0.5 * np.log(2 * np.pi)

    def draw_logpdf(self, d):
        if self.tiles:
            return np.concatenate([tile.draw_logpdf(d) for tile in self.tiles], axis=1)
        return special.logsumexp(self.log_factor - 0.5 * ((d - self.mean) / self.sd) ** 2, axis=-1)

    def draw_cdf(self, d):
        if self.tiles:
            return np.concatenate([tile.draw_cdf(d) for tile in self.tiles], axis=1)
        return np.sum(self.weights * special.ndtr((d - self.mean) / self.sd), axis=-1)

    def logpdf(self, d):
        def one(value):
            likelihood = self.draw_logpdf(value)
            return special.logsumexp(likelihood) - np.log(likelihood.size)

        return _evaluate(one, d)

    def cdf(self, d):
        return _evaluate(lambda value: self.draw_cdf(value).mean(), d)

    def quantile(self, probability):
        if not 0 < probability < 1:
            raise ValueError("Predictive quantiles require interior probabilities")
        center = self.theta[..., 0].mean()
        # Chebyshev bounds bracket the quantile of this finite draw mixture.
        radius = np.sqrt(
            sum(self.variance_components().values()) / min(probability, 1 - probability)
        )
        return optimize.brentq(
            lambda v: self.cdf(v) - probability, center - radius, center + radius
        )

    def quantiles(self):
        return np.array([self.quantile(p) for p in LEVELS])

    def variance_components(self):
        mb, ml, tb, tl, _ = np.moveaxis(self.theta, -1, 0)
        return {
            "expected_within_variance": float(np.exp(ml + tl**2 / 2).mean()),
            "expected_between_bias_variance": float((tb**2).mean()),
            "variance_of_overall_bias": float(mb.var()),
        }

    def population_endpoint_draws(self, p):
        """One population endpoint per joint hyperparameter draw."""
        if self.tiles:
            return np.concatenate(
                [tile.population_endpoint_draws(p) for tile in self.tiles], axis=1
            )
        bounds = self.mean + special.ndtri(p) * self.sd
        lo, hi = bounds.min(axis=-1), bounds.max(axis=-1)
        current = np.clip(
            self.theta[..., 0]
            + special.ndtri(p)
            * np.sqrt(
                self.theta[..., 2] ** 2 + np.exp(self.theta[..., 1] + self.theta[..., 3] ** 2 / 2)
            ),
            lo,
            hi,
        )
        for _ in range(48):
            z = (current[..., None] - self.mean) / self.sd
            residual = np.sum(self.weights * special.ndtr(z), axis=-1) - p
            done = (abs(residual) < 1e-12) | (hi - lo < 1e-8)
            if done.all():
                return current
            lo = np.where(residual < 0, current, lo)
            hi = np.where(residual >= 0, current, hi)
            density = np.sum(np.exp(self.log_factor - 0.5 * z**2), axis=-1)
            step = current - residual / np.maximum(density, np.finfo(float).tiny)
            current = np.where(
                done, current, np.where((step > lo) & (step < hi), step, (lo + hi) / 2)
            )
        return current


def _evaluate(fun, values):
    values = np.asarray(values, dtype=float)
    if np.isnan(values).any():
        raise ValueError("Invalid predictive evaluation point")
    if values.ndim == 0:
        return float(fun(float(values)))
    return np.array([fun(v) for v in values.flat]).reshape(values.shape)


def empirical_support(support, weights):
    a, w = np.asarray(support, dtype=float), np.asarray(weights, dtype=float)
    if (
        a.ndim != 1
        or a.size == 0
        or a.shape != w.shape
        or not np.isfinite(a).all()
        or not np.isfinite(w).all()
        or (a <= 0).any()
        or (w < 0).any()
        or not (w > 0).any()
    ):
        raise ValueError("Invalid empirical PaCO2 support or masses")
    positive = w > 0
    a, inv = np.unique(a[positive], return_inverse=True)
    w = np.bincount(inv, weights=w[positive] / w[positive].max())
    return a, w / w.sum()


def inverse_summary(kernel, support, weights, tcco2, *, threshold=45.0):
    """Condition on a fixed empirical prior using the entire marginal error density."""
    if not isinstance(kernel, PredictiveKernel):
        raise TypeError("Inverse inference requires a future-setting posterior kernel")
    if not np.isfinite([tcco2, threshold]).all() or min(tcco2, threshold) <= 0:
        raise ValueError("Pressures must be positive and finite")
    a, w = empirical_support(support, weights)
    masses = normalize_log_weights(np.log(w) + kernel.logpdf(a - tcco2))
    return {
        "quantiles": weighted_quantile(a, masses, LEVELS),
        "probability_ge": float(np.clip(masses[a >= threshold].sum(), 0, 1)),
    }

"""Candidate-only future-setting reports and numerical precision checks."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from ..core.future_setting import (
    LEVELS,
    METHOD,
    PredictiveKernel,
    empirical_support,
    inverse_summary,
)
from ..workflows.future_setting_data import ROOT, SPEC, load_ledger, select_units
from ..workflows.meta import run_meta_checks

GRID = (35, 40, 45, 50, 55)


def batch_mcse(values):
    """Contiguous equal batches within each chain, including between-chain variation."""
    chains, n = values.shape
    size = int(np.sqrt(n))
    k = n // size
    means = values[:, : k * size].reshape(chains, k, size).mean(axis=2)
    return float(means.std(ddof=1) / np.sqrt(means.size))


def _inverse_checks(kernel, support, weights, tcco2):
    a, w = empirical_support(support, weights)
    # Only used for the small synthetic reporting prior. Each row keeps its
    # evidence denominator until after integration over the parameter posterior.
    logjoint = np.stack(
        [kernel.draw_logpdf(v - tcco2) + np.log(mass) for v, mass in zip(a, w, strict=True)]
    )
    joint = np.exp(logjoint - logjoint.max())
    den = joint.sum(axis=0)
    if den.mean() <= 0:
        raise ValueError("Cannot normalize inverse likelihood")

    def ratio(mask):
        num = joint[mask].sum(axis=0)
        p = num.mean() / den.mean()
        se = batch_mcse((num - p * den) / den.mean())
        return float(p), se

    result = inverse_summary(kernel, a, w, tcco2)
    p, se = ratio(a >= 45)
    if not np.isclose(p, result["probability_ge"], rtol=1e-12, atol=1e-14):
        raise ValueError("Inconsistent posterior marginalization")
    limits = []
    for q, v in zip(LEVELS, result["quantiles"], strict=True):
        left, lse = ratio(a < v)
        right, rse = ratio(a <= v)
        limits.append(
            dict(
                level=q,
                left=left,
                at=right,
                left_mcse=lse,
                at_mcse=rse,
                resolved=bool(left + 2 * lse < q and right - 2 * rse >= q),
            )
        )
    return result, se, limits


def _ppc(theta, data):
    rng = np.random.default_rng(88033)
    flat = theta.reshape(-1, 5)
    chosen = flat[np.linspace(0, len(flat) - 1, min(len(flat), 2000), dtype=int)]
    y = np.column_stack([data["bias"], data["ell"]])
    n = len(y)
    sampling = np.zeros((n, 2, 2))
    i = data["I"]
    sampling[:i, 0, 0] = data["v_bias"]
    sampling[:i, 1, 1] = data["v_ell"]
    sampling[:i, 0, 1] = sampling[:i, 1, 0] = data["cov_bias_ell"]

    def metrics(z):
        return [*z.mean(axis=0), *z.std(axis=0, ddof=1), np.corrcoef(z.T)[0, 1]]

    reps = []
    for mb, ml, tb, tl, r in chosen:
        sigma = np.array([[tb**2, r * tb * tl], [r * tb * tl, tl**2]])
        fake = np.empty_like(y)
        fake[:i] = [mb, ml] + np.einsum(
            "nij,nj->ni", np.linalg.cholesky(sampling[:i] + sigma), rng.normal(size=(i, 2))
        )
        start = 0
        for k in data["block_size"]:
            idx = slice(2 * start, 2 * (start + k))
            ids = np.asarray(data["setting_id"][start : start + k])
            cov = data["V_dep"][idx, idx] + np.kron(ids[:, None] == ids, sigma)
            fake[i + start : i + start + k] = rng.multivariate_normal(
                np.tile([mb, ml], k), cov
            ).reshape(k, 2)
            start += k
        reps.append(metrics(fake))
    return [
        {"statistic": name, "observed": v, "replicated_q025": lo, "replicated_q975": hi}
        for name, v, (lo, hi) in zip(
            ("mean_bias", "mean_log_variance", "sd_bias", "sd_log_variance", "correlation"),
            metrics(y),
            np.quantile(reps, [0.025, 0.975], axis=0).T,
            strict=True,
        )
    ]


def assess(theta, data, prior, group, mode):
    support, weights = prior.paco2_bin, prior.weight
    low = PredictiveKernel(theta, order=32)
    previous = low.quantiles()
    prev_inverse = [inverse_summary(low, support, weights, t) for t in GRID]
    del low
    for order in (64, 128, 256):
        k = PredictiveKernel(theta, order=order)
        qs = k.quantiles()
        inv = [_inverse_checks(k, support, weights, t) for t in GRID]
        qerr = float(np.max(abs(previous - qs)))
        perr = max(
            abs(p["probability_ge"] - j[0]["probability_ge"])
            for p, j in zip(prev_inverse, inv, strict=True)
        )
        same = all(
            np.array_equal(p["quantiles"], j[0]["quantiles"])
            for p, j in zip(prev_inverse, inv, strict=True)
        )
        if qerr <= 0.01 and perr <= 0.0001 and same:
            break
        previous, prev_inverse = qs, [j[0] for j in inv]
        if order < 256:
            del k
    endpoint_se = [batch_mcse(k.draw_cdf(q)) / np.exp(k.logpdf(q)) for q in qs]
    width_influence = (k.draw_cdf(qs[0]) - LEVELS[0]) / np.exp(k.logpdf(qs[0])) - (
        k.draw_cdf(qs[2]) - LEVELS[2]
    ) / np.exp(k.logpdf(qs[2]))
    width_mcse = batch_mcse(width_influence)
    acc = {
        "quadrature_order": order,
        "endpoint_quadrature_error": qerr,
        "probability_quadrature_error": perr,
        "quadrature_discrete_match": same,
        "predictive_mcse_mmHg": endpoint_se,
        "width_mcse_mmHg": width_mcse,
        "probability_mcse": {str(t): j[1] for t, j in zip(GRID, inv, strict=True)},
        "discrete_quantile_checks": {str(t): j[2] for t, j in zip(GRID, inv, strict=True)},
    }
    acc["passed"] = bool(
        max(*endpoint_se, width_mcse) <= 0.025
        and qerr <= 0.01
        and perr <= 0.0001
        and same
        and all(j[1] <= 0.00025 and all(row["resolved"] for row in j[2]) for j in inv)
    )
    status = "synthetic_recovery" if mode == "recovery" else "candidate"
    common = dict(
        stratum=group,
        method_version=METHOD,
        source_revision=SPEC["observation_version"],
        interval_level=0.95,
        unit="mmHg",
        result_status=status,
    )
    predictive = dict(
        **common,
        quantity_type="posterior_predictive_disagreement",
        lower=qs[0],
        median=qs[1],
        upper=qs[2],
        width=qs[2] - qs[0],
    )
    target = next((s for s, g in SPEC["target_mapping"].items() if g == group), "all")
    inverse = [
        dict(
            **common,
            quantity_type="posterior_paco2_given_tcco2",
            target_stratum=target,
            tcco2=t,
            lower=q["quantiles"][0],
            median=q["quantiles"][1],
            upper=q["quantiles"][2],
            probability_ge=q["probability_ge"],
            threshold=45,
            operator=">=",
            probability_unit="probability",
            prior_revision="synthetic_fixture_v1",
            prior_uncertainty="fixed_empirical",
            support_representation="synthetic discrete bin centers",
            reference_standard="arterial PaCO2",
            sampling_unit="invented aggregate prior masses",
        )
        for t, (q, _, _) in zip(GRID, inv, strict=True)
    ]
    var = k.variance_components()
    variance = dict(
        stratum=group,
        method_version=METHOD,
        source_revision=SPEC["observation_version"],
        result_status=status,
        quantity_type="predictive_variance_decomposition",
        unit="mmHg^2",
        **var,
        total=sum(var.values()),
    )
    result = dict(predictive=predictive, inverse=inverse, variance=variance)
    if acc["passed"]:
        result["endpoint_intervals"] = []
        for p in (0.025, 0.975):
            limits = k.population_endpoint_draws(p)
            ci = np.quantile(limits, [0.025, 0.975])
            chains, n = limits.shape
            batches = np.array_split(limits, max(4, int(np.sqrt(n))), axis=1)
            batch_limits = np.concatenate(
                [np.quantile(b, [0.025, 0.975], axis=1) for b in batches], axis=1
            )
            se = batch_limits.std(axis=1, ddof=1) / np.sqrt(batch_limits.shape[1])
            acc.setdefault("pointwise_endpoint_interval_mcse", {})[str(p)] = se.tolist()
            result["endpoint_intervals"].append(
                dict(
                    **common,
                    quantity_type="population_endpoint_pointwise_credible_interval",
                    endpoint_level=p,
                    lower=ci[0],
                    upper=ci[1],
                    lower_mcse=se[0],
                    upper_mcse=se[1],
                )
            )
            if se.max() > 0.025:
                acc["passed"] = False
        result["ppc"] = [dict(stratum=group, **row) for row in _ppc(theta, data)]
    return result, acc


def _recovery_truth(result):
    """Independent two-dimensional integration of latent b and ell using tensor quadrature."""
    nodes, weights = special.roots_hermitenorm(80)
    z1, z2 = np.meshgrid(nodes, nodes)
    mass = np.outer(weights, weights) / (2 * np.pi)
    b = 1 + 2 * z1
    sd = np.exp((np.log(9) + 0.5 * (-0.4 * z1 + np.sqrt(1 - 0.4**2) * z2)) / 2)

    def cdf(x):
        return (mass * stats.norm.cdf(x, b, sd)).sum()

    def density(x):
        return (mass * stats.norm.pdf(x, b, sd)).sum()

    true_q = [optimize.brentq(lambda x: cdf(x) - p, -50, 50) for p in LEVELS]
    pred = result["predictive"]
    qerr = float(np.max(abs(np.array(true_q) - [pred["lower"], pred["median"], pred["upper"]])))
    fixture = (
        pd.read_csv(ROOT / "tests/fixtures/synthetic_paco2_prior.csv")
        .query("group=='all'")
        .sort_values("paco2_bin")
    )
    a, w = fixture.paco2_bin.to_numpy(), fixture.weight.to_numpy()
    errors = []
    for r in result["inverse"]:
        joint = w * np.array([density(x - r["tcco2"]) for x in a])
        post = joint / joint.sum()
        truth = a[np.searchsorted(post.cumsum(), LEVELS)]
        recovered = np.array([r["lower"], r["median"], r["upper"]])
        step = int(max(abs(np.searchsorted(a, truth) - np.searchsorted(a, recovered))))
        prob = float(post[a >= 45].sum())
        errors.append(
            dict(
                tcco2=r["tcco2"],
                true_quantiles=truth.tolist(),
                true_probability=prob,
                probability_error=abs(prob - r["probability_ge"]),
                quantile_support_steps=step,
            )
        )
    return dict(
        true_predictive_quantiles=true_q,
        generating_mass_inside_fitted_interval=float(cdf(pred["upper"]) - cdf(pred["lower"])),
        max_predictive_error=qerr,
        inverse=errors,
        passed=qerr <= 1.5
        and all(
            e["probability_error"] <= 0.10 and e["quantile_support_steps"] <= 1 for e in errors
        ),
    )


def _bridge(manifest, summaries):
    published = pd.read_csv(ROOT / "tests/fixtures/conway_table1.csv").set_index("population")
    corrected = run_meta_checks().summary
    ledger = load_ledger()
    rows = []
    for rec in corrected.itertuples():
        g = rec.group
        if g not in manifest["strata"]:
            continue
        original = published.loc[rec.population]
        for method, version, rev, values, membership in (
            (
                "published",
                "Conway_Table1",
                "10.1136/thoraxjnl-2017-211466",
                original,
                f"Table 1: {int(original.studies)} reported studies",
            ),
            (
                "corrected_legacy",
                "agreement_natural_log_tau2_direct_v1",
                "Data/conway_studies.csv",
                rec,
                f"{len(ledger) if g == 'main' else len(select_units(ledger, g))} legacy effect rows",
            ),
        ):
            for quantity, lo, hi, uncertainty in (
                (
                    "reference_limits",
                    values.loa_l,
                    values.loa_u,
                    "point estimate, normal reference",
                ),
                (
                    "endpoint_confidence_envelope",
                    values.ci_l,
                    values.ci_u,
                    "outer pointwise endpoint confidence bounds",
                ),
            ):
                rows.append(
                    dict(
                        stratum=g,
                        method=method,
                        method_version=version,
                        source_revision=rev,
                        membership=membership,
                        quantity_type=quantity,
                        lower=lo,
                        upper=hi,
                        interval_level=0.95,
                        unit="mmHg",
                        uncertainty=uncertainty,
                        result_status="published" if method == "published" else "provisional",
                    )
                )
        p = summaries.get(g, {}).get("predictive", {})
        rows.append(
            dict(
                stratum=g,
                method="hierarchical",
                method_version=METHOD,
                source_revision=manifest["input_sha256"]["models/future_setting/source_ledger.csv"],
                membership=f"{len(select_units(ledger, g))} adjudication ledger units",
                quantity_type="posterior_predictive_disagreement",
                lower=p.get("lower"),
                upper=p.get("upper"),
                interval_level=0.95,
                unit="mmHg",
                uncertainty="posterior average over new settings and paired observations",
                result_status=manifest["strata"][g]["status"],
            )
        )
    return rows


def finish_run(out, manifest, summaries):
    collections = {
        "predictive_disagreement": [],
        "variance_bookkeeping": [],
        "paco2_given_tcco2": [],
        "population_endpoint_intervals": [],
        "summary_ppc": [],
    }
    for g, r in summaries.items():
        collections["predictive_disagreement"].append(r["predictive"])
        collections["variance_bookkeeping"].append(r["variance"])
        collections["paco2_given_tcco2"].extend(r["inverse"])
        collections["population_endpoint_intervals"].extend(r["endpoint_intervals"])
        collections["summary_ppc"].extend(r["ppc"])
        if manifest["mode"] == "recovery":
            truth = _recovery_truth(r)
            (out / "recovery.json").write_text(json.dumps(truth, indent=2) + "\n")
            if not truth["passed"]:
                manifest["strata"][g]["status"] = "failed_recovery"
    if manifest["mode"] == "public":
        collections["interval_bridge"] = _bridge(manifest, summaries)
    for name, rows in collections.items():
        df = pd.DataFrame(rows)
        if not df.empty:
            df["run_id"] = manifest["run_id"]
            df["input_revision"] = (
                manifest["input_sha256"]["models/future_setting/source_ledger.csv"]
                if manifest["mode"] == "public"
                else "synthetic_generator_88031"
            )
            if name == "interval_bridge":
                for method, source in (
                    ("published", "tests/fixtures/conway_table1.csv"),
                    ("corrected_legacy", "Data/conway_studies.csv"),
                ):
                    df.loc[df.method == method, "input_revision"] = manifest["input_sha256"][source]
            df.to_csv(out / (name + ".csv"), index=False)
        else:
            (out / (name + ".csv")).unlink(missing_ok=True)
    states = ", ".join(f"{g} ({i['status']})" for g, i in manifest["strata"].items())
    (out / "manuscript_framing.md").write_text(
        "# Candidate methods and interpretation\n\n"
        f"Run {manifest['run_id']} recorded {states}.\n\n"
        "Conway et al. used outer confidence bounds to assess interchangeability. Their construction has a tolerance-style purpose. The archive and corrected reconstruction differ in the log-variance calculation and the endpoint uncertainty formula. The reconstructed overall pool adds the Bolliger ICU row. The adjudicated inputs also incorporate source-specific corrections. These differences limit comparisons of interval widths.\n\n"
        "The hierarchical model predicts one arterial minus transcutaneous difference in mmHg in a new comparable setting. The two study summary statistics enter a joint likelihood with estimated sampling covariance. Posterior integration includes setting heterogeneity and uncertainty about the model parameters. The interval endpoints are quantiles of that integrated distribution. Pointwise intervals for the population endpoints describe uncertainty about the limits themselves.\n\n"
        "The overall source population includes children and neonates. The LFT source group includes respiratory inpatients. The conditional demonstration combines a synthetic empirical arterial distribution with the fitted disagreement density, assuming that the error distribution and mixture of settings transport across candidate PaCO₂ within each stratum. The real target would be the selected ABG-observed cohort. Its prior would be treated as fixed, with cohort selection and prior sampling uncertainty outside this calculation.\n\n"
        "The replicated study summaries assess model fit for aggregate bias and log variance. Claims of individual tail calibration, interchangeability, clinical benefit and safety require separate evidence.\n"
    )

"""Independent likelihood, unit and prior checks for the actual Stan binary."""

import numpy as np
import pandas as pd
from scipy import special, stats

from tcco2_accuracy.workflows.future_setting_data import SPEC, stan_data
from tcco2_accuracy.workflows.future_setting_inverse import index_support


def _prior_check():
    spec = SPEC["prior"]
    rng = np.random.default_rng(20261115)
    n = 600_000
    mb = rng.normal(spec["prior_mu_b"], spec["prior_mu_b_sd"], n)
    ml = rng.normal(spec["prior_mu_ell"], spec["prior_mu_ell_sd"], n)
    tb = abs(rng.normal(0, spec["prior_tau_b"], n))
    tl = abs(rng.normal(0, spec["prior_tau_ell"], n))
    rho = 2 * rng.beta(spec["eta"], spec["eta"], n) - 1
    z = rng.normal(size=n)
    b = mb + tb * (rho * z + np.sqrt(1 - rho**2) * rng.normal(size=n))
    d = b + np.exp((ml + tl * z) / 2) * rng.normal(size=n)
    return dict(
        seed=20261115, draws=n, unit="mmHg", quantiles=np.quantile(d, [0.025, 0.5, 0.975]).tolist()
    )


def _latent_log_density(y, v, theta):
    # Integrate a latent (bias, ell) pair explicitly; the production likelihood
    # uses its analytic Gaussian convolution. Both summaries in a block share it.
    nodes, w = special.roots_hermitenorm(72)
    z1, z2 = np.meshgrid(nodes, nodes)
    rho = theta["rho"]
    bias = theta["mu_b"] + theta["tau_b"] * z1
    ell = theta["mu_ell"] + theta["tau_ell"] * (rho * z1 + np.sqrt(1 - rho**2) * z2)
    latent = np.tile(np.stack((bias, ell), axis=-1), y.size // 2)
    return special.logsumexp(
        np.log(np.outer(w, w) / (2 * np.pi))
        + stats.multivariate_normal.logpdf(y.ravel() - latent, cov=v)
    )


def check_binary(model):
    y = np.array([[1.0, 1.8], [-0.7, 2.7]])
    alt = y + np.array([[1.4, -0.3], [-0.2, 0.25]])
    v = np.array([[[0.7, 0.11], [0.11, 0.27]], [[0.8, -0.06], [-0.06, 0.33]]])
    theta = dict(mu_b=0.4, mu_ell=2.1, tau_b=0.7, tau_ell=0.4, rho=-0.35)

    def lp(data, param):
        return float(model.log_prob(param, data, jacobian=False, sig_figs=16)["lp__"].iloc[0])

    errs = {}
    blocks = []
    for kind in ("independent", "dependent_shared_setting"):
        if kind == "dependent_shared_setting":
            cov = np.block([[v[0], np.diag([0.18, 0.07])], [np.diag([0.18, 0.07]), v[1]]])
            blocks = [dict(analysis_units=[1, 2], setting_id=[1, 1], sampling_covariance=cov)]
            expected = _latent_log_density(y, cov, theta) - _latent_log_density(alt, cov, theta)
        else:
            expected = sum(
                _latent_log_density(a, c, theta) - _latent_log_density(b, c, theta)
                for a, b, c in zip(y, alt, v, strict=True)
            )
        got = lp(stan_data(y, v, [1, 2], blocks), theta) - lp(
            stan_data(alt, v, [1, 2], blocks), theta
        )
        errs[kind] = float(abs(got - expected))
    # The density acquires a constant Jacobian under unit conversion, so compare
    # log posterior contrasts at two parameter points, transforming both priors.
    c = 0.133322
    i = np.diag([c, 1])

    def convert_theta(t):
        return {
            **t,
            "mu_b": t["mu_b"] * c,
            "tau_b": t["tau_b"] * c,
            "mu_ell": t["mu_ell"] + 2 * np.log(c),
        }

    converted = stan_data(y @ i + [0, 2 * np.log(c)], i @ v @ i, [1, 2])
    for key in ("prior_mu_b", "prior_mu_b_sd", "prior_tau_b"):
        converted[key] *= c
    converted["prior_mu_ell"] += 2 * np.log(c)
    other = {**theta, "mu_b": -0.5, "tau_b": 1.2, "rho": 0.27}
    data = stan_data(y, v, [1, 2])
    errs["transformed_units"] = abs(
        (lp(converted, convert_theta(theta)) - lp(converted, convert_theta(other)))
        - (lp(data, theta) - lp(data, other))
    )
    if max(errs.values()) > 1e-7:
        raise ValueError(f"Stan likelihood verification failed: {errs}")
    # Verify within-setting and pooled index selection using synthetic records.
    records = pd.DataFrame(
        [
            ("s1", "e2", 2, 1, "icu", 64.0),
            ("s1", "e1", 1, 2, "pft", 51.0),
            ("s1", "e1", 1, 1, "pft", 40.0),
            ("s2", "e3", 3, 1, "icu", 40.0),
        ],
        columns=[
            "patient_id",
            "encounter_id",
            "encounter_order",
            "measurement_order",
            "subgroup",
            "paco2",
        ],
    )
    for group, (a, w) in {
        "icu": ([40, 64], [1, 1]),
        "pft": ([40], [1]),
        "all": ([40], [2]),
    }.items():
        v, n = index_support(records, group)
        np.testing.assert_array_equal(v, a)
        np.testing.assert_array_equal(n, w)
    return {
        "absolute_log_density_contrast_errors": errs,
        "index_fixture": "passed",
        "prior_predictive": _prior_check(),
    }


def check_source_archive(path):
    """Reconcile every ledger row and source-class membership against the pinned RData."""
    import pyreadr

    from tcco2_accuracy.workflows.future_setting_data import ROOT, digest, load_ledger

    checksum = digest(path)
    if checksum != SPEC["figshare_rdata_sha256"]:
        raise ValueError("Expected the version-2 Conway source archive")
    objects = pyreadr.read_r(str(path))
    ledger = load_ledger().set_index("study_id")
    historical = pd.read_csv(ROOT / "Data/conway_studies.csv").set_index("study_id")
    counts = {}
    for name in ("main", "ICU", "ARF", "LFT"):
        source = objects[name].set_index("Study")
        if not source.index.is_unique:
            raise ValueError(f"Duplicate analysis units in {name}")
        selected = ledger[
            ledger.strata.str.split("|", regex=False).map(lambda x: name.lower() in x)
        ]
        if set(source.index) != set(selected.index):
            raise ValueError(f"Source membership differs for {name}")
        cols = ["archive_bias", "archive_lower", "archive_upper", "archive_s2", "archive_s2_star"]
        np.testing.assert_allclose(
            selected[cols],
            source.loc[selected.index, ["bias", "lower95", "upper95", "S2", "S2*"]],
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_array_equal(
            source[["N", "n"]], historical.loc[source.index, ["n_pairs", "n_participants"]]
        )
        repeats = source.N / source.n
        reconstructed = source.S2 * (1 + (repeats - 1) / (source.N - repeats))
        np.testing.assert_allclose(source["S2*"], reconstructed, rtol=1e-5)
        counts[name] = {
            "effect_rows": len(selected),
            "publications": selected.publication_id.nunique(),
        }
    midpoint = (ledger.archive_upper + ledger.archive_lower) / 2
    sd_from_span = (ledger.archive_upper - ledger.archive_lower) / 3.92
    return {
        "archive_sha256": checksum,
        "source_membership": counts,
        "union_rows": len(ledger),
        "row_checks": pd.DataFrame(
            {
                "analysis_unit_id": ledger.analysis_unit_id,
                "midpoint_minus_archived_bias_mmHg": midpoint - ledger.archive_bias,
                "sd_from_archive_span_minus_archived_sd_mmHg": sd_from_span
                - np.sqrt(ledger.archive_s2),
                "ledger_status": ledger.status,
                "decision": ledger.decision,
            }
        ).to_dict("records"),
    }


def check_index_handoff(run_dir):
    """Pass invented encounter records through the fixed-prior application interface."""
    from tcco2_accuracy.workflows.future_setting_inverse import (
        EmpiricalPriorMetadata,
        infer_index_cohort,
    )

    # Earliest measurements recreate the versioned ICU fixture exactly. Later visits
    # and repeat observations would produce a different prior if selected.
    records = []
    for j, value in enumerate([34, 42, 42, 50, 50, 50, 50, 58, 58, 66]):
        records.extend(
            [
                (j, "later", 2, 1, "icu", 73),
                (j, "first", 1, 2, "icu", 74),
                (j, "first", 1, 1, "icu", value),
            ]
        )
    frame = pd.DataFrame(
        records,
        columns=[
            "patient_id",
            "encounter_id",
            "encounter_order",
            "measurement_order",
            "subgroup",
            "paco2",
        ],
    )
    result = infer_index_cohort(
        run_dir,
        frame,
        target="icu",
        metadata=EmpiricalPriorMetadata(
            "invented-index-v1", "invented-eligibility", "invented-window"
        ),
    )
    expected = pd.read_csv(run_dir / "paco2_given_tcco2.csv").query("target_stratum=='icu'")
    cols = ["tcco2", "lower", "median", "upper"]
    np.testing.assert_array_equal(result[cols], expected[cols])
    np.testing.assert_allclose(result.probability_ge, expected.probability_ge, atol=1e-12, rtol=0)
    return {
        "agreement_run": result.agreement_run.iloc[0],
        "source": "invented patient and encounter records",
        "target": "icu",
        "matching_synthetic_grid_points": len(result),
    }


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Verify source reconciliation or the fixed-prior handoff."
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--archive", type=Path)
    selection.add_argument("--public-run", type=Path)
    args = parser.parse_args()
    result = (
        check_source_archive(args.archive) if args.archive else check_index_handoff(args.public_run)
    )
    print(json.dumps(result, indent=2, allow_nan=False))

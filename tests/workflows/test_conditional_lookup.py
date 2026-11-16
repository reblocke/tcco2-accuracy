"""Reference cases for the fixed-prior lookup and its conditional variance budget."""

import json

import numpy as np
import pytest
from scipy.integrate import quad_vec
from scipy.stats import norm

from tcco2_accuracy.workflows.conditional_lookup import lookup_row, validate_lookup


def test_conditional_budget_against_independent_adaptive_integration():
    # Unequal evidence across both fitted populations and latent settings.
    a = np.array([36.0, 45.0, 52.0])
    prior = np.array([0.25, 0.6, 0.15])
    t = 48
    models = [(-3.0, 3.0, 2.0), (4.0, 4.0, 1.0)]
    theta = np.tile([[mb, np.log(sd**2), tb, 0, 0.5] for mb, sd, tb in models], (4, 32, 1))
    terms = []
    for mb, sd, tb in models:

        def moments(z):
            joint = prior * norm.pdf(a - t, mb + z * tb, sd)
            den = joint.sum()
            m = joint @ a
            return norm.pdf(z) * np.array([den, m, joint @ (a * a), m * m / den if den else 0])

        terms.append(quad_vec(moments, -12, 12, epsabs=1e-13)[0])
    z, m, m2, h = np.array(terms).T
    overall = (m * m / z).sum()
    expected = [
        (m2 - h).sum() / z.sum(),
        (h.sum() - overall) / z.sum(),
        overall / z.sum() - (m.sum() / z.sum()) ** 2,
    ]
    r = lookup_row(theta, a, prior, t, order=64)
    np.testing.assert_allclose(r["variance"], expected, atol=2e-6)
    mixture = sum(norm.pdf(a - t, mb, np.hypot(sd, tb)) for mb, sd, tb in models) * prior
    expected_q = a[np.searchsorted(np.cumsum(mixture / mixture.sum()), [0.025, 0.5, 0.975])]
    np.testing.assert_equal(r["interval"], expected_q)
    assert r["tails"] == pytest.approx(
        [mixture[1:].sum() / mixture.sum(), mixture[-1] / mixture.sum()]
    )


def test_point_prior_and_extreme_offset():
    theta = np.tile([0, 0, 0, 0, 0], (4, 8, 1))
    row = lookup_row(theta, [150], [1], 20, order=16)
    assert row["interval"] == [150, 150, 150]
    assert row["tails"] == [1, 1]
    assert row["variance"] == [0, 0, 0]


def test_release_contract_rejects_private_fields_and_incomplete_grid():
    from tcco2_accuracy.workflows.future_setting_data import ROOT

    obj = json.loads((ROOT / "models/pages/synthetic-lookup.json").read_text())
    validate_lookup(obj)
    assert len(obj["groups"]["all"]["rows"]) == 81
    bad = json.loads(json.dumps(obj))
    bad["groups"]["all"]["rows"][0]["support"] = [40, 41]
    with pytest.raises(ValueError):
        validate_lookup(bad)
    # Exact distribution fields are specific to the public invented fixture.
    bad = dict(obj, source="trinetx")
    with pytest.raises(ValueError):
        validate_lookup(bad)
    obj["groups"]["icu"]["rows"].pop()
    with pytest.raises(ValueError):
        validate_lookup(obj)


def test_preview_grid_budget_against_uncompressed_quadrature():
    from scripts.export_synthetic_lookup import grid_variance
    from tcco2_accuracy.core.future_setting import PredictiveKernel

    theta = np.tile(
        [[-1.0, np.log(12), 2.0, 0.2, -0.5], [1.0, np.log(20), 2.5, 0.3, 0.4]], (4, 8, 1)
    )
    a = np.array([31.0, 41.0, 49.0, 58.0])
    w = np.array([0.1, 0.3, 0.4, 0.2])
    readings = np.array([24, 45, 80])
    kernel = PredictiveKernel(theta, order=64)
    dens = np.exp(kernel.logpdf(a[:, None] - readings))
    values = grid_variance(theta, 64, a, w, readings, dens)
    for t, v in zip(readings, values, strict=True):
        direct = lookup_row(theta, a, w, int(t), order=64)["variance"]
        if t == 80:
            # The mesh cannot resolve this small tail variance; the row must be held.
            assert v is None
            continue
        assert direct is not None and v is not None
        np.testing.assert_allclose(v, direct, atol=0.025)


def test_private_lookup_handoff_selects_index_and_writes_complete_pair(tmp_path, monkeypatch):
    import hashlib

    from scripts import conditional_lookup as cli

    # A leading zero in an identifier matters. Later measurements and later
    # encounters must not contribute to either the overall or stratum prior.
    source = tmp_path / "eligible.csv"
    source.write_text(
        "patient_id,encounter_id,encounter_order,measurement_order,subgroup,paco2\n"
        "01,01,1,2,icu,65\n01,01,1,1,icu,40\n01,02,2,1,pft,47\n"
        "1,02,1,1,ed_inp,54\n"
    )
    receipt = tmp_path / "receipt.json"
    dest = tmp_path / "private"
    dest.mkdir()
    receipt.write_text(
        json.dumps(
            dict(
                use="approved_private_index_analysis",
                private_destination=str(dest),
                eligible_extract=str(source),
                eligible_extract_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                revisions=dict(
                    cohort_revision="fixture",
                    window_revision="fixture",
                    eligibility_revision="fixture",
                ),
            )
        )
    )
    from tcco2_accuracy.pages_prediction import GROUPS

    def accepted(run, target):
        return (
            np.tile([0, np.log(16), 1.0, 0, 0.1], (4, 4, 1)),
            16,
            dict(run_id=run.name, strata={GROUPS[target]: {"posterior_sha256": "a" * 64}}),
        )

    original_receipt_hash = hashlib.sha256(receipt.read_bytes()).hexdigest()
    read_csv = cli.pd.read_csv

    def refresh_during_parse(*args, **kwargs):
        # Both paths can be refreshed during a long authorized private run.
        replacement = tmp_path / "refresh.csv"
        replacement.write_bytes(source.read_bytes().replace(b"icu,40", b"icu,39"))
        replacement.replace(source)
        receipt.write_bytes(receipt.read_bytes() + b"\n")
        return read_csv(*args, **kwargs)

    monkeypatch.setattr(cli.pd, "read_csv", refresh_during_parse)
    monkeypatch.setattr(cli, "load_candidate_draws", accepted)
    cli.private_build(receipt, tmp_path)
    folder = dest / "conditional-lookup"
    out = folder / "conditional-lookup.json"
    obj = json.loads(out.read_text())
    assert obj["groups"]["all"]["prior_interval"] == [40, 54]
    assert obj["groups"]["pft"]["rows"][25]["interval"] == [47, 47, 47]
    assert obj["groups"]["pft"]["rows"][25]["tails"] == [1, 0]
    assert obj["groups"]["icu"]["rows"][25]["variance"] == [0, 0, 0]
    audit = json.loads((folder / "receipt.json").read_text())
    assert audit["input_receipt_sha256"] == original_receipt_hash
    assert audit["lookup_sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        cli.private_build(receipt, tmp_path)

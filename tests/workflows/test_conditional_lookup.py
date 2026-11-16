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


def test_private_chart_uses_full_index_distribution_and_averages_likelihood_first(
    tmp_path, monkeypatch
):
    import hashlib

    from scripts import conditional_lookup as cli

    # A later ABG for the same patient must disappear. All includes patients whose
    # only ABG lies outside the browser kernel. The two possible fits have unequal
    # evidence, so normalizing each fit first would give different threshold odds.
    source = tmp_path / "eligible.csv"
    source.write_text(
        "patient_id,encounter_id,encounter_order,measurement_order,subgroup,paco2\n"
        "01,A,1,2,icu,21\n01,A,1,1,icu,45\n01,B,2,1,icu,52\n"
        "02,C,1,1,icu,45\n03,D,1,1,icu,52\n04,E,1,1,icu,104\n"
    )
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            dict(
                use="approved_private_index_analysis",
                private_destination=str(tmp_path),
                eligible_extract=str(source),
                eligible_extract_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                revisions=dict(
                    cohort_revision="invented",
                    eligibility_revision="invented",
                    window_revision="invented",
                ),
            )
        )
    )
    original = json.loads(receipt.read_bytes())
    receipt.write_text(json.dumps(original | {"index_policy": "first_measure_per_encounter"}))
    with pytest.raises(ValueError, match="index policy"):
        cli.private_build(receipt, tmp_path, chart_target="all")
    receipt.write_text(json.dumps(original | {"index_policy": cli.INDEX_POLICY}))
    theta = np.tile([[-3, np.log(16), 0, 0, 0], [15, np.log(64), 0, 0, 0]], (4, 32, 1))

    def accepted(run, target):
        assert target == "all"
        return theta, 16, dict(run_id=run.name, strata={"main": {"posterior_sha256": "a" * 64}})

    monkeypatch.setattr(cli, "load_candidate_draws", accepted)
    cli.private_build(receipt, tmp_path, chart_target="all")
    folder = tmp_path / "private-chart-all"
    path = folder / "private-chart.json"
    doc = json.loads(path.read_text())
    a = np.array([45.0, 52.0, 104.0])
    weights = np.array([0.5, 0.25, 0.25])
    likelihood = (norm.pdf(a - 47, -3, 4) + norm.pdf(a - 47, 15, 8)) / 2
    p = weights * likelihood / (weights @ likelihood)
    assert doc["version"] == "conditional_lookup_private_chart_v1"
    assert doc["target"] == "all" and doc["tcco2"] == 47 and doc["posterior"] == "a" * 64
    assert doc["prior_uncertainty"] == "fixed_empirical" and doc["source"] == "trinetx"
    np.testing.assert_equal(doc["support"], a)
    np.testing.assert_allclose(doc["prior"], weights)
    np.testing.assert_allclose(doc["mass"], p, atol=0)
    np.testing.assert_allclose(doc["likelihood"], likelihood / max(likelihood), atol=0)
    np.testing.assert_equal(doc["interval"], a[np.searchsorted(np.cumsum(p), [0.025, 0.5, 0.975])])
    np.testing.assert_allclose(doc["tails"], [1.0, p[1:].sum()])
    assert "patient_id" not in path.read_text() and "01,A" not in path.read_text()
    audit = json.loads((folder / "receipt.json").read_text())
    assert audit["chart_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert not (tmp_path / "conditional-lookup").exists()
    with pytest.raises(ValueError):
        validate_lookup(doc)
    with pytest.raises(FileExistsError):
        cli.private_build(receipt, tmp_path, chart_target="all")


def test_private_chart_smoothing_changes_only_the_drawing():
    from tcco2_accuracy.workflows.private_chart import private_chart

    # The exact prior is a point AT the selected cutoff. Smoothing necessarily
    # draws probability density on both sides; none may enter the reported tail.
    theta = np.zeros((4, 8, 5))
    doc = private_chart(theta, [52.0], [1], target="icu", posterior="b" * 64, order=16)
    assert doc["interval"] == [52, 52, 52] and doc["tails"] == [1, 1]
    display = doc["display"]
    x = np.array(display["x"])
    assert display["bandwidth"] == 1 and x[0] < 49 < 55 < x[-1]
    assert np.diff(x).max() <= 0.25
    for key in ("prior_density", "posterior_density"):
        np.testing.assert_allclose(display[key], norm.pdf(x, 52, 1), atol=1e-15)


def test_private_chart_holds_uncertain_quantiles_and_tails_independently():
    from tcco2_accuracy.workflows.private_chart import private_chart

    # Chains concentrate on different sides of 52, and both give exactly P(A>=45)=1.
    theta = np.zeros((4, 25, 5))
    theta[2:, :, 0] = 10
    doc = private_chart(theta, [47, 57], [1, 1], target="all", posterior="a" * 64, order=8)
    assert doc["interval"] is None and doc["tails"] == [1.0, None]
    np.testing.assert_allclose(doc["mass"], [0.5, 0.5])


@pytest.mark.parametrize("field,values", [("count", [2, 3]), ("weight", [0.4, 0.6])])
def test_private_chart_requires_attested_target_matched_index_frequencies(
    tmp_path, monkeypatch, field, values
):
    import hashlib

    from scripts import conditional_lookup as cli

    source = tmp_path / "freq.csv"
    receipt = tmp_path / "input.json"
    meta = dict(
        use="approved_private_index_analysis",
        private_destination=str(tmp_path),
        eligible_extract=str(source),
        revisions=dict(
            cohort_revision="fixture", eligibility_revision="fixture", window_revision="fixture"
        ),
        input_kind="index_frequencies",
        index_target="all",
        index_policy="earliest_eligible_measurement_in_earliest_eligible_encounter_per_patient",
    )

    def write(data, **changes):
        source.write_text(data)
        receipt.write_text(
            json.dumps(
                dict(meta, eligible_extract_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
                | changes
            )
        )

    first = "46.125000000000011"
    data = f"paco2_bin,{field}\n{first},{values[0]}\n61.25,{values[1]}\n"
    write(data, index_policy="first_measure_per_encounter")
    with pytest.raises(ValueError):
        cli.private_build(receipt, tmp_path, chart_target="all")
    if field == "count":
        write(
            data,
            index_policy="equal_patient_weight_split_across_eligible_draws_on_first_testing_day",
        )
        with pytest.raises(ValueError):
            cli.private_build(receipt, tmp_path, chart_target="all")
    write(data)
    for wrong in ("icu", None):
        with pytest.raises(ValueError):
            cli.private_build(receipt, tmp_path, chart_target=wrong)
    for bad_data in (
        data.replace("61.25", first),
        data.replace(first, "invented-A," + first).replace("61.25", "invented-B,61.25"),
        data.replace(field, "patient_id"),
        data.replace(str(values[0]) + "\n", ("1.5" if field == "count" else "0.9") + "\n"),
    ):
        write(bad_data)
        with pytest.raises(ValueError):
            cli.private_build(receipt, tmp_path, chart_target="all")
    write(data)
    monkeypatch.setattr(
        cli,
        "load_candidate_draws",
        lambda run, target: (
            np.zeros((4, 4, 5)),
            16,
            dict(run_id=run.name, strata={"main": {"posterior_sha256": "a" * 64}}),
        ),
    )
    cli.private_build(receipt, tmp_path, chart_target="all")
    folder = tmp_path / "private-chart-all"
    doc = json.loads((folder / "private-chart.json").read_text())
    np.testing.assert_equal(doc["support"], [float(first), 61.25])
    np.testing.assert_allclose(doc["prior"], [0.4, 0.6])
    audit = json.loads((folder / "receipt.json").read_text())
    assert audit["chart_sha256"] in (folder / "preview.html").read_text()

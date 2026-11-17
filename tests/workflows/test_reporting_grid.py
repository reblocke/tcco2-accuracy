"""A corrupted summary must fail before the manuscript renderer can receive it."""

import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest


def test_no_curves_export_still_checks_the_likelihood(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[2]
    scripts = root / "scripts/manuscript"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "review_reporting_grid", scripts / "reporting_grid.py"
    )
    workflow = importlib.util.module_from_spec(spec)
    renderer = types.ModuleType("exhibits")
    renderer.render = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "exhibits", renderer)
    spec.loader.exec_module(workflow)
    out = tmp_path / "output"
    (out / "cache").mkdir(parents=True)
    audit = tmp_path / "reporting-audit"
    audit.mkdir()
    screen = tmp_path / "screen"
    screen.mkdir()
    (audit / "split-validation.json").write_text(
        json.dumps({k: dict(passed=12, separate_half_cut_checks=12) for k in workflow.DRAWS})
    )
    a, w = np.array([40.0, 45.0, 52.0]), np.array([0.2, 0.5, 0.3])
    probabilities = workflow.evidence_masks(a, a, [45, 52]) @ w
    grid = []
    certificates = []
    for tc in range(20, 101):
        grid.append(
            dict(
                tc=tc,
                fine_q=a.tolist(),
                ratios=probabilities.tolist(),
                coarse=probabilities.tolist(),
                se=[0.0] * 8,
            )
        )
        certificates.append(
            dict(
                t=tc,
                q=a.tolist(),
                limits=[[v, v] for v in a],
                tail_radius_pp=[0.0, 0.0],
                tail_quad_pp=[0.0, 0.0],
            )
        )
    # Shift both quadrature summaries together. Their discrepancy remains zero.
    grid[0]["ratios"][0] += 0.01
    grid[0]["coarse"][0] += 0.01
    (screen / "precision.json").write_text(json.dumps(grid))
    for setting, n in workflow.DRAWS.items():
        cert = audit / f"{setting}-{n}-private-certificates.json"
        cert.write_text(json.dumps(certificates))
        monkeypatch.setitem(
            workflow.CERTIFICATES, setting, hashlib.sha256(cert.read_bytes()).hexdigest()
        )
    monkeypatch.setattr(
        workflow, "inputs", lambda *_: (None, dict(posterior_sha256="synthetic"), a, w, screen)
    )
    for setting in workflow.DRAWS:
        np.savez(
            out / "cache" / f"{setting}.npz",
            x=np.arange(-60, 33, dtype=float),
            density=np.ones(93),
            posterior="synthetic",
        )
    rendered = []
    monkeypatch.setattr(workflow, "render", lambda *args, **kwargs: rendered.append(True))
    with pytest.raises(AssertionError, match="Not equal"):
        workflow.build(tmp_path, out, curves=False)
    assert not rendered
    assert not list(out.glob("*-predictions.csv"))

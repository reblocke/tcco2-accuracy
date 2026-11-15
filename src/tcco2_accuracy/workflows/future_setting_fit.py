"""Pinned offline fitter. This module is outside the Pyodide staging allowlist."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

import numpy as np

from ..core.future_setting import PARAMETERS
from .future_setting_data import MODEL_DIR, ROOT, SPEC, digest

SEED = 88032
FILES = [
    *sorted(MODEL_DIR.glob("*")),
    Path(__file__).with_name("future_setting_data.py"),
    ROOT / "Data/conway_studies.csv",
    ROOT / "tests/fixtures/conway_table1.csv",
    ROOT / "tests/fixtures/synthetic_paco2_prior.csv",
]


def _git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def evidence_revision():
    files = {}
    for p in FILES:
        rel = str(p.relative_to(ROOT))
        current = digest(p)
        tracked = subprocess.check_output(["git", "-C", str(ROOT), "show", "HEAD:" + rel])
        if hashlib.sha256(tracked).hexdigest() != current:
            raise ValueError(f"Commit the approved input before fitting: {rel}")
        files[rel] = current
    return {"git_commit": _git("rev-parse", "HEAD"), "input_sha256": files}


def offline_model():
    import cmdstanpy

    if cmdstanpy.__version__ != SPEC["cmdstanpy"]:
        raise ValueError("Use the locked fitting dependency group")
    default = Path.home() / ".cache" / "tcco2-cmdstan"
    stan = Path(os.environ.get("CMDSTAN", default / ("cmdstan-" + SPEC["cmdstan"]))).resolve()
    archive = stan.parent / ("cmdstan-" + SPEC["cmdstan"] + ".tar.gz")
    if not archive.is_file() or digest(archive) != SPEC["archive_sha256"]:
        raise ValueError(
            "Supply the checksum-pinned CmdStan release archive beside its installation"
        )
    cmdstanpy.set_cmdstan_path(str(stan))
    if cmdstanpy.cmdstan_version() != tuple(map(int, SPEC["cmdstan"].split(".")[:2])):
        raise ValueError("CmdStan version mismatch")
    # Use a content-addressed compilation directory without workspace-path spaces.
    source = MODEL_DIR / "model.stan"
    cache = stan.parent / ("tcco2-" + digest(source)[:16])
    cache.mkdir(exist_ok=True)
    cached = cache / source.name
    if not cached.exists() or digest(cached) != digest(source):
        shutil.copyfile(source, cached)
    model = cmdstanpy.CmdStanModel(stan_file=str(cached))
    info = model.exe_info()
    if (
        ".".join(
            str(info[k]) for k in ("stan_version_major", "stan_version_minor", "stan_version_patch")
        )
        != SPEC["cmdstan"]
    ):
        raise ValueError("Compiled model toolchain version mismatch")
    return model, {
        "cmdstan": SPEC["cmdstan"],
        "cmdstanpy": cmdstanpy.__version__,
        "archive_sha256": digest(archive),
        "model_binary_sha256": digest(model.exe_file),
        "platform": platform.platform(),
        "compiler": subprocess.check_output(["c++", "--version"], text=True).splitlines()[0],
        "python": platform.python_version(),
        "numpy": np.__version__,
    }


def sample_attempt(model, data, path, retained):
    import cmdstanpy

    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    fit = model.sample(
        data=data,
        seed=SEED,
        chains=4,
        parallel_chains=4,
        chain_ids=[1, 2, 3, 4],
        iter_warmup=1000,
        iter_sampling=retained,
        adapt_delta=0.95,
        output_dir=str(path),
        sig_figs=16,
        show_progress=False,
        refresh=1000,
    )
    # Preserve chain × draw order. Reject unrelated posterior column sets.
    cols = list(fit.column_names)
    theta = np.moveaxis(fit.draws()[:, :, [cols.index(p) for p in PARAMETERS]], 0, 1)
    summary = fit.summary()
    diagnostics = summary.loc[list(PARAMETERS), ["R_hat", "ESS_bulk", "ESS_tail"]].to_dict("index")
    div = int(fit.method_variables()["divergent__"].sum())
    ok = div == 0 and all(
        v["R_hat"] < 1.01 and min(v["ESS_bulk"], v["ESS_tail"]) >= 400 for v in diagnostics.values()
    )
    np.savez_compressed(
        path / "posterior.npz",
        theta=theta,
        parameters=PARAMETERS,
        method_version=SPEC["method_version"],
        chain_ids=fit.chain_ids,
        draw_ids=np.arange(1, retained + 1),
    )
    receipt = {
        "sampler_pass": bool(ok),
        "diagnostics": diagnostics,
        "divergences": div,
        "treedepth_hits": int((fit.method_variables()["treedepth__"] >= 10).sum()),
        "seed": SEED,
        "chains": 4,
        "retained_per_chain": retained,
        "warmup_per_chain": 1000,
        "adapt_delta": 0.95,
        "posterior_sha256": digest(path / "posterior.npz"),
        "cmdstanpy": cmdstanpy.__version__,
    }
    (path / "sampler.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return theta, receipt

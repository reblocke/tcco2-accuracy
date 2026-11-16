"""Export the four accepted public fits to compact, independently checked Pages assets."""

import csv
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import ndtr

from tcco2_accuracy.core.future_setting import PredictiveKernel
from tcco2_accuracy.workflows.future_setting_data import ROOT, SPEC, digest
from tcco2_accuracy.workflows.future_setting_inverse import load_candidate_draws

BASE = ROOT / ".pytest_tmp/future-setting"
DEST = ROOT / "models/pages"
GRID = np.arange(-48, 48.25, 0.25)


def chain_grid(chain, order):
    size = int(np.sqrt(len(chain)))
    total = np.zeros((3, len(GRID)))
    batches = []
    for start in range(0, len(chain), size):
        rows = chain[start : start + size]
        batch = np.zeros_like(total)
        for q, index in [(order, 0), (order // 2, 1)]:
            k = PredictiveKernel(rows[None, :, :], order=q)
            inv = 1 / k.sd.ravel()
            mu = k.mean.ravel()
            w = np.broadcast_to(k.weights, k.sd.shape).ravel()
            scale = w * inv / np.sqrt(2 * np.pi)
            for j, x in enumerate(GRID):
                z = (x - mu) * inv
                batch[index, j] = np.sum(scale * np.exp(-0.5 * z * z))
                if index == 0:
                    batch[2, j] = np.sum(w * ndtr(z))
        total += batch
        if len(rows) == size:
            batches.append(batch[0] / size)
    return total, np.array(batches)


def checked_rows(folder, name, manifest, group):
    path = folder / name
    if digest(path) != manifest["output_sha256"][name]:
        raise ValueError(f"Changed run output {name}")
    with path.open() as f:
        return [row for row in csv.DictReader(f) if row["stratum"] == group]


def forward_variance(folder, manifest, group):
    [row] = checked_rows(folder, "variance_bookkeeping.csv", manifest, group)
    parts = [
        float(row[key])
        for key in (
            "expected_within_variance",
            "expected_between_bias_variance",
            "variance_of_overall_bias",
        )
    ]
    if row["unit"] != "mmHg^2" or not np.isclose(sum(parts), float(row["total"])):
        raise ValueError("Invalid forward variance decomposition")
    return parts


def main():
    DEST.mkdir(exist_ok=True)
    summary = {}
    for target, group in SPEC["target_mapping"].items():
        folder = BASE / ("public-main-simon-2003" if group == "main" else "public-adjudicated-v2")
        theta, order, m = load_candidate_draws(folder, target)
        with ThreadPoolExecutor(max_workers=4) as pool:
            chains = list(pool.map(lambda chain: chain_grid(chain, order), theta))
        fine, coarse, cdf = sum(c[0] for c in chains) / np.prod(theta.shape[:2])
        batches = np.concatenate([c[1] for c in chains])
        knots = GRID[::2]
        logs = np.log(np.vstack([fine, coarse, batches]))
        err = np.abs(CubicSpline(knots, logs[:, ::2], axis=1)(GRID[1::2]) - logs[:, 1::2]).max()
        if err > 2e-5:
            raise ValueError(f"{group}: log density interpolation error {err}")
        asset = DEST / f"{group}.npz"
        np.savez_compressed(
            asset,
            x=knots,
            logpdf=logs[:2, ::2],
            batch_logpdf=logs[2:, ::2].astype("float32"),
            cdf=cdf[::2],
            group=group,
        )
        [pred] = checked_rows(folder, "predictive_disagreement.csv", m, group)
        bridge = checked_rows(folder, "interval_bridge.csv", m, group)
        summary[group] = dict(
            run=m["run_id"],
            ledger=pred["input_revision"],
            method=m["method_version"],
            status="candidate",
            posterior=m["strata"][group]["posterior_sha256"],
            kernel_sha256=digest(asset),
            max_checked_logpdf_error=float(err),
            max_checked_cdf_error=float(
                np.max(np.abs(CubicSpline(knots, cdf[::2])(GRID[1::2]) - cdf[1::2]))
            ),
            quantiles=[float(pred[k]) for k in ["lower", "median", "upper"]],
            variance=forward_variance(folder, m, group),
            intervals=[
                {
                    **{
                        k: r[k]
                        for k in [
                            "method",
                            "quantity_type",
                            "membership",
                            "unit",
                            "result_status",
                            "input_revision",
                        ]
                    },
                    "lower": float(r["lower"]),
                    "upper": float(r["upper"]),
                }
                for r in bridge
            ],
        )
        print(group, asset.stat().st_size, err, flush=True)
    (DEST / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()

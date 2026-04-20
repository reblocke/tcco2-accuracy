# TcCO2 Accuracy — Decisions

## Open decisions
- None yet.

## Workflow and monorepo stabilization
- The repository remains a monorepo for this wave: Python package, static browser app, source/reference
  data, artifacts, and manuscript drafts stay in one repository with nested `AGENTS.md` guardrails
  where rules differ.
- `pyproject.toml` and `uv.lock` are authoritative for Python dependencies. Use `uv sync --locked`
  and the root `Makefile` command surface for local and CI verification.
- The active package/test layout is root `src/` and root `tests/`; the prior nested `python/`
  package layout has been retired.
- The public app deployment target is static GitHub Pages. See `docs/adr/0001-streamlit-to-pages.md`.
- Browser computation loads staged Python through Pyodide in a worker. JavaScript manages UI,
  uploads, and plotting but does not duplicate the statistical model.
- Browser default inference uses staged canonical assets: `Data/conway_studies.csv`,
  `Data/paco2_prior_bins.csv`, and `artifacts/bootstrap_params.csv`.
- Offline PaCO2 distribution loading accepts `Data/In Silico TCCO2 Database.dta` and the local alias
  `Data/in_silico_tcco2_db.dta`; the static app still uses binned prior assets instead of `.dta`.
- Pyodide 0.29.0, Plotly.js 2.35.2, and SheetJS 0.18.5 are pinned CDN browser dependencies.
- User-entered values and uploads remain client-side; the app has no backend, telemetry,
  persistence, or patient-value URL state.
- Generated `.pytest_tmp/`, `.tmp/`, and `*.egg-info/` outputs are not source artifacts and should
  not be tracked.
- Generated static app assets under `web/assets/py/` and `web/assets/data/` are not tracked; Pages,
  E2E tests, and local serving regenerate them from `src/`, `Data/`, and `artifacts/` with
  `scripts/stage_web_python.py`.
- The browser posterior chart uses a posterior-focused x-axis to keep the displayed distribution
  readable when the prior has a long tail. This does not change posterior/prior arrays or numeric
  summaries.
- Pure numerical code is separated under `src/tcco2_accuracy/core/`; top-level modules remain
  compatibility wrappers for existing public imports.
- Malformed continuity-ledger paths are retired; durable project decisions belong in this file or
  `docs/adr/`.

## Logged divergences
- `logs2` inputs are recomputed as `log10(S2*) + 1/(n_2 - 1)` with `v_logs2 = 2/(n_2 - 1)` to match `Conway Meta/data.dta`, even though `Code/3_tcco2_uncertainty_and_simulation_do.do:94` describes `logs2` as a natural log.
- Bootstrap τ² draws are truncated at 0 to enforce non-negative between-study variance for simulation/inference draws in `src/tcco2_accuracy/core/bootstrap.py`.
- Bootstrap workflows default to `cluster_plus_withinstudy` to align outer CI scale with Conway, while low-level bootstrap functions default to `cluster_only` in `src/tcco2_accuracy/workflows/bootstrap.py` and `src/tcco2_accuracy/core/bootstrap.py`.
- Hybrid bootstrap perturbations treat bias and log-variance inputs as independent due to missing covariance data in `src/tcco2_accuracy/core/bootstrap.py`.
- Meta-analysis τ² defaults to untruncated values to reproduce Conway Table 1, but if the method-of-moments denominator is non-finite or zero we set τ² = 0 for stability in `src/tcco2_accuracy/core/conway_meta.py`.
- LoA confidence intervals are undefined for single-study summaries, so CI bounds are returned as NaN when df ≤ 0 in `src/tcco2_accuracy/core/conway_meta.py`.
- Main-analysis descriptive counts aggregate by `study_base` (strip trailing parentheses) and treat identical-bias multi-row citations as overlapping cohorts (use max counts) in `src/tcco2_accuracy/core/conway_meta.py`.
- PaCO2 subgroup assignment follows `docs/SPEC.md:20-27`, which includes ED in `ed_inp` by construction; this differs from `Code/2_trinetx_cleaning_do.do:8-13`, where `ed_inp_group` excludes ED (`is_emer==0`) and requires `cc_time==0`.
- PaCO2 subgroups map to Conway bootstrap groups via the shared helper in `src/tcco2_accuracy/core/_params.py` (pft→lft, ed_inp→arf, icu→icu, all→main) to align ambulatory and acute respiratory failure sub-analyses.
- Simulation/inference parameter validation requires finite numeric values with non-negative σ² and τ² in `src/tcco2_accuracy/core/utils.py`.
- When subgroup-specific parameters are missing, simulation and inference use the shared selector in `src/tcco2_accuracy/core/_params.py` and fall back to all parameters (with a warning) rather than dropping the subgroup.
- `format_inference_demo` only supports a single threshold and raises a ValueError otherwise in `src/tcco2_accuracy/workflows/infer.py`.
- Conway study exports read bias/S2 and subgroup membership from the RData objects (`main`, `ICU`, `ARF`, `LFT`) and merge counts from `data.dta` (or `data_counts.csv` fallback); the Bolliger ICU row uses a 49/49/1 count fallback with bias/S2 pulled from `ICU` when absent from `main` to preserve Table 1 reproduction (`scripts/export_conway_rdata.py:29`, `scripts/export_conway_rdata.py:107`).

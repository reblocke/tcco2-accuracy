# Codex AGENTS

## Purpose
- This repository contains the TcCO2 to PaCO2 agreement meta-analysis, uncertainty simulation, static GitHub Pages app, validation artifacts, and manuscript support files.
- Keep it as a stabilized monorepo. Do not split app, analysis, data, or manuscript assets in this wave.
- The importable Python package is `tcco2_accuracy` under `src/`.

## Repo Map
- `src/tcco2_accuracy/` - Python package, numerical core, workflow modules, browser contract, and UI API.
- `tests/` - package, workflow, staging, browser-contract, and E2E tests.
- `web/` - static GitHub Pages app; staged Python lives under `web/assets/py/`.
- `scripts/stage_web_python.py` - stages Python and data assets for the browser app.
- `Data/` and `Conway Meta/` - source/reference data and supplements.
- `artifacts/` - small derived outputs intended for review/manuscript workflows.
- `Drafts/` - manuscript and presentation drafts; edit only when explicitly requested.
- `Code/` - Stata reference code; `Code/Legacy/` is read-only reference material.
- `.agents/skills/` - focused local workflows for recurring agent tasks.

## Commands
- Install locked environment: `uv sync --locked`
- Stage browser assets: `make stage-web`
- Python tests: `make test`
- Browser E2E: `make e2e`
- Full local gate: `make verify`
- Serve static app: `make serve`
- Rebuild artifacts: `uv run python scripts/rebuild_artifacts.py --out artifacts`

## Authority
1. Conway Thorax 2019 paper and supplementary methods/code in `Conway Meta/` and `Data/`.
2. `docs/SPEC.md`, `docs/VALIDATION.md`, `docs/CONWAY_DATA_SCHEMA.md`, and `docs/DECISIONS.md`.
3. Stata code in `Code/` as reference only; `Code/Legacy/` may contain bugs.
4. Existing Python code, tests, browser contract tests, staged app assets, and artifacts.

When Stata conflicts with the paper or docs, implement the paper/docs and record the divergence in `docs/DECISIONS.md` or an ADR.

## Working Rules
- Before non-trivial edits, state assumptions, ambiguities, tradeoffs, a brief plan, risks, and verification commands.
- Keep changes small and directly tied to the request; do not make drive-by refactors.
- Do not modify `Code/Legacy/` or `Drafts/` unless the user explicitly asks for that file family.
- Do not commit patient-level or large raw extracts. Use small, de-identified fixtures under `tests/fixtures/`.
- Keep core math pure and isolate I/O in package I/O, workflow modules, or the browser contract.
- Python remains the single numerical source of truth. JavaScript may manage UI state, uploads, worker calls, and plotting, but must not reimplement the statistical model.
- Browser app changes must keep user-entered values client-side: no backend, telemetry, persistence, or PHI-bearing URLs unless explicitly approved.
- Keep `pyproject.toml` and `uv.lock` authoritative for dependencies.
- Generated scratch outputs belong in `.pytest_tmp/` or `.tmp/` and should not be tracked.

## Skill Triggers
- Planning a non-trivial change: `.agents/skills/implementation-strategy/SKILL.md`.
- Verifying a code change: `.agents/skills/code-change-verification/SKILL.md`.
- Updating docs after behavior/workflow changes: `.agents/skills/docs-sync/SKILL.md`.
- Static Pages/Pyodide/browser contract work: `.agents/skills/static-browser-pyodide-verification/SKILL.md`.
- Reviewing numerical/statistical behavior: `.agents/skills/scientific-validation/SKILL.md`.
- Reviewing clinical, privacy, public-copy, provenance, or app surfaces: use the matching focused skill in `.agents/skills/`.

## Done Criteria
- Relevant tests pass locally.
- `make stage-web` succeeds after browser-contract or package changes.
- Browser contract outputs match Python reference behavior for canonical cases.
- Validation targets and small artifacts are updated when behavior changes.
- Decisions, divergences, browser-runtime choices, and data provenance changes are documented.
- The final report names changed files, verification commands, warnings, skips, and remaining risks.

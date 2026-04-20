# Codex AGENTS

## Purpose
- This repository contains the TcCO2 to PaCO2 agreement meta-analysis, uncertainty simulation, Streamlit app, validation artifacts, and manuscript support files.
- Keep it as a stabilized monorepo for now. Do not split app, analysis, data, or manuscript assets in this wave.
- The importable Python package is `tcco2_accuracy` under `python/src/`.

## Repo Map
- `python/src/tcco2_accuracy/` - Python package, numerical core, workflow modules, and UI API.
- `python/tests/` - package and workflow tests.
- `app/` and `streamlit_app.py` - Streamlit entry points.
- `Data/` and `Conway Meta/` - source/reference data and supplements.
- `artifacts/` - small derived outputs intended for review/manuscript workflows.
- `Drafts/` - manuscript and presentation drafts; edit only when explicitly requested.
- `Code/` - Stata reference code; `Code/Legacy/` is read-only reference material.
- `.agents/skills/` - focused local workflows for recurring agent tasks.

## Commands
- Install with pip: `python3 -m pip install -r requirements.txt -r requirements-dev.txt`
- Ephemeral test run without adopting uv: `uv run --no-project --with-requirements requirements.txt --with-requirements requirements-dev.txt pytest -q`
- Tests in an active environment: `pytest -q`
- Rebuild artifacts: `python scripts/rebuild_artifacts.py --out artifacts`
- Streamlit app: `streamlit run streamlit_app.py`

## Authority
1. Conway Thorax 2019 paper and supplementary methods/code in `Conway Meta/` and `Data/`.
2. `docs/SPEC.md`, `docs/VALIDATION.md`, `docs/CONWAY_DATA_SCHEMA.md`, and `docs/DECISIONS.md`.
3. Stata code in `Code/` as reference only; `Code/Legacy/` may contain bugs.
4. Existing Python code, tests, and artifacts.

When Stata conflicts with the paper or docs, implement the paper/docs and record the divergence in `docs/DECISIONS.md` or an ADR.

## Working Rules
- Before non-trivial edits, state assumptions, ambiguities, tradeoffs, a brief plan, risks, and verification commands.
- Keep changes small and directly tied to the request; do not make drive-by refactors.
- Do not modify `Code/Legacy/` or `Drafts/` unless the user explicitly asks for that file family.
- Do not commit patient-level or large raw extracts. Use small, de-identified fixtures under `python/tests/fixtures/`.
- Keep core math pure and isolate I/O in package I/O or workflow modules.
- Keep `requirements.txt` and `requirements-dev.txt` authoritative for this wave; do not add or commit `uv.lock` unless a later dependency-normalization decision adopts uv.
- Generated scratch outputs belong in `.pytest_tmp/` or `.tmp/` and should not be tracked.

## Skill Triggers
- Planning a non-trivial change: `.agents/skills/implementation-strategy/SKILL.md`.
- Verifying a code change: `.agents/skills/code-change-verification/SKILL.md`.
- Updating docs after behavior/workflow changes: `.agents/skills/docs-sync/SKILL.md`.
- Preparing PR text: `.agents/skills/pr-draft-summary/SKILL.md`.
- Reviewing numerical/statistical behavior: `.agents/skills/scientific-validation/SKILL.md`.
- Reviewing clinical, privacy, public-copy, provenance, or app surfaces: use the matching focused skill in `.agents/skills/`.

## Done Criteria
- Relevant tests pass locally.
- Validation targets and small artifacts are updated when behavior changes.
- Decisions, divergences, and data provenance changes are documented.
- The final report names changed files, verification commands, warnings, skips, and remaining risks.

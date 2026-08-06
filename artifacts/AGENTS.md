# Artifacts Instructions

- Files here are small derived outputs intended for review, validation, or manuscript support.
- Prefer deterministic rebuilds through `scripts/rebuild_artifacts.py`.
- Do not hand-edit generated tables unless explicitly requested; update the generating code instead.
- When artifact content changes, verify the relevant workflow and summarize the before/after implication.
- Large or patient-level outputs do not belong here.
- Do not track exact-count or reconstructable restricted-derived PaCO2 outputs. Restricted rebuilds
  belong under `.pytest_tmp/`, `.tmp/`, or an explicitly approved external private workspace.
- `artifacts/STATUS.md` and `docs/data_release_contract.json` define the public-release allowlists.
  Retained rounded/aggregate downstream files are frozen historical comparators, not
  release-approved outputs.

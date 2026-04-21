# Data Instructions

- Treat files here as source or reference data unless clearly documented otherwise.
- Do not edit raw/reference data in place; write cleaned or derived outputs separately.
- Do not commit patient-level or large raw extracts.
- Keep `Data/paco2_public_prior.csv` public and weight-only; do not add exact count columns.
- Keep exact count-bearing outputs such as `Data/paco2_prior_bins.csv` local/generated unless explicitly approved.
- Any new external artifact needs provenance: source, retrieval date, license or access terms, and transformation notes.
- Maintain `Data/PROVENANCE.md` and cross-check public/restricted boundaries against `docs/DATA_GOVERNANCE.md`.
- If a fixture is needed for tests, keep it small, de-identified, and under `tests/fixtures/`.

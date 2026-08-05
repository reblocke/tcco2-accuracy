# Data Instructions

- Treat files here as source or reference data unless clearly documented otherwise.
- Do not edit raw/reference data in place; write cleaned or derived outputs separately.
- Do not commit patient-level or large raw extracts.
- Do not track `Data/paco2_public_prior.csv`, `Data/paco2_prior_bins.csv`, or other exact or
  reconstructable restricted-derived PaCO2 distributions. Normalized weights are not automatically
  safe merely because count columns were removed.
- Generate restricted-derived outputs only under `.pytest_tmp/`, `.tmp/`, or an explicitly approved
  external private workspace, never as a candidate public asset without release review.
- Any new external artifact needs provenance: source, retrieval date, license or access terms, and transformation notes.
- Maintain `Data/PROVENANCE.md` and cross-check public/restricted boundaries against `docs/DATA_GOVERNANCE.md`.
- Follow `docs/data_release_contract.json` and complete
  `docs/restricted_data_provenance.template.json` before restricted-data use or release review;
  unresolved fields remain `HUMAN REVIEW REQUIRED`.
- If a fixture is needed for tests, keep it small, de-identified, and under `tests/fixtures/`.

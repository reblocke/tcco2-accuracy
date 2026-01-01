Goal (incl. success criteria):
- Make CI/Streamlit robust without large .dta; pytest passes; exporter succeeds in CI.

Constraints/Assumptions:
- Follow Conway/spec/docs; do not modify Code/Legacy or Drafts.
- Default prior bins CSV shipped; prefer bins CSV then .dta fallback.
- Run pytest at end; update artifacts; print git add/commit commands.

Key decisions:
- Use CSV fallback for Conway exporter counts (avoid committing large .dta).

State:
- Changes implemented; pytest passed; git add/commit blocked by .git/index.lock permissions.

Done:
- Located repo; checked for AGENTS; CONTINUITY created.
- Added default prior loader/comments and CSV counts fallback; updated tests/docs/artifacts.
 - Created `Data/data_counts.csv` and ran `pytest -q` (48 passed, 2 warnings).
 - Re-ran `pytest -q` after counts CSV update (48 passed, 2 warnings).
 - Attempted `git add`; failed due to inability to create `.git/index.lock`.

Now:
- Provide git add/commit commands for user to run.

Next:
- Report summary and commands.

Open questions (UNCONFIRMED if needed):
- None.

Working set (files/ids/commands):
- http/CONTINUITY.md
- python/src/tcco2_accuracy/data.py
- python/tests/test_paco2_distribution.py
- python/tests/test_simulation.py
- scripts/export_conway_rdata.py
- Data/data_counts.csv
- docs/DECISIONS.md
- artifacts/ui_overview.md

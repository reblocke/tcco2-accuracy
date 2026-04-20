# Python Package Instructions

- Keep importable code under `python/src/tcco2_accuracy/` and tests under `python/tests/`.
- Keep numerical/statistical functions pure where practical; isolate file I/O in `io.py`, `data.py`, scripts, or workflow modules.
- Preserve Conway difference direction: PaCO2 minus TcCO2.
- Parameter uncertainty should continue to use route-1 study-level bootstrap unless a documented decision changes that.
- Add or update tests for any change to model assumptions, input validation, intervals, bootstrap behavior, artifact tables, or public API payloads.
- Use `docs/DECISIONS.md` for divergences from Conway, Stata, or prior validation behavior.

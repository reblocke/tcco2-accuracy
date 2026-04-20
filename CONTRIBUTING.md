# Contributing to TcCO₂ Accuracy

This repository supports reproducible scientific analysis. Contributions are welcome if they improve
correctness, transparency, and reproducibility.

## Scope

Suitable contributions include:
- Bug fixes, numerical correctness improvements, and robustness checks
- Reproducibility upgrades (pinned dependencies, deterministic behavior, CI smoke tests)
- Documentation improvements (README clarity, figure/table provenance)
- New analyses only if they are clearly separated and do not change existing primary results without discussion

## Ground rules

- Do not add or commit any protected health information (PHI) or other restricted patient-level data.
- Prefer small, reviewable pull requests.
- Maintain a clear mapping from scripts to outputs (figures/tables) and keep outputs out of version control unless
  they are intentionally archived artifacts.

## Development setup

### Python
1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   . .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```
2. Install in editable mode:
   ```bash
   pip install -e .
   ```
   If editable install is not configured yet, install the requirements listed in `pyproject.toml` manually.

### Stata
- Use Stata 17+ if possible.
- Keep `.do` scripts runnable from the repository root (use relative paths).

## Testing

Run the package tests before opening a PR:

```bash
pytest -q
```

If `pytest` is not installed in the active shell, use the documented ephemeral check:

```bash
uv run --no-project --with-requirements requirements.txt --with-requirements requirements-dev.txt pytest -q
```

For workflow or artifact changes, also describe any generated outputs and the exact rebuild command used.

## Style

- Python: follow PEP8, prefer type hints for public functions, and keep I/O at the edges.
- Stata: keep `.do` files sectioned and comment assumptions; avoid hard-coded absolute paths.

## Submitting a pull request

1. Fork the repository and create a feature branch.
2. Make your change with minimal diffs.
3. Update documentation if behavior or outputs change.
4. Open a PR with:
   - what changed
   - why it changed
   - how to reproduce (exact command)
   - any expected output diffs

## Code of Conduct

This project follows the Contributor Covenant. See `CODE_OF_CONDUCT.md`.

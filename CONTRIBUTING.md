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
- Do not commit exact-count or reconstructable restricted-derived summaries, including normalized
  PaCO2-bin weights. Removing a count column does not by itself make a distribution public-safe.
- Follow `docs/data_release_contract.json`; restricted outputs may be written only to
  `.pytest_tmp/`, `.tmp/`, or an explicitly approved external private workspace. Complete
  `docs/restricted_data_provenance.template.json` before proposing any restricted-data release.
- Prefer small, reviewable pull requests.
- Maintain a clear mapping from scripts to outputs (figures/tables) and keep outputs out of version control unless
  they are intentionally archived artifacts.

## Development setup

### Python
Install the locked environment from the repository root:

```bash
uv sync --locked
```

### Stata
- Use Stata 17+ if possible.
- Keep `.do` scripts runnable from the repository root (use relative paths).

## Testing

Run the local verification gate before opening a PR:

```bash
make verify
```

For a narrower Python-only check:

```bash
make test
```

For workflow or artifact changes, also describe any generated outputs and the exact rebuild command
used. Do not regenerate or promote frozen PaCO2-dependent outputs without explicit scientific and
governance approval.

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

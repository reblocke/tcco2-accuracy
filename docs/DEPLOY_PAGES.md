# GitHub Pages Deployment

## Deployment Model
The public app is a static GitHub Pages site served from the `web/` directory.
The Pages workflow stages Python and data assets during CI, uploads `web/` as the
Pages artifact, and deploys it from `main`.

## Local Parity
```bash
uv sync --locked
make stage-web
make serve
```

Open http://127.0.0.1:8000 and verify the default likelihood-only calculation completes.

## GitHub Setup
1. Enable GitHub Pages for the repository.
2. Set the Pages source to GitHub Actions.
3. Push to `main`.
4. Confirm `.github/workflows/pages.yml` completes successfully.

## Runtime Notes
- Pyodide, Plotly.js, and SheetJS are pinned CDN dependencies loaded by the browser.
- User-entered values and uploads are processed client-side only.
- The default app does not require `Data/In Silico TCCO2 Database.dta`.
- Pages stages only `conway_studies.csv` and `bootstrap_params.csv` as data assets. It must not
  contain or fetch `paco2_public_prior.csv`, exact prior bins, or another restricted-derived prior.
- Prior-weighted inference requires an explicit user upload and fails closed without one.
- `docs/data_release_contract.json` is the machine-readable current-tree, Pages, and public-history
  release contract. Pages runs the public-history check before staging deployment assets.
- If Pyodide CDN availability is a deployment concern, vendor Pyodide assets in a follow-up decision and update `docs/DECISIONS.md`.

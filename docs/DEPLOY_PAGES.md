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

Open http://127.0.0.1:8000 and select the synthetic example at TcCO₂ 47 mmHg. Verify the overall PaCO₂ interval of 38.75 to 56.25 mmHg and inclusive probabilities of 71.61% at 45 and 13.43% at 52.

## GitHub Setup
1. Enable GitHub Pages for the repository.
2. Set the Pages source to GitHub Actions.
3. Push to `main`.
4. Confirm `.github/workflows/pages.yml` completes successfully.

## Runtime Notes
- Pyodide and SheetJS are pinned CDN dependencies loaded by the browser.
- User-entered values and uploads are processed client-side only.
- The default app does not require `Data/In Silico TCCO2 Database.dta`.
- Pages stages the four public kernels, comparison summary and synthetic lookup as data assets. It must not
  contain or fetch `paco2_public_prior.csv`, exact prior bins, or another restricted-derived prior.
- Private chart JSON contains the empirical distribution and stays outside Pages. Staging accepts a real summary lookup only when its exact SHA-256 is approved in the release contract and its data path is allowlisted. Default staging has no TriNetX lookup.
- `docs/data_release_contract.json` is the machine-readable current-tree, Pages, and public-history
  release contract. Pages runs the public-history check before staging deployment assets.
- If Pyodide CDN availability is a deployment concern, vendor Pyodide assets in a follow-up decision and update `docs/DECISIONS.md`.

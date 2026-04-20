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

Open http://127.0.0.1:8000 and verify the default calculation completes.

## GitHub Setup
1. Enable GitHub Pages for the repository.
2. Set the Pages source to GitHub Actions.
3. Push to `main`.
4. Confirm `.github/workflows/pages.yml` completes successfully.

## Runtime Notes
- Pyodide, Plotly.js, and SheetJS are pinned CDN dependencies loaded by the browser.
- User-entered values and uploads are processed client-side only.
- The default app does not require `Data/In Silico TCCO2 Database.dta`.
- If Pyodide CDN availability is a deployment concern, vendor Pyodide assets in a follow-up decision and update `docs/DECISIONS.md`.

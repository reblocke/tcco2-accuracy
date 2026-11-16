# Future-setting CO₂ calculator

The initial HTML compares five 95% intervals for each source stratum. The default view shows the median and central interval of arterial minus transcutaneous CO₂ in one new paired observation. The conditional view takes a positive observed TcCO₂, a positive threshold, and a local fixed empirical prior in CSV/XLSX format. Supply group, paco2_bin and weight for all four groups. The existing count column takes precedence when present. All positive-weight support points enter the calculation, including values outside the chart's visible resolution.

The estimator uses the accepted overall run after Simon adjudication and the three accepted subgroup fits from the previous public run. Staging uses the reviewed summary and the four per-stratum kernels in `models/pages/`. Each kernel contains log densities under quadrature orders 64 and 32, a CDF, and fine-order log densities averaged within contiguous chain batches. The exporter retains all draws in the marginal densities. It checks cubic interpolation at the intermediate quarter-mmHg grid points before storing the half-mmHg knots. The browser verifies the selected binary's checksum before computing. The 95% agreement summaries come directly from the accepted fit.

If any positive-mass offset falls outside −48 to +48 mmHg, the browser withholds the whole conditional result. This is the numerical extent of the browser asset. For the full posterior, use the accepted run directory for the requested setting and your local prior file:

```sh
uv run python scripts/future_setting_prior.py .pytest_tmp/future-setting/public-main-simon-2003 prior.csv all 45 --threshold 45
```

Use `public-adjudicated-v2` for `icu`, `ed_inp`, and `pft`. Both evaluators require resolved discrete quantiles and adequate Monte Carlo and quadrature precision. Priors are processed locally. The public assets contain only literature-derived agreement information. The legacy custom-study and bootstrap workflows are available in the offline Python package.

Run `make serve` for a local preview, `make verify` for contract and browser tests, and `make visual-qa` for desktop and mobile screenshots and measured load times. To regenerate the four public assets from their accepted run receipts, run `uv run python scripts/export_pages_predictions.py` followed by `make stage-web`. Independent biostatistical review and release are separate from numerical verification.

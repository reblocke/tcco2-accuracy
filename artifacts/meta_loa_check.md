# Meta-analysis LoA Check

- Reference fixture: `python/tests/fixtures/conway_table1.csv`.
- Formula: SD_total = sqrt(sigma^2 + tau^2); LoA = delta ± 2 * SD_total.
- Main analysis: delta = -0.1, sigma = 1.9, tau2 = 8.9 → SD_total = 3.537, LoA = -7.17 to 6.97 (matches Table 1 -7.1 to 6.9 within rounding).

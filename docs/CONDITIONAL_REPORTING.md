# Conditional PaCO₂ reporting amendment, 17 November 2026

Version 2 applies to fixed-prior conditional predictions. The original TCCO2-008 decision and earlier holds remain reproducible. The earlier results were known before this amendment. The original WIP grid qualified 230 of 324 intervals and 643 of 648 threshold probabilities. The largest saved fits had 12 exact-support interval holds.

Average the signed disagreement likelihood across the posterior before a single normalization on the complete empirical arterial support. For each ordered CDF cut use two contiguous within-chain batch Monte Carlo standard errors of the numerator/denominator ratio plus the absolute order-64 versus order-32 difference. Invert outward monotone lower and upper CDF envelopes. Qualify each of the 2.5th, 50th and 97.5th percentiles independently at bracket width ≤0.5 mmHg. Both endpoints qualify the interval. For each inclusive threshold require twice the MCSE plus the order discrepancy ≤0.001 and the discrepancy alone ≤0.0001 in probability units. Account for interpolation error in accelerated evaluations. These are pointwise numerical checks.

Round halves upward to whole mmHg and whole percentages after calculation. Display both rounded values when a qualified bracket crosses a rounding boundary. A rounded extreme can be shown as <1% or >99% when its entire numerical bracket supports the inequality. Structural zero and one are exact. Hold each failing quantity independently.

Retain four chains, rank-normalized R-hat <1.01, bulk and tail ESS ≥400 and zero post-warmup divergences. Refine quadrature on the saved draws when integration is limiting. Stop when all requested quantities qualify. Retain any remaining holds and their reasons. This amendment uses saved draws only.

## Acceptance cases

Check a point mass at each inclusive threshold, a nearby two-point ambiguity inside half a mmHg, a wide ambiguity, an independently held median, a rounding boundary, and a threshold failing the quadrature gate alone. Check signed offsets and normalization after likelihood averaging. Replay every integer reading 20–100 in all four strata. Compare the direct prefix, largest-fit screen, direct spot checks and the 48 split-prefix checks with their respective receipts. Check synthetic browser output against its matching offline fit. The PFT example at 47 rounds to median 45, interval 37–52 and probabilities 58% and 4%. At 100 its lower limit rounds to 74 mmHg.

## Implementation evidence

The private full-grid cache reuses all saved draws from each setting's largest archived fit. Degree-20 local Hermite expansions accelerate evaluation of the quadrature-weighted Gaussian mixture. The cache matched direct evaluation on a 2,048-draw sample across the complete set of signed offsets in each setting, with maximum absolute density error below 6 × 10⁻¹⁶ per mmHg. Every final row reproduced the saved threshold and quantile-cut probabilities within 2 × 10⁻¹⁰. The saved audit supplies the numerical brackets, including its conservative prefix bounds and 48 disjoint-prefix checks. The public synthetic export uses direct native Gaussian summation and retains its original same-fit variance calculations.

The amended grid qualifies all 324 intervals, 324 medians and 648 threshold probabilities. Raw support values still determine probabilities and quantiles. Display rounding applies afterward. Displayed widths use the difference of unrounded endpoints, with bounds propagated from both endpoint brackets.

The Word review copies show additions in bold and deletions in strikethrough. The preceding Word originals remain the last imported Mac sources, as recorded in the source inventory. The new review copy is a repository revision derived from that import.

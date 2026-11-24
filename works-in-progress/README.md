# Works in progress

## Manuscripts for coauthor review

| Draft | Clean reading copy | Changes |
| --- | --- | --- |
| Original Research | [Read the full paper](original-research.docx) | [Compare with the preceding full paper](original-research-review.docx) |
| Short Report | [Read the standalone report](letter.docx) | [Compare with the preceding Short Report](letter-review.docx) · [Compare with December 30](letter-december-comparison.docx) |
| Supplement to Original Research | [Read the supplement](supplement.docx) | [Compare with the preceding supplement](supplement-review.docx) |

Bold text marks additions and strikethrough marks deletions in the comparison copies. The clean manuscripts use double-spaced body text and references. Table 2 in Original Research uses 1.1-line spacing. The Short Report includes two figures. Figure 1 shows PaCO₂ predictions across settings, and Figure 2 compares referral policies in All settings. The supplement accompanies the full paper only. The private December originals retain their Word comments and Zotero fields.

**Earlier drafts.** [December 24 BWL](december-2025/2025-12-24-BWL-letter.docx) · [December 30 DAB](december-2025/2025-12-30-letter.docx) · [Preceding GitHub letter](https://github.com/reblocke/tcco2-accuracy/blob/c35d1b0ee2d7fc958b0ab47b5ce91c5a9cd5a4b3/works-in-progress/letter.docx)

The private manuscript archive preserves the intermediate drafts and their redlines, including the supplement that introduced the 75-row agreement audit.

## Evidence and review

| Material | Question it answers |
| --- | --- |
| [Prediction figure](paired-prediction.png) · [Cohort and prediction tables](cohort-and-prediction.md) | Given a reading from 20 to 100 mmHg, what simultaneous arterial values are plausible and how likely is PaCO₂ to reach 45 or 52? |
| [Arterial distribution](arterial-distribution.png) | How common were different arterial values before observing TcCO₂? |
| [Agreement intervals](agreement.md) · [Study appraisal](study-appraisal.csv) · [Source ledger](../models/future_setting/source_ledger.csv) | Which populations and measurement protocols contributed to each agreement fit? The appraisal adds protocol and risk-of-bias fields from Conway’s CC BY 4.0 archive, published supplement and selected primary reports. [Conway archive](https://doi.org/10.6084/m9.figshare.6244058.v2) |
| [Classification and referral](classification.md) · [Four-setting figure](classification-figure.png) · [December comparison](classification-comparison.png) | At a given arterial value, how often would TcCO₂ reach 45, and what would the referral rule miss per 100 eligible occasions? |
| [Aggregate manuscript tables](manuscript-aggregate-tables.md) | How do the results change with the arterial code, reference population, agreement fit, or omission of an ambulatory study? How do the two ABG strategies differ? |
| [Reviewer questions](reviewer-questions.md) · [Contrasts recorded before the runs](manuscript-contrasts.json) · [Same-75-row envelope replay](overall-envelope.R) | Which assumptions and calculations need independent verification? |

The custodian approved the manuscript aggregate tables for draft PR #24 on November 24, 2026. Independent statistical review, coauthor approval and submission remain pending. Individual records, exact empirical weights and full curves stay on the mini. [December source inventory](source-revisions.json)

## Secondary analysis of displayed TcCO₂

The four-panel figure compares referrals and residual errors at displayed TcCO₂ values under a fixed 40–50 mmHg band and 1%, 5% and 10% wrong-call caps. The x-axis is the measurement available when the decision is made. The earlier arterial-axis figure answers the forward classification question. [Overall policies](referral-all.png) · [ICU](referral-icu.png) · [ED/noncritical inpatient](referral-ed_inp.png) · [Ambulatory](referral-pft.png) · [Efficiency across error caps](referral-efficiency.png)

[Operating characteristics and paired contrasts](referral-tables.md) · [Encounter order and arterial predictions](encounter-selection.md) · [Reviewer questions](reviewer-questions.md)

## Importing a newer Word revision

From the Air checkout, run `make manuscript-wip` with Python, Pandoc, GitHub CLI and SSH access to the mini. Both hosts are checked for letter and supplement Word files in the checkout’s top-level `Drafts/` and the older working clone’s top-level `Drafts/`. Subdirectories contain conference material and are outside this import. Identical editable copies named “Current TcCO2 Letter.docx” and “Current TcCO2 Supplement.docx” were placed in both checkout draft folders. New revisions may use dated filenames containing “letter” or “supplement.”

The command compares SHA-256 hashes against current remote main. It prepares visible text and tables in a local Word review copy. Inspect it beside the current public Word file and reconcile changes in Word. Existing approved figures are carried forward by content hash. Other illustrations receive a review marker. Native Word equations retain their mathematical structure. Verify the prose, tables, authorship, graphics and source inventory before using the printed digest to open a draft PR. A changed review copy produces a new digest. Compare divergent edits, stale copies and newly named revisions with current main in Word. Save the reconciled document under the current source filename on its Mac. Keep superseded revisions in a local archive outside the import folders. An unavailable Mac is identified in the command output.

The WIP folder is part of the GitHub repository. The Pages deployment stages its own allowlist of calculator assets. [Private analysis and reproduction](../docs/MANUSCRIPT_OUTPUTS.md)

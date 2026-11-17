# Works in progress

[Read the Word research letter](letter.docx) · [Open the methods and agreement supplement](supplement.docx)

The research letter estimates paired arterial CO₂ given observed transcutaneous CO₂ in the population undergoing arterial blood gas testing. The December 30, 2025 letter is the legacy starting version. The Air and mini Word originals matched byte for byte at import and retain their comments. The working version replaces the legacy figures and results. [Source inventory](source-revisions.json)

| Review material | Question |
| --- | --- |
| [Cohort and prediction tables](cohort-and-prediction.md) | Who contributed and how do intervals and probabilities change with the reading and setting? |
| [Arterial distribution figure](arterial-distribution.png) | How do first-day arterial distributions differ across ABG-tested populations? |
| [Paired prediction figure](paired-prediction.png) | How does arterial uncertainty change as observed TcCO₂ rises? Gaps mark unresolved estimates. |
| [Agreement comparison](agreement.md) | How do published limits, corrected legacy limits and the new target differ? |
| [Classification and ABG referral](classification.md) | How many patients would receive an ABG or a false classification under each strategy? |

## Review and remaining work

- [x] Reconstruct the four patient-weighted populations, demographic summaries and threshold prevalences. The historical 7,460-encounter gap consists of ED encounters with critical-care billing. All includes eligible patients outside the named setting definitions.
- [x] Generate the distribution figure from grouped probabilities on the mini. The exact empirical support, weights, patient records and full posterior density curves stay there.
- [x] Evaluate both inclusive thresholds and the 95% paired prediction interval on the integer 20–100 mmHg grid. An independent calculation on the mini checked all 324 rows, including the precision holds. The table gives six example readings and counts of resolved values. The 52 mmHg outcome refers to the full selected population.
- [x] Recalculate threshold classification and the reflex-ABG strategy with agreement-parameter uncertainty. The historical outputs under `artifacts/` retain their legacy status.
- [ ] Independent biostatistical review of the study ledger, overlap, uncertainty, transport assumptions, operating characteristics and unresolved grid points.
- [ ] Coauthors to confirm the byline, affiliations, institutional protocol and data-access wording, funding and disclosures. Check journal length and figure requirements after choosing the destination.
- [ ] Submission and manuscript release. This GitHub package is a working draft. Clinical coverage requires an independent paired validation sample.

## Importing a newer Word revision

From the Air checkout, run `make manuscript-wip` with Python, Pandoc, GitHub CLI and SSH access to the mini. Both hosts are checked for letter and supplement Word files in the checkout’s top-level `Drafts/` and the older working clone’s top-level `Drafts/`. Subdirectories contain conference material and are outside this import. Identical editable copies named “Current TcCO2 Letter.docx” and “Current TcCO2 Supplement.docx” were placed in both checkout draft folders. New revisions may use dated filenames containing “letter” or “supplement.”

The command compares SHA-256 hashes against current remote main. It prepares visible text and tables in a local Word review copy. Inspect it beside the current public Word file and reconcile changes in Word. Existing approved figures are carried forward by content hash. Other illustrations receive a review marker. Native Word equations retain their mathematical structure. Verify the prose, tables, authorship, graphics and source inventory before using the printed digest to open a draft PR. A changed review copy produces a new digest. Compare divergent edits, stale copies and newly named revisions with current main in Word. Save the reconciled document under the current source filename on its Mac. Keep superseded revisions in a local archive outside the import folders. An unavailable Mac is identified in the command output.

The WIP folder is part of the GitHub repository. The Pages deployment stages its own allowlist of calculator assets. [Private analysis and reproduction](../docs/MANUSCRIPT_OUTPUTS.md)

# Works in progress · Biostatistical review packet

## Manuscripts for coauthor review

| Draft | Clean reading copy | Changes |
| --- | --- | --- |
| Original Research | [Read the full paper](original-research.docx) | [Compare with the preceding full paper](original-research-review.docx) |
| Short Report | [Read the standalone report](letter.docx) | [Compare with the preceding Short Report](letter-review.docx) · [Compare with December 30](letter-december-comparison.docx) |
| Supplement to Original Research | [Read the supplement](supplement.docx) | [Compare with the preceding supplement](supplement-review.docx) |

Bold text marks additions and strikethrough marks deletions in the comparison copies. The clean manuscripts use double-spaced body text and references. Table 2 in Original Research uses 1.1-line spacing. The Short Report includes two figures. Figure 1 shows PaCO₂ predictions across settings, and Figure 2 compares referral policies in All settings. The supplement accompanies the full paper only. The private December originals retain their Word comments and Zotero fields.

**Earlier drafts.** [December 24 BWL](december-2025/2025-12-24-BWL-letter.docx) · [December 30 DAB](december-2025/2025-12-30-letter.docx) · [Preceding GitHub letter](https://github.com/reblocke/tcco2-accuracy/blob/c35d1b0ee2d7fc958b0ab47b5ce91c5a9cd5a4b3/works-in-progress/letter.docx)

The private manuscript archive preserves the intermediate drafts and their redlines, including the supplement that introduced the 75-row agreement audit.

## Evidence and questions for review

| Read in this order | Question |
| --- | --- |
| [Arterial distribution](arterial-distribution.png) · [Cohort and prediction tables](cohort-and-prediction.md) | Who underwent ABG testing, and how common was hypercapnia before a TcCO₂ reading? |
| [Four-setting prediction figure](paired-prediction.png) · [24 example predictions](cohort-and-prediction.md#given-observed-tcco₂) | Given TcCO₂, what simultaneous PaCO₂ values are plausible? What are the probabilities of ≥45 and ≥52 mmHg? |
| [Agreement limits and endpoint audit](agreement.md) · [Study appraisal](study-appraisal.csv) · [Source ledger](../models/future_setting/source_ledger.csv) | How much can a paired difference vary in a new setting? How much uncertainty surrounds the estimated agreement limits? |
| [Single-cutoff figure](classification-figure.png) · [Classification and continuous 40–50 referral tables](classification.md) · [December comparison](classification-comparison.png) | At a given arterial value, how often does TcCO₂ reach 45? What happens if continuous readings of 40–50 mmHg prompt an ABG? |
| **Whole-number device policies** · [All settings](referral-all.png) · [Critical Care Billing](referral-icu.png) · [ED/noncritical inpatient](referral-ed_inp.png) · [Ambulatory](referral-pft.png) · [Efficiency](referral-efficiency.png) · [Referral tables](referral-tables.md) | Per 100 eligible occasions, how often would each rule refer for ABG, classify without ABG, or leave a wrong call? Compare the 1%, 5%, and 10% caps with fixed 40–50. |
| [Sensitivity analyses](manuscript-aggregate-tables.md) · [Encounter selection](encounter-selection.md) · [Prespecified contrasts for the post hoc analyses](manuscript-contrasts.json) | How do the arterial record definition, population, agreement fit, and four ambulatory studies influence the results? |
| [Questions for the biostatistician](reviewer-questions.md) · [Endpoint calculation](overall-envelope.R) | Please review the population weighting, transport assumptions, endpoint uncertainty, and referral denominators. |

Critical Care Billing identifies eligible inpatient encounters with a critical-care billing indicator. ICU bed location was unavailable. The matched agreement evidence comes from ICU studies.

The packet uses the approved aggregate results. Independent statistical review, coauthor approval and validation of individual-patient coverage remain pending. [December source inventory](source-revisions.json) · [Manuscript maintenance and Word import](../docs/MANUSCRIPT_OUTPUTS.md#importing-a-newer-word-revision)

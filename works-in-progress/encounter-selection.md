# How encounter selection affects the arterial reference population

The November 2023 request specified an initial qualifying 2022 encounter for each of six indications within each of three settings. The historical notebooks selected the first encounter for each indication and then joined the setting file. The maintained implementation selects within setting first and also changes gas eligibility. We replayed both orders on one recovered candidate snapshot with the historical gas rules. Both replays applied the historical demographic and regional exclusions. We used the archived patient–encounter membership for the primary predictions. No contemporaneous execution receipt binds the recovered snapshot and notebook to the archived May 2025 table.
The historical arterial gas path accepted ordinary arterial PCO₂, temperature-adjusted arterial PCO₂ and total arterial CO₂ (2026-3). Its venous gas path accepted specimen-unspecified PCO₂ (11557-6). Numeric gas values had to exceed 5 and be below 200. The maintained gas rule uses values above 45 and below 200 after conversion. The legacy encounter-level Stata PaCO₂ averaged ordinary arterial, temperature-adjusted arterial and specimen-unspecified records. These choices can change which encounters enter before arterial outcome selection.
The present arterial distribution uses source-recorded LOINC 2019-8 or 32771-8, a convertible pressure unit, and finite PaCO₂ above 2 and below 250 mmHg. We kept all eligible arterial result records on the first testing calendar day and divided each patient’s unit weight equally across those records. All and the three settings use independent first days. One patient can contribute to more than one setting. Records may share a specimen. The available dates do not establish an intraday order or pretreatment status.
The reason-first reconstruction matched all 36 saved before- and after-screen notebook key sets. Its final keys matched archived membership within all 18 indication-by-setting groups. It also reproduced each of the four patient-weighted arterial distributions. The setting-first reconstruction changed membership.

[Original cohort specification](https://github.com/reblocke/trinetx-hypercapnia-code/blob/d73c0559f480da16db0a5b46258a507ce4b75338/archive/legacy-projects/code-database-documentation/2023-11-27%20TriNetX%20Hypercapnia%20Model%20Data%20Specification.docx) · [Historical encounter selection](https://github.com/reblocke/trinetx-preprocessing/blob/daf45eafcc87b86badd21a4c8dcdb434bd5673a3/Hypercapnia%20Final%20Dataset%20Generation%20-%20Master.ipynb) · [Historical gas definitions](https://github.com/reblocke/trinetx-preprocessing/blob/daf45eafcc87b86badd21a4c8dcdb434bd5673a3/Hypercapnia%20NEW%20DATA%20-%20RFS%20Processing.ipynb) · [Maintained definitions](https://github.com/reblocke/trinetx-preprocessing/blob/daf45eafcc87b86badd21a4c8dcdb434bd5673a3/docs/SPEC.md) · [Archived Stata variable](https://github.com/reblocke/trinetx-hypercapnia-code/blob/d73c0559f480da16db0a5b46258a507ce4b75338/stata/do/10_preprocessing.do#L846-L877)

## Ordered archived-cohort flow

| Selection in order | Patients | Encounters | Arterial result records |
| --- | --- | --- | --- |
| archived encounters | 749,019 | 833,476 | — |
| candidate arterial codes | 142,352 | 147,246 | 1,046,594 |
| source-recorded | 142,352 | 147,246 | 1,046,594 |
| convertible pressure unit | 142,352 | 147,246 | 1,046,594 |
| finite pressure | 142,280 | 147,165 | 1,024,104 |
| eligible 2 < PaCO₂ < 250 | 142,156 | 147,028 | 1,019,099 |
| first eligible testing day | 142,156 | 142,424 | 385,867 |

Each row retains the preceding restrictions. The first row counts membership keys. Source results become countable at the arterial-code screen.

## Prediction at observed TcCO₂ 47 mmHg

Within each group all three rows use the same agreement fit. Probabilities are for simultaneous arterial PaCO₂ at or above 45 or 52 mmHg. An en dash within a rounded endpoint or probability spans two display bins allowed by its numerical bracket. An unresolved estimate reads Held.

| Setting | Population | Baseline ≥45 | Baseline ≥52 | Median | 95% interval | After ≥45 | After ≥52 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| All settings | archived | 29.4% | 15.3% | 44 | 35 to 53 | 47% | 5% |
| All settings | reason-first | 29.4% | 15.3% | 44 | 35 to 53 | 47% | 5% |
| All settings | setting-first | 29.4% | 15.3% | 44 | 35 to 53 | 47% | 5% |
| Critical Care Billing | archived | 31.6% | 18.8% | 44 | 35 to 54 | 46% | 6% |
| Critical Care Billing | reason-first | 31.6% | 18.8% | 44 | 35 to 54 | 46% | 6% |
| Critical Care Billing | setting-first | 31.6% | 18.8% | 44 | 35 to 54 | 46% | 6% |
| ED/noncritical inpatient | archived | 27.8% | 12.9% | 45 | 35 to 56 | 54% | 10% |
| ED/noncritical inpatient | reason-first | 27.8% | 12.9% | 45 | 35 to 56 | 54% | 10% |
| ED/noncritical inpatient | setting-first | 27.9% | 13.0% | 45 | 35 to 56 | 54% | 10% |
| Ambulatory | archived | 26.5% | 10.1% | 45 | 37 to 52 | 58% | 4% |
| Ambulatory | reason-first | 26.5% | 10.1% | 45 | 37 to 52 | 58% | 4% |
| Ambulatory | setting-first | 27.3% | 10.9% | 45 | 37 to 53 | 58% | 4% |

## reason-first (A) and setting-first (B), before setting quality screens

Reason and setting memberships overlap. Shared patients may have different encounters. Encounter identity includes both patient and encounter keys. Archive comparisons always use its final membership.

| Setting | Indication | Patients both | Only A | Only B | Encounters both | Only A | Only B |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AMB | ABG | 11,099 | 0 | 1,043 | 11,099 | 0 | 1,043 |
| AMB | VBG | 11,005 | 0 | 2,154 | 11,005 | 0 | 2,154 |
| AMB | RESPFAIL | 45,252 | 0 | 19,596 | 45,252 | 0 | 19,596 |
| AMB | OBESITY | 669,427 | 0 | 76,688 | 669,427 | 0 | 76,688 |
| AMB | VENTSUPPORT | 5,040 | 0 | 662 | 5,040 | 0 | 662 |
| AMB | PREDISPOSITION | 1,294,920 | 0 | 334,875 | 1,294,920 | 0 | 334,875 |
| EMER | ABG | 17,804 | 0 | 1,316 | 17,804 | 0 | 1,316 |
| EMER | VBG | 76,465 | 0 | 8,062 | 76,465 | 0 | 8,062 |
| EMER | RESPFAIL | 20,013 | 0 | 4,474 | 20,013 | 0 | 4,474 |
| EMER | OBESITY | 63,882 | 0 | 35,288 | 63,882 | 0 | 35,288 |
| EMER | VENTSUPPORT | 14,207 | 0 | 1,162 | 14,207 | 0 | 1,162 |
| EMER | PREDISPOSITION | 174,187 | 0 | 138,723 | 174,187 | 0 | 138,723 |
| INPAT | ABG | 139,388 | 0 | 3,698 | 139,388 | 0 | 3,698 |
| INPAT | VBG | 149,449 | 0 | 8,605 | 149,449 | 0 | 8,605 |
| INPAT | RESPFAIL | 124,665 | 0 | 20,112 | 124,665 | 0 | 20,112 |
| INPAT | OBESITY | 47,325 | 0 | 40,884 | 47,325 | 0 | 40,884 |
| INPAT | VENTSUPPORT | 98,518 | 0 | 2,927 | 98,518 | 0 | 2,927 |
| INPAT | PREDISPOSITION | 243,409 | 0 | 201,740 | 243,409 | 0 | 201,740 |

## reason-first (A) and archive (B), before setting quality screens

Reason and setting memberships overlap. Shared patients may have different encounters. Encounter identity includes both patient and encounter keys. Archive comparisons always use its final membership.

| Setting | Indication | Patients both | Only A | Only B | Encounters both | Only A | Only B |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AMB | ABG | 8,326 | 2,773 | 0 | 8,326 | 2,773 | 0 |
| AMB | VBG | 8,834 | 2,171 | 0 | 8,834 | 2,171 | 0 |
| AMB | RESPFAIL | 6,255 | 38,997 | 0 | 6,255 | 38,997 | 0 |
| AMB | OBESITY | 141,551 | 527,876 | 0 | 141,551 | 527,876 | 0 |
| AMB | VENTSUPPORT | 1,583 | 3,457 | 0 | 1,583 | 3,457 | 0 |
| AMB | PREDISPOSITION | 185,099 | 1,109,821 | 0 | 185,099 | 1,109,821 | 0 |
| EMER | ABG | 15,705 | 2,099 | 0 | 15,705 | 2,099 | 0 |
| EMER | VBG | 54,226 | 22,239 | 0 | 54,226 | 22,239 | 0 |
| EMER | RESPFAIL | 15,819 | 4,194 | 0 | 15,819 | 4,194 | 0 |
| EMER | OBESITY | 31,655 | 32,227 | 0 | 31,655 | 32,227 | 0 |
| EMER | VENTSUPPORT | 11,555 | 2,652 | 0 | 11,555 | 2,652 | 0 |
| EMER | PREDISPOSITION | 95,819 | 78,368 | 0 | 95,819 | 78,368 | 0 |
| INPAT | ABG | 117,167 | 22,221 | 0 | 117,167 | 22,221 | 0 |
| INPAT | VBG | 122,768 | 26,681 | 0 | 122,768 | 26,681 | 0 |
| INPAT | RESPFAIL | 99,367 | 25,298 | 0 | 99,367 | 25,298 | 0 |
| INPAT | OBESITY | 33,558 | 13,767 | 0 | 33,558 | 13,767 | 0 |
| INPAT | VENTSUPPORT | 83,330 | 15,188 | 0 | 83,330 | 15,188 | 0 |
| INPAT | PREDISPOSITION | 174,260 | 69,149 | 0 | 174,260 | 69,149 | 0 |

## setting-first (A) and archive (B), before setting quality screens

Reason and setting memberships overlap. Shared patients may have different encounters. Encounter identity includes both patient and encounter keys. Archive comparisons always use its final membership.

| Setting | Indication | Patients both | Only A | Only B | Encounters both | Only A | Only B |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AMB | ABG | 8,326 | 3,816 | 0 | 8,326 | 3,816 | 0 |
| AMB | VBG | 8,834 | 4,325 | 0 | 8,834 | 4,325 | 0 |
| AMB | RESPFAIL | 6,255 | 58,593 | 0 | 6,255 | 58,593 | 0 |
| AMB | OBESITY | 141,551 | 604,564 | 0 | 141,551 | 604,564 | 0 |
| AMB | VENTSUPPORT | 1,583 | 4,119 | 0 | 1,583 | 4,119 | 0 |
| AMB | PREDISPOSITION | 185,099 | 1,444,696 | 0 | 185,099 | 1,444,696 | 0 |
| EMER | ABG | 15,705 | 3,415 | 0 | 15,705 | 3,415 | 0 |
| EMER | VBG | 54,226 | 30,301 | 0 | 54,226 | 30,301 | 0 |
| EMER | RESPFAIL | 15,819 | 8,668 | 0 | 15,819 | 8,668 | 0 |
| EMER | OBESITY | 31,655 | 67,515 | 0 | 31,655 | 67,515 | 0 |
| EMER | VENTSUPPORT | 11,555 | 3,814 | 0 | 11,555 | 3,814 | 0 |
| EMER | PREDISPOSITION | 95,819 | 217,091 | 0 | 95,819 | 217,091 | 0 |
| INPAT | ABG | 117,167 | 25,919 | 0 | 117,167 | 25,919 | 0 |
| INPAT | VBG | 122,768 | 35,286 | 0 | 122,768 | 35,286 | 0 |
| INPAT | RESPFAIL | 99,367 | 45,410 | 0 | 99,367 | 45,410 | 0 |
| INPAT | OBESITY | 33,558 | 54,651 | 0 | 33,558 | 54,651 | 0 |
| INPAT | VENTSUPPORT | 83,330 | 18,115 | 0 | 83,330 | 18,115 | 0 |
| INPAT | PREDISPOSITION | 174,260 | 270,889 | 0 | 174,260 | 270,889 | 0 |

## reason-first (A) and setting-first (B), after setting quality screens

Reason and setting memberships overlap. Shared patients may have different encounters. Encounter identity includes both patient and encounter keys. Archive comparisons always use its final membership.

| Setting | Indication | Patients both | Only A | Only B | Encounters both | Only A | Only B |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AMB | ABG | 8,326 | 0 | 701 | 8,326 | 0 | 701 |
| AMB | VBG | 8,834 | 0 | 1,731 | 8,834 | 0 | 1,731 |
| AMB | RESPFAIL | 6,255 | 0 | 3,432 | 6,255 | 0 | 3,432 |
| AMB | OBESITY | 141,551 | 0 | 19,570 | 141,551 | 0 | 19,570 |
| AMB | VENTSUPPORT | 1,583 | 0 | 289 | 1,583 | 0 | 289 |
| AMB | PREDISPOSITION | 185,099 | 0 | 60,306 | 185,099 | 0 | 60,306 |
| EMER | ABG | 15,705 | 0 | 1,015 | 15,705 | 0 | 1,015 |
| EMER | VBG | 54,226 | 0 | 5,636 | 54,226 | 0 | 5,636 |
| EMER | RESPFAIL | 15,819 | 0 | 2,797 | 15,819 | 0 | 2,797 |
| EMER | OBESITY | 31,655 | 0 | 19,169 | 31,655 | 0 | 19,169 |
| EMER | VENTSUPPORT | 11,555 | 0 | 855 | 11,555 | 0 | 855 |
| EMER | PREDISPOSITION | 95,819 | 0 | 76,752 | 95,819 | 0 | 76,752 |
| INPAT | ABG | 117,167 | 0 | 3,092 | 117,167 | 0 | 3,092 |
| INPAT | VBG | 122,768 | 0 | 7,037 | 122,768 | 0 | 7,037 |
| INPAT | RESPFAIL | 99,367 | 0 | 15,631 | 99,367 | 0 | 15,631 |
| INPAT | OBESITY | 33,558 | 0 | 27,010 | 33,558 | 0 | 27,010 |
| INPAT | VENTSUPPORT | 83,330 | 0 | 2,574 | 83,330 | 0 | 2,574 |
| INPAT | PREDISPOSITION | 174,260 | 0 | 138,145 | 174,260 | 0 | 138,145 |

## reason-first (A) and archive (B), after setting quality screens

Reason and setting memberships overlap. Shared patients may have different encounters. Encounter identity includes both patient and encounter keys. Archive comparisons always use its final membership.

| Setting | Indication | Patients both | Only A | Only B | Encounters both | Only A | Only B |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AMB | ABG | 8,326 | 0 | 0 | 8,326 | 0 | 0 |
| AMB | VBG | 8,834 | 0 | 0 | 8,834 | 0 | 0 |
| AMB | RESPFAIL | 6,255 | 0 | 0 | 6,255 | 0 | 0 |
| AMB | OBESITY | 141,551 | 0 | 0 | 141,551 | 0 | 0 |
| AMB | VENTSUPPORT | 1,583 | 0 | 0 | 1,583 | 0 | 0 |
| AMB | PREDISPOSITION | 185,099 | 0 | 0 | 185,099 | 0 | 0 |
| EMER | ABG | 15,705 | 0 | 0 | 15,705 | 0 | 0 |
| EMER | VBG | 54,226 | 0 | 0 | 54,226 | 0 | 0 |
| EMER | RESPFAIL | 15,819 | 0 | 0 | 15,819 | 0 | 0 |
| EMER | OBESITY | 31,655 | 0 | 0 | 31,655 | 0 | 0 |
| EMER | VENTSUPPORT | 11,555 | 0 | 0 | 11,555 | 0 | 0 |
| EMER | PREDISPOSITION | 95,819 | 0 | 0 | 95,819 | 0 | 0 |
| INPAT | ABG | 117,167 | 0 | 0 | 117,167 | 0 | 0 |
| INPAT | VBG | 122,768 | 0 | 0 | 122,768 | 0 | 0 |
| INPAT | RESPFAIL | 99,367 | 0 | 0 | 99,367 | 0 | 0 |
| INPAT | OBESITY | 33,558 | 0 | 0 | 33,558 | 0 | 0 |
| INPAT | VENTSUPPORT | 83,330 | 0 | 0 | 83,330 | 0 | 0 |
| INPAT | PREDISPOSITION | 174,260 | 0 | 0 | 174,260 | 0 | 0 |

## setting-first (A) and archive (B), after setting quality screens

Reason and setting memberships overlap. Shared patients may have different encounters. Encounter identity includes both patient and encounter keys. Archive comparisons always use its final membership.

| Setting | Indication | Patients both | Only A | Only B | Encounters both | Only A | Only B |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AMB | ABG | 8,326 | 701 | 0 | 8,326 | 701 | 0 |
| AMB | VBG | 8,834 | 1,731 | 0 | 8,834 | 1,731 | 0 |
| AMB | RESPFAIL | 6,255 | 3,432 | 0 | 6,255 | 3,432 | 0 |
| AMB | OBESITY | 141,551 | 19,570 | 0 | 141,551 | 19,570 | 0 |
| AMB | VENTSUPPORT | 1,583 | 289 | 0 | 1,583 | 289 | 0 |
| AMB | PREDISPOSITION | 185,099 | 60,306 | 0 | 185,099 | 60,306 | 0 |
| EMER | ABG | 15,705 | 1,015 | 0 | 15,705 | 1,015 | 0 |
| EMER | VBG | 54,226 | 5,636 | 0 | 54,226 | 5,636 | 0 |
| EMER | RESPFAIL | 15,819 | 2,797 | 0 | 15,819 | 2,797 | 0 |
| EMER | OBESITY | 31,655 | 19,169 | 0 | 31,655 | 19,169 | 0 |
| EMER | VENTSUPPORT | 11,555 | 855 | 0 | 11,555 | 855 | 0 |
| EMER | PREDISPOSITION | 95,819 | 76,752 | 0 | 95,819 | 76,752 | 0 |
| INPAT | ABG | 117,167 | 3,092 | 0 | 117,167 | 3,092 | 0 |
| INPAT | VBG | 122,768 | 7,037 | 0 | 122,768 | 7,037 | 0 |
| INPAT | RESPFAIL | 99,367 | 15,631 | 0 | 99,367 | 15,631 | 0 |
| INPAT | OBESITY | 33,558 | 27,010 | 0 | 33,558 | 27,010 | 0 |
| INPAT | VENTSUPPORT | 83,330 | 2,574 | 0 | 83,330 | 2,574 | 0 |
| INPAT | PREDISPOSITION | 174,260 | 138,145 | 0 | 174,260 | 138,145 | 0 |

## Change with setting-first selection at 47 mmHg

Setting-first minus archived. Baseline differences use the underlying patient-weighted probabilities and are in percentage points. For paired predictions the differences are between the certified whole-mmHg endpoints or median and whole-percentage-point displays. A zero in those columns means both outputs have the same displayed value.

| Setting | Baseline ≥45 (pp) | Baseline ≥52 (pp) | Lower (mmHg) | Median (mmHg) | Upper (mmHg) | After ≥45 (pp) | After ≥52 (pp) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| All settings | +0.04 | +0.02 | +0 | +0 | +0 | +0 | +0 |
| Critical Care Billing | +0.02 | +0.00 | +0 | +0 | +0 | +0 | +0 |
| ED/noncritical inpatient | +0.12 | +0.08 | +0 | +0 | +0 | +0 | +0 |
| Ambulatory | +0.82 | +0.78 | +0 | +0 | +1 | +0 | +0 |

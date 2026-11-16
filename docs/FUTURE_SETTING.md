# Reproduce TCCO2-008 offline

Use the repository checkout and a C++17 compiler. Install the fitting dependency group with `uv sync --locked --group fitting`. Provision the toolchain once using the official [CmdStan 2.40.0 release](https://github.com/stan-dev/cmdstan/releases/tag/v2.40.0). These commands use an untracked local cache.

```sh
mkdir -p "$HOME/.cache/tcco2-cmdstan"
cd "$HOME/.cache/tcco2-cmdstan"
curl -fL --retry 2 -o cmdstan-2.40.0.tar.gz https://github.com/stan-dev/cmdstan/releases/download/v2.40.0/cmdstan-2.40.0.tar.gz
echo '26d21c9737b35728e926f9b7a5d7362b38dc23e2e7d5e166c890e1fa74dac90c  cmdstan-2.40.0.tar.gz' | shasum -a 256 -c - &&
  tar -xzf cmdstan-2.40.0.tar.gz && make -C cmdstan-2.40.0 -j2 build
```

For the Apple Command Line Tools installation on the development machine, the standard C++ headers were in the SDK. Set `SDKROOT="$(xcrun --show-sdk-path)"` and export `CPLUS_INCLUDE_PATH="$SDKROOT/usr/include/c++/v1"` before building. Set `PRECOMPILED_HEADERS=false` in CmdStan's `make/local` to reduce disk requirements. An alternative installation can set `CMDSTAN` to its extracted directory and keep the verified release archive beside it. Fitting checks the archive digest, CmdStanPy version, executable version, and committed public inputs.

Run these commands from the repository root.

```sh
make future-setting-recovery
make future-setting-fit
```

Each command creates a uniquely named run under `.pytest_tmp/future-setting/` and writes its path on completion. Supply `FUTURE_SETTING_ARGS='--run-id my-run'` to name the run, or `FUTURE_SETTING_ARGS='--run-id my-run --resume'` to reuse verified sampler receipts. The latter checks the evidence revision, software and posterior hashes. Each attempted chain length retains its CSVs, sampler log, indexed posterior and precision record. The manifest identifies the accepted attempt for each stratum or the reason for a hold. Input and run hashes connect all candidate tables to their model.

To fit only the overall public stratum, run `make future-setting-fit FUTURE_SETTING_ARGS='--run-id public-main --stratum main'`. The public command accepts `main`, `icu`, `arf`, or `lft` and defaults to all four. The interval bridge covers each selected stratum. Resuming a run preserves its original selection. Start a new run ID after committing a revised adjudication ledger.

To reproduce the row-level archive audit, run `uv run python scripts/future_setting_validation.py --archive PATH` with the untracked Figshare RData. It verifies the archive checksum, four source memberships, paired and participant counts, source variances, and repeated-measures adjustment. The receipt includes each row's signed midpoint and SD discrepancy. To check the empirical-prior interface with invented encounter records against a completed public ICU candidate, run `uv run python scripts/future_setting_validation.py --public-run RUN_DIRECTORY`.

The public fit uses the Conway summaries. Its conditional table uses synthetic prior bin centers and weights, with `P(PaCO₂ ≥ 45 mmHg)` on the 35–55 mmHg TcCO₂ grid at intervals of 5 mmHg. The ED/inpatient mapping draws on acute respiratory failure studies. The PFT mapping draws on lung-function studies, including respiratory inpatients. Each mapping assumes a comparable mix of setting effects within the ABG-observed target cohort. A cleared public stratum supplies one kernel for the predictive, variance and inverse tables. Discrete posterior quantiles are observed support points. A pointwise interval for a population endpoint has its own quantity label.

### Fixed empirical TriNetX application

After the private provenance and destination have been approved, supply the eligible records from that workspace to `tcco2_accuracy.workflows.future_setting_inverse.infer_index_cohort` in memory. Supply an `EmpiricalPriorMetadata` object with non-sensitive cohort, eligibility and observation-window revisions, the public candidate run directory, and one target from `pft`, `ed_inp`, `icu`, `all`. Use `PatientInputColumns` for custom patient, encounter and order fields. The caller specifies the cohort and applies its window before this call.

The function selects the earliest eligible measurement in the earliest eligible encounter per patient within target, or across all settings for pooled All. Exact ties fail closed. All finite positive index values contribute to the normalized empirical mass. The return value contains private candidate summaries with their precision checks. Store any output in the approved private workspace. The clinical source, prior support and patient identifiers remain private. The [provenance template](restricted_data_provenance.template.json) and [governance policy](DATA_GOVERNANCE.md) govern private application and release.

## Private conditional lookup

The approved input must already contain arterial measurements in mmHg within the specified eligibility window. Each row needs patient ID, encounter ID, encounter order, measurement order and either a setting or the documented assignment flags. The legacy local in-silico file lacks the patient, encounter and measurement-order keys. A first-encounter indicator alone cannot establish the index rule.

In the approved workspace, prepare a JSON receipt with `use` set to `approved_private_index_analysis`, `eligible_extract` (CSV or Stata path), `eligible_extract_sha256`, `private_destination` (existing approved directory), and `revisions` containing custodian-confirmed `cohort_revision`, `eligibility_revision` and `window_revision`. Optional `columns` maps the four index roles to input field names. The [restricted provenance record](restricted_data_provenance.template.json) documents authority and permitted destinations. Validate the extraction and window before issuing the receipt.

Run `uv run python scripts/conditional_lookup.py /approved/private/receipt.json`. It selects one index measurement per patient in each target, and one overall for All. It writes the aggregate lookup and provenance receipt together in the new `conditional-lookup/` directory under the approved private destination. Each lookup contains four prior intervals and 81 ordered rows per setting on the inclusive integer grid 20 through 100 mmHg. A local JSON file may be opened in the browser. Public promotion requires review of the complete file, approval of its SHA-256 and the Pages destination, and an explicit data allowlist change. The prior support and weights stay private.

### Private distribution preview

Use the existing fit for the requested setting and one approved input. With the eligible-record receipt above, the builder selects the index ABG within that setting. For `all`, it selects one index ABG per patient across all eligible settings.

Alternatively, the custodian can supply an exact frequency table for that single target. Each CSV row represents one distinct arterial PaCO₂ value at its original precision. Use `paco2_bin,count` for positive integer patient counts or `paco2_bin,weight` for positive proportions summing to one. The counts or weights must come from the same index rule. Broad bins, encounter-level averages and pooling setting-specific tables can change the prior.

Use the same receipt fields for the approved file, its SHA-256, revisions, use and destination. For a frequency table, also set `input_kind` to `index_frequencies`, `index_target` to the requested setting, and `index_policy` to `earliest_eligible_measurement_in_earliest_eligible_encounter_per_patient`. These entries document the upstream index selection. Only the custodian can confirm that the underlying records satisfy it.

The original-cohort all-settings preview uses `index_policy` set to `equal_patient_weight_split_across_eligible_draws_on_first_testing_day`. Determine the first eligible arterial testing date per patient across all their cohort encounters over the full observation period. A patient with three eligible results on that date contributes one third of their weight to each result. Sum contributions at each observed PaCO₂ value, divide by the number of contributing patients and supply the resulting `paco2_bin,weight` table. The private viewer identifies this population rule.

From the checkout containing the accepted public fits, run

```sh
uv run python scripts/conditional_lookup.py /approved/private/receipt.json --private-chart all
```

Choose `all`, `icu`, `ed_inp` or `pft`. The command writes `private-chart-<target>/` inside the approved destination. Open its `preview.html` directly in a browser, then select the adjacent `private-chart.json`. The viewer reads that file locally and works offline. The accompanying `receipt.json` records the input and fit provenance. Keep all derived distributions and real-data screenshots in the approved workspace.

The model evaluates the disagreement density at each observed arterial value minus 47 mmHg, multiplies by its empirical prior mass and normalizes once. The displayed interval and inclusive probabilities use the complete support. Zoom and threshold selection change the display only. An unresolved interval or threshold probability is withheld independently.

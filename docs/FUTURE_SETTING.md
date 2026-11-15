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

To reproduce the row-level archive audit, run `uv run python scripts/future_setting_validation.py --archive PATH` with the untracked Figshare RData. It verifies the archive checksum, four source memberships, paired and participant counts, source variances, and repeated-measures adjustment. The receipt includes each row's signed midpoint and SD discrepancy. To check the empirical-prior interface with invented encounter records against a completed public ICU candidate, run `uv run python scripts/future_setting_validation.py --public-run RUN_DIRECTORY`.

The public fit uses the Conway summaries. Its conditional table uses synthetic prior bin centers and weights, with `P(PaCO₂ ≥ 45 mmHg)` on the 35–55 mmHg TcCO₂ grid at intervals of 5 mmHg. The ED/inpatient mapping draws on acute respiratory failure studies. The PFT mapping draws on lung-function studies, including respiratory inpatients. Each mapping assumes a comparable mix of setting effects within the ABG-observed target cohort. A cleared public stratum supplies one kernel for the predictive, variance and inverse tables. Discrete posterior quantiles are observed support points. A pointwise interval for a population endpoint has its own quantity label.

### Fixed empirical TriNetX application

After the private provenance and destination have been approved, supply the eligible records from that workspace to `tcco2_accuracy.workflows.future_setting_inverse.infer_index_cohort` in memory. Supply an `EmpiricalPriorMetadata` object with non-sensitive cohort, eligibility and observation-window revisions, the public candidate run directory, and one target from `pft`, `ed_inp`, `icu`, `all`. Use `PatientInputColumns` for custom patient, encounter and order fields. The caller specifies the cohort and applies its window before this call.

The function selects the earliest eligible measurement in the earliest eligible encounter per patient within target, or across all settings for pooled All. Exact ties fail closed. All finite positive index values contribute to the normalized empirical mass. The return value contains private candidate summaries with their precision checks. Store any output in the approved private workspace. The clinical source, prior support and patient identifiers remain private. The [provenance template](restricted_data_provenance.template.json) and [governance policy](DATA_GOVERNANCE.md) govern private application and release.

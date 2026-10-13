# Historical facility information under fixed heat-rate relations

This directory contains full-source and six single-source-omission evaluations,
facility changes, matched-date comparisons, 1,000 assignment controls and
5,000 operator-cluster draws. `SCOPE_MAP.json` describes the available operations;
`INPUT_SCOPE.json` identifies the inputs and `DATA_DICTIONARY.csv` defines result
fields. Release version 1.0.0 is prepared for deposition.

## Sources And License

EPA CAMPD, EIA-860/EIA-923 and NOAA NCEI sources and their terms are recorded in
`attribution/` and the license matrices. The selected CC BY 4.0 scope for
author-controlled code and derived data is recorded in
`attribution/original/combined_cycle_heat_rate/data/followup_2025/LICENSE_SCOPE.json`.
Upstream source terms apply to the corresponding material. Load serialized
joblib caches only from trusted, hash-verified sources.

## Evaluate Supplied Inputs

Extract the code and data ZIPs into one empty directory. Set `PACKAGE_ROOT`
to its `combined_cycle_heat_rate/historical_transfer` directory and `PYTHON`
to the intended interpreter. The numerical environment uses Python 3.12,
NumPy 2.3.5, pandas 2.2.3, SciPy 1.15.3, scikit-learn 1.6.1, joblib 1.4.2
and threadpoolctl 3.6.0. SciPy 1.15.3 is compatible with the stored BSpline
cache state. Install the dependencies before a network-disabled evaluation.

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$PACKAGE_ROOT/code${RUNTIME_PATH:+:$RUNTIME_PATH}"
mkdir -p "$CHECK_ROOT"
cd "$PACKAGE_ROOT/code"
"$PYTHON" -m unittest -v test_historical_transfer test_package
"$PYTHON" package_tools.py --package-root "$PACKAGE_ROOT" \
  --output "$CHECK_ROOT/input_manifest.json"
"$PYTHON" check_prepared.py --package-root "$PACKAGE_ROOT" \
  --input-manifest "$CHECK_ROOT/input_manifest.json" \
  --analysis-spec "$PACKAGE_ROOT/analysis_spec.json" \
  --output "$CHECK_ROOT/verification.json"
```

Choose `CHECK_ROOT` outside the extracted package. The helper resolves declared
input roles and hashes. `check_prepared.py` evaluates the supplied full-source
and 12 omission-cache predictions, facility metrics, interval shifts, stored
matching, bootstrap draws and assignment summaries. Its output records the
operations evaluated. Tests cover invalid identities and synthetic matching
cases. Fitted models are evaluated with fitting methods disabled.

## Scientific Inputs

The annual prediction input uses the consistently processed 2024 weather bridge
and 2025 records. Historical 2024 omission evaluation uses the separate
corrected-measurement GSOD reference. Yearless input records obtain their year
from their declared manifest role.

Historical facility offsets use full 2024 support in both evaluation scopes.
The pooled offset averages the primary 120 facilities equally. Interval centers
and endpoints shift together at the stored width. Facility-level changes,
including positive and negative changes, are in `results/FACILITY_ALL_CHANGES.csv`;
`results/FACILITY_ALL_METRICS.csv` supplies the full metric set. Stored operator
draws retain complete facility identities.

Original EPA/EIA/NOAA reconstruction uses the separate portable replay workflow
and original inputs identified in the acquisition manifests. The inputs in this
directory support fixed-model evaluation and derived-result reconstruction.
`SHA256_MANIFEST.json` records the current member identities; source hashes in
`attribution/` identify the original scientific records.

## Figures

The manuscript renderer reads the supplied source values and writes current
Figure 2, Figure S2 and source-value mappings to a new output directory.

```bash
python3 -B code/render_historical_transfer.py --results results --output NEW_EMPTY_OUTPUT
```

Figure 2c displays the 1,000 permuted facility assignments per model with
shared bins and count scales; each blue marker is the actual facility mapping.
These ordinary permutations may retain original assignments. Figure S2 includes
the 120-facility main and 101-facility matched-date scopes. Facilities are sorted
independently in each panel, with identities recorded in
`supporting_tables/Figure_S2_identity_mapping.csv` at the package root.

For renderer tests, use a separate output directory:

```bash
python3 -B code/render_historical_transfer.py --results results --output NEW_EMPTY_TEST_OUTPUT --self-test
```

Rendering dependencies are in `requirements-figures.txt`; Times New Roman must
be lawfully installed. Current display numbering is in `MANUSCRIPT_INTERFACE.json`.
Figures are generated outputs and are not distributed in the ZIPs.

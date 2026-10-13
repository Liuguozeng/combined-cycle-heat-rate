## Portable lawful-input replay

This workflow rebuilds the corrected Quail Run facility-year from its hourly
carrier and weather. Other facilities enter through identified daily and
prediction references. It evaluates fixed-model, annual, common-panel and
42 cached-subset results.

Use Python 3.13.5 with the versions in `reproduction/requirements.txt` for
source replay. Copy `reproduction/input_manifest.example.json` outside the
package and supply the carriers listed in `reproduction/INPUT_REQUIREMENTS.json`.

```sh
python validation/verify_reported_values.py
python validation/verify_presentation_tables.py
python validation/rerun_submission_results.py --input-manifest input_manifest.json --output-dir replay_output
```

The first two commands verify supplied results. Replay uses an isolated
temporary directory and can save output outside the package and input
carriers. `--source-workspace` is an alternative input locator; choose it or
`--input-manifest`. Missing required inputs return `NOT EXECUTED` (exit 2);
changed identities, missing outputs and unequal results fail.
`rerun_primary_entrypoint.py` evaluates the corrected endpoint, primary and
common-panel results. Cached source-subset evaluation uses the submission-results
entry point. Implementation and dependency provenance are in
`reproduction/code` and `CODE_PROVENANCE.json`.

Numerical replay checks CSV values, discrete identities and categories.
Compression headers can vary while decompressed cells agree. Current
manuscript results are under `results/`; historical reference inputs carry
their own identities. Obtain raw carriers under the cited source terms.

## Hourly endpoint and follow-up denominator evidence

The apportioned unit-hour streaming product reports Gross Load as an MW rate.
Daily gross generation sums each retained unit's Gross Load times its Operating
Time once; Heat Input is an energy amount and is summed once over those same
members. This differs from the already integrated annual gross-generation product.
See `source_provenance/ENDPOINT_PRODUCT_PROVENANCE.json` for pinned official
implementation references and `supporting_tables/ENDPOINT_PARTIAL_HOUR_CASES.csv`
for partial-hour examples. `supporting_tables/S14_2025_LOW_TAIL_DENOMINATORS.csv`
records the low-tail facilities' eligible, supported and covered days, week clusters,
existing intervals, calendar-month coverage and actual weather branches.
Run `python -B validation/verify_supporting_tables.py` to independently reduce
these added denominators from the supplied frozen prediction records.


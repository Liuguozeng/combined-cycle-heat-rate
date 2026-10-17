# Historical mean point comparator

The historical level is the arithmetic mean of observed daily gross heat rates
on each facility's complete 2024 weather-bridge in-support dates.
Historical means and facility offsets are estimated from those 2024 records;
2025 observations supply the evaluation outcomes.

The main comparison uses the same 120 facilities and 29,443 facility-days as
the historical-transfer branch. The conditional comparison uses the existing
101 facilities and 6,723 matched 2025 facility-days. M1 and M2 are evaluated
on the same facility-date records.

Run the `historical-mean` command documented in the root README. Input paths are
relative to `input_manifest.json`, and their hashes identify the frozen weather
bridge, follow-up, stored matches and 2024 operator assignments.

## Outputs

`HISTORICAL_LEVELS.csv` records each model/facility's history-day count, observed
and predicted historical means, and historical offset. `DAILY_MEAN_CONTROL.csv.gz`
records the point predictions and observed outcomes. `FACILITY_MEAN_CONTROL.csv`
contains 442 rows; `SUMMARY_MEAN_CONTROL.csv` contains four complete comparisons.
`BOOTSTRAP_MEAN_CONTROL.csv.gz` retains all 5,000 registered draws per comparison.

The paired MAE difference is facility-centered minus historical-mean MAE;
positive values favor the historical mean. `improved` counts facilities with a
negative difference, and `worsened` those with a positive difference. CIs reuse
the 91 registered 2024 operator clusters; 77 operators are observed in the
conditional subset. Facilities are equally weighted. Percentile intervals
condition on the observed historical levels, fixed models and stored matches.
This branch evaluates point predictions; interval coverage and Winkler scores
belong to the separate fixed-interval evaluation.

Implementation provenance and derived-input terms follow the root attribution
records. The inputs are prepared daily records, with original EPA/EIA/NOAA
acquisition documented in the source-data workflows.

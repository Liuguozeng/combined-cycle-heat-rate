# Plant levels and operating-condition response

This module compares historical means and medians, original centered M1/M2,
within-source response estimation and a single historical local response coefficient.
It uses the original six-source 2024 daily records and the existing 120-plant
2024-weather-bridge-to-2025 support. All response coefficients are estimated
from 2024; 2025 outcomes evaluate them. This is retrospective development.

Run from the package root after extracting the code and data archives together:

```sh
python3 -B reproduce.py response --package-root . --output /absolute/new-response-output
python3 -B reproduce.py figures --package-root . --output /absolute/new-figure-output
```

The first command performs the declared new response fits; it does not overwrite
the original M1/M2 specifications or caches. The second only renders supplied
tables, including main Figure 2 and Supplementary Figure S4. Outputs include all
five candidates for both models, all 120 plants, the 101-plant matched-date scope,
all nine historical temperature-load strata and all four 2025 calendar quarters.
The 5,000 registered operator draws define pointwise conditional intervals.

Requirements are NumPy, pandas, SciPy and the existing plotting environment.
Original data terms remain those in the package license matrices. Code is covered
by the package's author-controlled code license; source records retain attribution.

# Bridge and follow-up replay

This implementation reconstructs the 2024 weather bridge and 2025 follow-up
from declared EPA, EIA and NOAA inputs. The original 2022-2024 evaluation has
separate input identities. The run creates an operation record in its chosen
output directory.

## Run

Use Python 3.13.5 and the versions in `requirements.txt`. Install dependencies
before a network-disabled run and construct an input manifest from
`input_manifest.example.json` using the cited lawful source records.

```sh
python -B portable.py --input-manifest INPUT_MANIFEST.json --output-dir NEW_EMPTY_OUTPUT
python -B run_tests.py --work-dir NEW_EMPTY_TEST_OUTPUT
```

To verify a completed output directory:

```sh
python -B portable.py --input-manifest INPUT_MANIFEST.json --verify-existing-output COMPLETED_OUTPUT --output-dir NEW_EMPTY_VERIFICATION
```

The input roles and hashes are defined in `INPUT_CONTRACT.json`. Each operation
uses a new output directory separate from inputs and code. Missing lawful inputs
are identified as `NOT_EXECUTED`; changed hashes, duplicate identities, unknown
measurement codes and result differences are rejected.

## Computation

EPA unit-hours use the retained measurement dictionary, operating-time
integration and hourly/day aggregation. EIA 2025 qualification uses final
EIA-860/EIA-923 inputs. The 2024 frame and 135-facility follow-up population
are retained. The preliminary EIA-923 input supports the recorded revision
comparison.

Weather reconstruction uses the available GHCNh and SSOD inputs, recorded
station identities and the station-specific two-year preparation rule. The
original GSOD bridge precedes the corrected-weather branch. M1/M2 predictions
use stored spline bases, coefficients, scales, support parameters and interval
radii. The numerical workflow includes facility metrics, interval summaries,
supported-and-covered proportions and facility persistence with the stored
resampling definitions and seeds.

## Numerical Checks

`RESULT_DIFFERENCES.json` compares the declared scientific tables. Daily endpoint
tolerance is 1e-10 where specified; prediction and result tolerance is 1e-12 with
zero relative tolerance. Three weather references use canonical table signatures
covering all rows and columns apart from the execution timestamp.
`INDEPENDENT_VERIFICATION.json` evaluates support and reductions;
`WEATHER_INDEPENDENT.json` evaluates selected-hour means and recorded conflicts.

The Python execution guard controls network access, declared reads and output
locations. Local input manifests and run outputs remain separate from the
distributed code and scientific input package.

## Provenance And License

`CODE_PROVENANCE.json` records source and exported code hashes. Source meanings,
input terms and the selected author-material license scope are documented in
the data tree's `LICENSE_SCOPE.json` and `LICENSE_MATRIX.csv`.

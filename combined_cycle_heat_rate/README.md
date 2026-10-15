# Combined cycle heat rate

Release version: 1.0.2.

This package contains fixed-relation results, receiving-facility evaluations,
historical-assignment controls and figure source values. Extract the code and
data ZIPs into the same empty directory to form one
`combined_cycle_heat_rate/` root. Install the listed dependencies locally.

## Run

```bash
python3 reproduce.py verify --package-root .
python3 reproduce.py figures --package-root . --output NEW_EMPTY_OUTPUT
```

`verify` checks the complete package member list and file identities. `figures`
renders data figures from their source values. `fixed` evaluates included
daily inputs and serialized model caches, together with historical-assignment
comparisons. Choose a separate, new output directory for each operation.
All four commands accept the combined-cycle heat-rate release only.

Numerical reductions of the supplied tables are separate:

```bash
python3 -B validation/verify_reported_values.py
python3 -B validation/verify_supporting_tables.py
python3 -B validation/verify_source_composition.py
```

```bash
python3 reproduce.py fixed --package-root . --output NEW_EMPTY_OUTPUT
```

For the complete historical-mean point comparator in Supplementary Section S15.3:

```bash
python3 reproduce.py historical-mean --package-root . --input-manifest historical_mean/input_manifest.json --output /absolute/new-output
```

This command calculates 442 facility/model/scope metrics and four summaries
from the supplied 2024 weather-bridge and 2025 support records, using the
stored 5,000 operator-cluster weights. It includes the 120-facility main
panel and the existing 101-facility matched-date panel. Definitions and output
fields are documented in `historical_mean/README.md`.


## Inputs and provenance

Original-source replay uses separately obtained lawful EPA, EIA and NOAA
source files and an explicit input manifest. Original 2024 results and the
follow-up weather bridge have distinct input identities. Required missing
inputs are reported as `NOT_EXECUTED`.

Figure exports are generated outputs rather than package inputs. Permitted
derived data, model specifications, dictionaries and source values accompany
the code. Source terms and redistribution scope are in `LICENSE_MATRIX.csv`
and the attribution records.

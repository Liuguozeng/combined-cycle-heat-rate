# Historical facility information

This directory contains the retained source settings, daily model predictions,
facility changes, 1,000 identity controls and 5,000 operator-cluster draws.
Scientific result files and serialized caches are byte-identical to registered
inputs. Release paths are scientific names; prior source identities and hashes
remain in attribution records.

Extract the two release ZIPs together and use `reproduce.py` at the release root.
`fixed` prepares an explicit manifest from included registered inputs and checks
fixed full/omission-cache predictions, same-day alternatives, facility metrics,
stored matching, cluster summaries and identity-control tables. Supply the declared Python, NumPy,
pandas, SciPy, scikit-learn and joblib environment. Fitting methods and network
access are disabled by the check. Missing or altered inputs fail.

`figures` regenerates current data figures into a separate output directory.
Original 2024 GSOD results and the follow-up weather bridge remain distinct.
Offsets use full 2024 support; conditional comparisons reuse stored matched
dates. Both interval endpoints shift together without changing widths.
Permutation summaries serve as descriptive assignment controls.

Source terms and author permissions are recorded in `attribution/`. Joblib
contains executable serialization: load only trusted, verified caches.

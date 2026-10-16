# Historical facility departures retain later-year information in fixed combined-cycle heat-rate baselines

Code, fixed model specifications and derived evidence for multi-year combined-cycle heat-rate evaluation.

## Reproduce

Clone this repository, then download `combined_cycle_heat_rate_data.zip` from
[release 1.0.5](https://github.com/Liuguozeng/combined-cycle-heat-rate/releases/tag/v1.0.5) and extract it in the repository
root. Its members complete the existing `combined_cycle_heat_rate/` directory.

```bash
git clone https://github.com/Liuguozeng/combined-cycle-heat-rate.git
cd combined-cycle-heat-rate
unzip /path/to/combined_cycle_heat_rate_data.zip
cd combined_cycle_heat_rate
python3 reproduce.py verify --package-root .
python3 reproduce.py figures --package-root . --output ../generated_figures
```

Install the requirements listed inside the package before running the figure
or numerical workflows. The [package README](combined_cycle_heat_rate/README.md) describes the
available commands and input requirements. Figure files are generated outputs.

The release also provides a code ZIP for use without Git. Extract both ZIPs
into the same empty directory; their disjoint members form one package root.

## License and Sources

Original analysis and plotting code is available under the MIT license.
Source data, upstream material and derived data retain the terms documented in
`LICENSE_MATRIX.csv` in the data archive and the package attribution
records. The MIT license does not replace those source-specific terms.
Original source archives are obtained from the cited official records.

The 1.0.5 release adds the historical-level and operating-condition-response comparisons. See `combined_cycle_heat_rate/response_transfer/README.md` for the new fit-and-evaluate command.

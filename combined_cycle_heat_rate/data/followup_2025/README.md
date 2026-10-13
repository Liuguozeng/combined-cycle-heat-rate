# Weather bridge and 2025 follow-up data

`reference/` contains frozen model/support inputs, qualification descriptors,
mapping adjudications, and corrected historical comparison references.
`expected/` contains the scientific tables needed for complete result
comparison with replay outputs.

Raw EPA, EIA, GHCNh, SSOD, and download receipts must be supplied privately
under lawful access. Their identities are
declared in the Code tree's `INPUT_CONTRACT.json`; they are not redistributed
here. Raw-message-bearing expected weather tables are represented by full
canonical signatures in `expected/WEATHER_TABLE_SIGNATURES.json`; the actual
restricted expected tables are not needed by the portable runtime. Absolute
input paths are removed from public tables. Their private
original-to-export provenance remains separate from this tree.

This tree accompanies the 2024 weather-bridge and 2025 replay implementation.
Original 2022-2024 evaluation results remain under their own source identities.

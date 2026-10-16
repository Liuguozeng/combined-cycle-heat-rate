#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = 'combined_cycle_heat_rate'
ANCHORS = {'MULTIYEAR_NATIONAL_SUMMARY.csv': [{'where': {'year': '2022', 'model': 'M1', 'release_status': 'final'}, 'column': 'facilities', 'expected': 124.0, 'tolerance': 0}, {'where': {'year': '2024', 'model': 'M1', 'release_status': 'final'}, 'column': 'facilities', 'expected': 135.0, 'tolerance': 0}, {'where': {'year': '2022', 'model': 'M1', 'release_status': 'final'}, 'column': 'facility_equal_mean_coverage', 'expected': 0.954987957953282, 'tolerance': 1e-12}, {'where': {'year': '2024', 'model': 'M1', 'release_status': 'final'}, 'column': 'median_width_over_local_mae', 'expected': 6.777930740382546, 'tolerance': 0.1}, {'where': {'year': '2024', 'model': 'M2', 'release_status': 'final'}, 'column': 'median_width_over_local_mae', 'expected': 8.52853279021126, 'tolerance': 0.1}], 'COHORT_ATTRITION.csv': [{'where': {'year': '2022', 'stage': 'fixed_outcome_free_frame'}, 'column': 'facility_count', 'expected': 142.0, 'tolerance': 0}, {'where': {'year': '2024', 'stage': 'outcome_free_metadata_topology_candidate_frame'}, 'column': 'facility_count', 'expected': 182.0, 'tolerance': 0}, {'where': {'year': '2024', 'stage': 'endpoint_ge_150_facility_days'}, 'column': 'facility_count', 'expected': 135.0, 'tolerance': 0}], 'COMMON_PANEL_SUMMARY.csv': [{'where': {'year': '2022', 'model': 'M1'}, 'column': 'facilities', 'expected': 120.0, 'tolerance': 0}, {'where': {'year': '2024', 'model': 'M2'}, 'column': 'facility_equal_coverage_90', 'expected': 0.979298301314141, 'tolerance': 1e-12}], 'SUPPORTED_AND_COVERED_SUMMARY.csv': [{'where': {'year': '2022', 'model': 'M1'}, 'column': 'facility_equal_supported_and_covered', 'expected': 0.796309741910297, 'tolerance': 1e-12}, {'where': {'year': '2023', 'model': 'M2'}, 'column': 'operator_equal_supported_and_covered', 'expected': 0.8464576966688234, 'tolerance': 1e-12}], 'BIAS_PERSISTENCE.csv': [{'where': {'model': 'M1', 'year_a': '2022', 'year_b': '2023'}, 'column': 'spearman_rho', 'expected': 0.9244183623862767, 'tolerance': 1e-12}, {'where': {'model': 'M2', 'year_a': '2022', 'year_b': '2024'}, 'column': 'spearman_rho', 'expected': 0.8858879088825612, 'tolerance': 1e-12}], 'DISCOVERY_FACILITY_CONTEXT.csv': [{'where': {'facility_id': '7153'}, 'column': 'linked_ct_ca_nameplate_mw', 'expected': 1098.2, 'tolerance': 1e-12}, {'where': {'facility_id': '56047'}, 'column': 'facility_days', 'expected': 337.0, 'tolerance': 0}]}

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

checks = []
declared_lines = (ROOT / 'SHA256SUMS').read_text().splitlines()
declared_names = [line.split('  ', 1)[1] for line in declared_lines]
actual_names = {str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name != 'SHA256SUMS'}
checks.append((len(declared_names) == len(set(declared_names)), 'unique-hash-identities'))
checks.append((actual_names == set(declared_names), 'exact-payload-allowlist'))
checks.append((all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in declared_names), 'safe-relative-paths'))
for line in (ROOT / 'SHA256SUMS').read_text(encoding='utf-8').splitlines():
    expected, relative = line.split('  ', 1)
    path = ROOT / relative
    checks.append((path.exists() and sha256(path) == expected, f"hash:{relative}"))

for filename, specs in ANCHORS.items():
    with (ROOT / 'results' / filename).open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    for spec in specs:
        matches = [row for row in rows if all(row.get(key, '') == value for key, value in spec['where'].items())]
        ok = len(matches) == 1 and abs(float(matches[0][spec['column']]) - float(spec['expected'])) <= float(spec['tolerance'])
        checks.append((ok, f"anchor:{filename}:{spec['where']}:{spec['column']}"))

for contract in sorted((ROOT / 'figures').glob('Figure_*/FIGURE_NUMERIC_CONTRACT.json')):
    spec = json.loads(contract.read_text(encoding='utf-8'))
    plotted = contract.parent / 'plotted_values.csv'
    checks.append((plotted.exists() and sha256(plotted) == spec['plotted_values_sha256'], f"numeric-contract:{contract.parent.name}"))

scope_rows = json.loads((ROOT / 'REPRODUCTION_SCOPE_MAP.json').read_text(encoding='utf-8'))
required_scope_fields = {
    'artifact', 'scientific_role', 'source_version', 'verification_mode',
    'runner', 'required_inputs', 'expected_sha256', 'license_boundary',
}
allowed_modes = {'recomputed', 'hash_verified', 'presentation_derived', 'source_restricted',
                 'derived_point_reduction', 'executable_or_specification', 'response_fit_and_evaluation'}
for row in scope_rows:
    artifact = ROOT / row['artifact']
    schema_ok = set(row) == required_scope_fields and row['verification_mode'] in allowed_modes
    hash_ok = artifact.is_file() and sha256(artifact) == row['expected_sha256']
    runner = (ROOT / row['runner']).resolve()
    runner_ok = runner.is_relative_to(ROOT) and runner.is_file()
    inputs_ok = row['verification_mode'] == 'source_restricted' or all(
        (ROOT / p).resolve().is_relative_to(ROOT) and (ROOT / p).is_file()
        for p in row['required_inputs'])
    if row['verification_mode'] == 'recomputed':
        runner_ok = runner_ok and runner.name == 'rerun_submission_results.py'
    elif row['verification_mode'] == 'derived_point_reduction':
        runner_ok = (runner_ok
                     and row['runner'] == 'historical_mean/code/historical_mean_control.py'
                     and row['required_inputs'] == ['historical_mean/input_manifest.json']
                     and (ROOT / row['required_inputs'][0]).is_file())
    elif row['verification_mode'] in {'executable_or_specification', 'response_fit_and_evaluation'}:
        runner_ok = (runner_ok
                     and row['artifact'].startswith('response_transfer/')
                     and row['runner'] == 'response_transfer/analysis.py'
                     and row['required_inputs'] == ['response_transfer/input_manifest.json'])
    checks.append((schema_ok and hash_ok and runner_ok and inputs_ok, f"reproduction-scope:{row['artifact']}"))

source_provenance_path = ROOT / 'source_provenance/SOURCE_ACQUISITION_MANIFEST.csv'
with source_provenance_path.open(encoding='utf-8-sig', newline='') as handle:
    source_rows = list(csv.DictReader(handle))
required_source_fields = set(['source_group', 'source_version_or_record', 'official_record_or_endpoint', 'carrier_id', 'access_record', 'source_or_manifest_sha256', 'hash_scope', 'license_or_terms', 'scientific_role', 'raw_redistributed', 'notes'])
private_path_markers = (
    '/' + 'Users' + '/',
    chr(92) + 'Users' + chr(92),
    '工作/' + 'Code' + 'X',
)
source_schema_ok = bool(source_rows) and set(source_rows[0]) == required_source_fields
source_values_ok = all(
    re.fullmatch('[0-9a-f]{64}', row['source_or_manifest_sha256'] or '')
    and row['raw_redistributed'] == 'no'
    and not any(marker in ' '.join(row.values()) for marker in private_path_markers)
    for row in source_rows
)
checks.append((source_schema_ok and source_values_ok, 'source-acquisition-provenance'))

model_spec = json.loads((ROOT / 'model_spec/FIXED_RELATION_SPECIFICATION.json').read_text(encoding='utf-8'))
support_threshold = float(model_spec['input_support']['nn_distance_threshold'])
checks.append((abs(support_threshold - 0.09476118101237305) <= 1e-12, 'model-spec:input-support-threshold'))
checks.append((abs(float(model_spec['models']['M1']['source_residual_interval']['half_width_mmbtu_per_mwh']) - 1.1340877422842477) <= 1e-12, 'model-spec:M1-radius'))
checks.append((abs(float(model_spec['models']['M2']['source_residual_interval']['half_width_mmbtu_per_mwh']) - 1.4148093265039403) <= 1e-12, 'model-spec:M2-radius'))
with (ROOT / 'model_spec/FIXED_RELATION_COEFFICIENTS.csv').open(encoding='utf-8', newline='') as handle:
    coefficient_rows = list(csv.DictReader(handle))
checks.append((len(coefficient_rows) == 16, 'model-spec:coefficient-row-count'))


from verify_source_composition import verify as verify_source_composition
checks.extend(verify_source_composition(ROOT))

failed = [label for ok, label in checks if not ok]
print(json.dumps({'paper': PAPER, 'checks': len(checks), 'failed': failed}, indent=2))
raise SystemExit(1 if failed else 0)

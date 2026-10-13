#!/usr/bin/env python3
'Verify retained derivatives and their reductions; never fit or predict.'
from collections import defaultdict
import csv
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime
import hashlib
from itertools import combinations
import json
from pathlib import Path
from statistics import mean, median


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root):
    checks = []
    contract = json.loads((root / 'analysis_provenance/source_composition/INCLUSION_CONTRACT.json').read_text())
    add = lambda ok, label: checks.append((bool(ok), 'source-composition:' + label))
    for item in contract['outputs']:
        path = root / item['artifact']
        add(digest(path) == item['sha256'] and len(rows(path)) == item['rows'], 'frozen-output:' + path.name)
    add(set(contract['packaged_protocol_records']) == set(contract['retained_original_records']), 'protocol-source-export-membership')
    for name, expected in contract['packaged_protocol_records'].items():
        add(digest(root / 'analysis_provenance/source_composition' / name) == expected, 'protocol:' + name)
    add(contract['new_fits_this_inclusion'] == 0 and contract['result_verification_mode'] == 'hash_verified', 'no-refit-claim')
    amendment = json.loads((root / 'analysis_provenance/source_composition/ANALYSIS_AMENDMENT.json').read_text())
    freeze = json.loads((root / 'analysis_provenance/source_composition/PROTOCOL_FREEZE.json').read_text())
    add(amendment['locked_at_utc'] < freeze['frozen_at_utc'], 'amendment-before-freeze')
    original_time = contract['original_result_time_record']['records'][0]
    add(datetime.fromisoformat(freeze['frozen_at_utc'].replace('Z', '+00:00')) < datetime.fromisoformat(original_time['timestamp_utc']), 'freeze-before-recorded-result')
    add(original_time['analysis_amendment_sha256'] == contract['retained_original_records']['ANALYSIS_AMENDMENT.json'], 'historical-record-amendment-identity')
    source_ids = {str(i) for i in amendment['source_facilities']}
    full = rows(root / 'results/SOURCE_COMPOSITION_FULL_COHORT_SUMMARY.csv')
    common = rows(root / 'results/SOURCE_COMPOSITION_COMMON_PANEL_SUMMARY.csv')
    metrics = rows(root / 'results/SOURCE_COMPOSITION_FACILITY_METRICS.csv')
    panel = {r['facility_id'] for r in rows(root / 'results/COMMON_FACILITY_PANEL.csv')}
    add(len(panel) == 120, 'common-panel-identity')
    keys = [(r['subset_id'], r['year'], r['model'], r['facility_id']) for r in metrics]
    add(len(keys) == len(set(keys)) == 16254, 'unique-facility-rows')
    grouped = defaultdict(list)
    for row in metrics:
        grouped[(row['subset_id'], row['year'], row['model'])].append(row)
    reference = rows(root / 'results/FACILITY_MODEL_METRICS.csv')
    reference_ids = defaultdict(set)
    for row in reference:
        if row.get('release_status', 'final') == 'final':
            reference_ids[(row['year'], row['model'])].add(row['facility_id'])
    add(len(grouped) == 126, 'subset-year-relation-count')
    for kind, omitted_count in (('leave_one_out', 1), ('leave_two_out', 2)):
        expected = {frozenset(c) for c in combinations(source_ids, omitted_count)}
        for year in ('2022', '2023', '2024'):
            for model in ('M1', 'M2'):
                selected = [r for r in full if (r['subset_type'], r['year'], r['model']) == (kind, year, model)]
                actual = {frozenset(r['omitted_source_facilities'].split('|')) for r in selected}
                add(actual == expected and len(selected) == len(expected), f"all-combinations:{kind}:{year}:{model}")
    for scope_name, summaries in (('full', full), ('common', common)):
        for summary in summaries:
            key = (summary['subset_id'], summary['year'], summary['model'])
            values = grouped[key]
            if scope_name == 'common':
                values = [r for r in values if r['facility_id'] in panel]
            ids = {r['facility_id'] for r in values}
            expected_ids = panel if scope_name == 'common' else reference_ids[(key[1], key[2])]
            add(ids == expected_ids, f"{scope_name}:identity:{key}")
            coverage = [float(r['coverage_90']) for r in values]
            operators = defaultdict(list)
            for r in values:
                operators[r['operator_id']].append(float(r['coverage_90']))
            expected = {
                'facilities': len(values), 'operator_identifiers': len(operators),
                'states': len({r['state'] for r in values}),
                'matched_support_days': sum(int(r['matched_support_days']) for r in values),
                'facility_equal_conditional_coverage': mean(coverage),
                'operator_equal_conditional_coverage': mean(mean(v) for v in operators.values()),
                'facility_equal_signed_bias_mmbtu_per_mwh': mean(float(r['signed_bias_mmbtu_per_mwh']) for r in values),
                'median_facility_mae_mmbtu_per_mwh': median(float(r['mae_mmbtu_per_mwh']) for r in values),
                'facility_equal_mean_winkler_score_mmbtu_per_mwh': mean(float(r['mean_winkler_score']) for r in values),
                'facilities_below_0_85': sum(v < .85 for v in coverage),
                'facilities_0_85_to_0_95': sum(.85 <= v <= .95 for v in coverage),
                'facilities_above_0_95': sum(v > .95 for v in coverage),
            }
            add(all(abs(float(summary[k])-v) <= 1e-12 for k, v in expected.items()), f"{scope_name}:summary-reduction:{key}")
            included = set(summary['included_source_facilities'].split('|'))
            omitted = set(summary['omitted_source_facilities'].split('|'))
            add(included.isdisjoint(omitted) and included | omitted == source_ids and len(included) == int(summary['source_facilities_in_fit']), f"{scope_name}:source-identity:{key}")
    gate = rows(root / 'results/SOURCE_COMPOSITION_LOO_GATE.csv')
    by_key = {(r['subset_id'], r['year'], r['model']): r for r in full}
    satisfied = 0
    for row in gate:
        key = (row['subset_id'], row['year'], row['model'])
        add(all(row[k] == v for k, v in by_key[key].items()), f"gate-row-identity:{key}")
        flags = [float(row['facility_equal_conditional_coverage']) > .95, int(row['facilities_below_0_85']) > 0, int(row['facilities_above_0_95']) > 0]
        satisfied += all(flags)
        add(all(row[k] == str(v) for k, v in zip(('passes_coverage', 'retains_low_tail', 'retains_high_tail', 'passes_all'), flags + [all(flags)])), f"descriptive-flags:{key}")
    add(satisfied == 32 and len(gate) == 36, '32-of-36-related-summaries')
    add(all(int(r['facilities_below_0_85']) > 0 and int(r['facilities_above_0_95']) > 0 for r in gate), 'all-single-omissions-retain-tails')
    for row in rows(root / 'results/SOURCE_COMPOSITION_LTO_STRESS_ENVELOPE.csv'):
        selected = [r for r in full if r['subset_type'] == 'leave_two_out' and (r['year'], r['model']) == (row['year'], row['model'])]
        extrema = {k: (min if k.startswith('minimum_') else max)(float(r[k.split('_', 1)[1]]) for r in selected) for k in row if k.startswith(('minimum_', 'maximum_'))}
        add(len(selected) == int(row['subsets']) == 15 and all(abs(float(row[k])-v) <= 1e-12 for k, v in extrema.items()), f"two-source-stress-reduction:{row['year']}:{row['model']}")
    model_spec = json.loads((root / 'model_spec/FIXED_RELATION_SPECIFICATION.json').read_text())
    for row in rows(root / 'results/FULL_SOURCE_REFERENCE_VALIDATION.csv'):
        width = model_spec['models'][row['model']]['source_residual_interval']['half_width_mmbtu_per_mwh']
        add(abs(float(row['frozen_interval_half_width_mmbtu_per_mwh'])-float(width)) <= 1e-12, f"retained-interval-half-width:{row['year']}:{row['model']}")
    for item in contract['display_rows']:
        selected = [r for r in full if (r['subset_type'], r['year'], r['model']) == (item['subset_type'], item['year'], item['model'])]
        for metric, spec in item['ranges'].items():
            lo = min(float(r[metric]) for r in selected)
            hi = max(float(r[metric]) for r in selected)
            formatted = lambda v: str(Decimal(str(v)).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP))
            add(lo == spec['minimum'] and hi == spec['maximum'] and spec['display'] == formatted(lo) + ' to ' + formatted(hi), f"display:{item['subset_type']}:{item['year']}:{item['model']}:{metric}")
    return checks


if __name__ == '__main__':
    checks = verify(Path(__file__).resolve().parents[1])
    failed = [name for ok, name in checks if not ok]
    print(json.dumps({'checks': len(checks), 'failed': failed, 'new_fits': 0, 'mode': 'hash verification and derivative reductions'}, indent=2))
    raise SystemExit(bool(failed))

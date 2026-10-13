from __future__ import annotations
import numpy as np
import pandas as pd
YEARS = (2022, 2023, 2024)
MODELS = ('M1', 'M2')
BOOTSTRAPS = 5000
SEED_HEAT_RATE = 20260831

def common_facility_panel(metrics: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    'Return facilities with both frozen relations in all three final years.'
    required = metrics.loc[metrics['year'].isin(YEARS) & metrics['model'].isin(MODELS) & metrics.get('release_status', pd.Series('final', index=metrics.index)).eq('final')].copy()
    observed = required[['facility_id', 'year', 'model']].drop_duplicates().groupby('facility_id', observed=True).size()
    identities = observed.loc[observed.eq(len(YEARS) * len(MODELS))].index
    panel = required.loc[required['facility_id'].isin(identities)].copy()
    identity = {'facilities': int(panel['facility_id'].nunique()), 'operator_identifiers': int(panel['operator_id'].nunique()), 'states': int(panel['state'].nunique())}
    return (panel.sort_values(['year', 'model', 'facility_id']).reset_index(drop=True), identity)

def common_panel_gate_rows(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (year, model), group in panel.groupby(['year', 'model'], observed=True, sort=True):
        coverage = pd.to_numeric(group['coverage_90'], errors='raise')
        mean = float(coverage.mean())
        has_low = bool(coverage.lt(0.85).any())
        has_over = bool(coverage.gt(0.95).any())
        rows.append({'year': int(year), 'model': str(model), 'facilities': int(group['facility_id'].nunique()), 'facility_equal_mean_coverage': mean, 'minimum_facility_coverage': float(coverage.min()), 'maximum_facility_coverage': float(coverage.max()), 'has_facility_below_0_85': has_low, 'has_facility_above_0_95': has_over, 'gate_pass': bool(mean > 0.95 and has_low and has_over)})
    return pd.DataFrame(rows)

def evaluate_common_panel_gate(summary: pd.DataFrame) -> dict[str, object]:
    expected = {(year, model) for year in YEARS for model in MODELS}
    found = set(zip(summary['year'].astype(int), summary['model'].astype(str)))
    per_row = summary['facility_equal_mean_coverage'].gt(0.95) & summary['has_facility_below_0_85'].astype(bool) & summary['has_facility_above_0_95'].astype(bool)
    return {'passed': bool(found == expected and per_row.all()), 'expected_rows': len(expected), 'observed_rows': len(found), 'failed_rows': int((~per_row).sum())}

def supported_and_covered_facility_rows(metrics: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    'Count joint support-and-coverage directly over all endpoint days.'
    keys = ['facility_id', 'operator_id', 'model', 'year']
    matched = predictions.groupby(keys, observed=True).agg(supported_days=('covered', 'size'), supported_and_covered_days=('covered', lambda values: int(pd.Series(values).astype(bool).sum()))).reset_index()
    base_columns = [*keys, 'state', 'daily_endpoint_days', 'matched_support_days']
    base = metrics[base_columns].drop_duplicates(keys)
    out = base.merge(matched, on=keys, how='left', validate='one_to_one')
    out[['supported_days', 'supported_and_covered_days']] = out[['supported_days', 'supported_and_covered_days']].fillna(0).astype(int)
    if not out['supported_days'].eq(out['matched_support_days'].astype(int)).all():
        mismatch = out.loc[~out['supported_days'].eq(out['matched_support_days'].astype(int)), keys]
        raise RuntimeError(f"Matched-support prediction count mismatch: {mismatch.head().to_dict('records')}")
    out['supported_and_covered_fraction'] = out['supported_and_covered_days'] / out['daily_endpoint_days'].astype(float)
    return out.sort_values(['year', 'model', 'facility_id']).reset_index(drop=True)

def bootstrap_operator_mean(group: pd.DataFrame, value: str, seed: int) -> tuple[float, float]:
    operator_values = group.groupby('operator_id', observed=True)[value].mean()
    values = operator_values.to_numpy(float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))
    samples = values[indices].mean(axis=1)
    return tuple((float(value) for value in np.quantile(samples, [0.025, 0.975])))

def summarize_common_panel(panel: pd.DataFrame) -> pd.DataFrame:
    metrics = ['support_fraction', 'signed_bias_mmbtu_per_mwh', 'mae_mmbtu_per_mwh', 'coverage_90', 'interval_width_mmbtu_per_mwh', 'mean_winkler_score']
    rows: list[dict[str, object]] = []
    for (year, model), group in panel.groupby(['year', 'model'], observed=True, sort=True):
        row: dict[str, object] = {'year': int(year), 'model': str(model), 'facilities': int(group['facility_id'].nunique()), 'operator_identifiers': int(group['operator_id'].nunique()), 'states': int(group['state'].nunique())}
        for metric in metrics:
            row[f'facility_equal_{metric}'] = float(group[metric].mean())
            row[f'operator_equal_{metric}'] = float(group.groupby('operator_id', observed=True)[metric].mean().mean())
        lower, upper = bootstrap_operator_mean(group, 'coverage_90', SEED_HEAT_RATE + int(year) + (0 if model == 'M1' else 100))
        row['operator_cluster_ci95_lower_coverage'] = lower
        row['operator_cluster_ci95_upper_coverage'] = upper
        rows.append(row)
    return pd.DataFrame(rows)

def summarize_supported_and_covered(facility_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (year, model), group in facility_rows.groupby(['year', 'model'], observed=True, sort=True):
        value = 'supported_and_covered_fraction'
        lower, upper = bootstrap_operator_mean(group, value, SEED_HEAT_RATE + 5000 + int(year) + (0 if model == 'M1' else 100))
        rows.append({'year': int(year), 'model': str(model), 'facilities': int(group['facility_id'].nunique()), 'operator_identifiers': int(group['operator_id'].nunique()), 'facility_equal_supported_and_covered': float(group[value].mean()), 'operator_equal_supported_and_covered': float(group.groupby('operator_id', observed=True)[value].mean().mean()), 'operator_cluster_ci95_lower': lower, 'operator_cluster_ci95_upper': upper})
    return pd.DataFrame(rows)

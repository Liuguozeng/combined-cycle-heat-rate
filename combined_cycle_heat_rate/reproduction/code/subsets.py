from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
TOL = 1e-12

def source_subsets(facilities: list[int] | tuple[int, ...]) -> list[dict[str, object]]:
    ordered = tuple(sorted((int(value) for value in facilities)))
    rows: list[dict[str, object]] = []
    for omitted_count, subset_type, prefix in ((1, 'leave_one_out', 'LOO'), (2, 'leave_two_out', 'LTO')):
        for omitted in itertools.combinations(ordered, omitted_count):
            included = tuple((value for value in ordered if value not in omitted))
            rows.append({'subset_type': subset_type, 'subset_id': prefix + '_' + '_'.join((str(value) for value in omitted)), 'included': included, 'omitted': omitted})
    return rows

def winkler_score(actual: np.ndarray, prediction: np.ndarray, radius: float, alpha: float=0.1) -> np.ndarray:
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    lower = prediction - float(radius)
    upper = prediction + float(radius)
    score = np.full(actual.shape, 2.0 * float(radius), dtype=float)
    below = actual < lower
    above = actual > upper
    score[below] += 2.0 / alpha * (lower[below] - actual[below])
    score[above] += 2.0 / alpha * (actual[above] - upper[above])
    return score

def operator_equal_mean(frame: pd.DataFrame, column: str) -> float:
    return float(frame.groupby('operator_id', observed=True)[column].mean().mean())

def summarize_facilities(frame: pd.DataFrame) -> dict[str, object]:
    coverage = frame['coverage_90'].astype(float)
    return {'facilities': int(frame['facility_id'].nunique()), 'operator_identifiers': int(frame['operator_id'].nunique()), 'states': int(frame['state'].nunique()) if 'state' in frame else 0, 'matched_support_days': int(frame['matched_support_days'].sum()), 'facility_equal_signed_bias_mmbtu_per_mwh': float(frame['signed_bias_mmbtu_per_mwh'].mean()), 'median_facility_mae_mmbtu_per_mwh': float(frame['mae_mmbtu_per_mwh'].median()), 'facility_equal_conditional_coverage': float(coverage.mean()), 'operator_equal_conditional_coverage': operator_equal_mean(frame, 'coverage_90'), 'facilities_below_0_85': int(coverage.lt(0.85).sum()), 'facilities_0_85_to_0_95': int(coverage.ge(0.85).mul(coverage.le(0.95)).sum()), 'facilities_above_0_95': int(coverage.gt(0.95).sum()), 'facility_equal_mean_winkler_score_mmbtu_per_mwh': float(frame['mean_winkler_score'].mean())}

def join_frozen_support_rows(frozen: pd.DataFrame, endpoint: pd.DataFrame, *, year: int, model: str) -> pd.DataFrame:
    required = ['facility_id', 'date', 'gross_heat_rate_mmbtu_per_mwh', 'facility_load_fraction', 'ambient_temperature_c', 'doy_sin', 'doy_cos']
    missing = [column for column in required if column not in endpoint]
    if missing:
        raise ValueError(f'Endpoint {year} lacks required fields: {missing}')
    local_frozen = frozen.loc[frozen['model'].eq(model)].copy()
    local_endpoint = endpoint[required].copy()
    local_frozen['date'] = local_frozen['date'].astype(str)
    local_endpoint['date'] = local_endpoint['date'].astype(str)
    if local_frozen.duplicated(['facility_id', 'date']).any():
        raise ValueError(f'Frozen {year} {model} support keys are not unique')
    if local_endpoint.duplicated(['facility_id', 'date']).any():
        raise ValueError(f'Endpoint {year} keys are not unique')
    joined = local_frozen.merge(local_endpoint, on=['facility_id', 'date'], how='left', validate='one_to_one', suffixes=('_frozen', '_endpoint'))
    if len(joined) != len(local_frozen) or joined['facility_load_fraction'].isna().any():
        raise ValueError(f'Endpoint features do not cover the frozen {year} {model} support mask')
    delta = np.abs(joined['gross_heat_rate_mmbtu_per_mwh_frozen'].to_numpy(float) - joined['gross_heat_rate_mmbtu_per_mwh_endpoint'].to_numpy(float))
    if not np.isfinite(delta).all() or float(delta.max(initial=0.0)) > TOL:
        raise ValueError(f'Outcome mismatch while joining frozen {year} {model} rows')
    joined['gross_heat_rate_mmbtu_per_mwh'] = joined['gross_heat_rate_mmbtu_per_mwh_endpoint']
    joined['year'] = int(year)
    return joined

def facility_metrics(rows: pd.DataFrame, prediction: np.ndarray, radius: float, subset: dict[str, object], year: int, model: str) -> pd.DataFrame:
    frame = rows[['facility_id', 'operator_id', 'state']].copy()
    frame['actual'] = rows['gross_heat_rate_mmbtu_per_mwh'].to_numpy(float)
    frame['prediction'] = np.asarray(prediction, dtype=float)
    frame['error'] = frame['prediction'] - frame['actual']
    frame['absolute_error'] = np.abs(frame['error'])
    frame['covered'] = frame['absolute_error'].le(float(radius))
    frame['winkler_score'] = winkler_score(frame['actual'].to_numpy(float), frame['prediction'].to_numpy(float), radius)
    grouped = frame.groupby(['facility_id', 'operator_id', 'state'], as_index=False, observed=True).agg(matched_support_days=('actual', 'size'), signed_bias_mmbtu_per_mwh=('error', 'mean'), mae_mmbtu_per_mwh=('absolute_error', 'mean'), coverage_90=('covered', 'mean'), mean_winkler_score=('winkler_score', 'mean')).sort_values('facility_id').reset_index(drop=True)
    grouped.insert(0, 'model', model)
    grouped.insert(0, 'year', int(year))
    grouped.insert(0, 'included_source_facilities', '|'.join(map(str, subset['included'])))
    grouped.insert(0, 'omitted_source_facilities', '|'.join(map(str, subset['omitted'])))
    grouped.insert(0, 'source_facilities_in_fit', len(subset['included']))
    grouped.insert(0, 'subset_id', subset['subset_id'])
    grouped.insert(0, 'subset_type', subset['subset_type'])
    return grouped

def add_subset_identity(summary: dict[str, object], subset: dict[str, object], year: int, model: str) -> dict[str, object]:
    return {'subset_type': subset['subset_type'], 'subset_id': subset['subset_id'], 'source_facilities_in_fit': len(subset['included']), 'omitted_source_facilities': '|'.join(map(str, subset['omitted'])), 'included_source_facilities': '|'.join(map(str, subset['included'])), 'year': int(year), 'model': model, **summary}

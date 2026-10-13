from __future__ import annotations
import numpy as np
import pandas as pd
MODELS = ('M1', 'M2')
N_BOOT = 5000
SEED = 20260827
NOMINAL_LOW = 0.85
NOMINAL_HIGH = 0.95

def week_cluster_coverage_ci(frame: pd.DataFrame, prediction: np.ndarray, radius: float, *, n_boot: int=N_BOOT, seed: int=SEED) -> tuple[float, float]:
    local = frame[['week', 'gross_heat_rate_mmbtu_per_mwh']].reset_index(drop=True).copy()
    local['prediction'] = np.asarray(prediction, dtype=float)
    groups = list(local.groupby('week', observed=True).indices.values())
    if not groups:
        return (float('nan'), float('nan'))
    error = local['prediction'].to_numpy(float) - local['gross_heat_rate_mmbtu_per_mwh'].to_numpy(float)
    covered_by_week = np.array([np.sum(np.abs(error[indices]) <= radius) for indices in groups], dtype=float)
    days_by_week = np.array([len(indices) for indices in groups], dtype=float)
    rng = np.random.default_rng(seed)
    selected = rng.integers(0, len(groups), size=(n_boot, len(groups)))
    values = covered_by_week[selected].sum(axis=1) / days_by_week[selected].sum(axis=1)
    return (float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)))

def operator_cluster_mean_ci(frame: pd.DataFrame, value_column: str, *, n_boot: int=N_BOOT, seed: int=SEED) -> tuple[float, float]:
    operators = sorted(frame['operator_id'].dropna().unique())
    if not operators:
        return (float('nan'), float('nan'))
    sums = np.array([frame.loc[frame['operator_id'].eq(operator), value_column].sum() for operator in operators], dtype=float)
    counts = np.array([frame['operator_id'].eq(operator).sum() for operator in operators], dtype=float)
    rng = np.random.default_rng(seed)
    selected = rng.choice(np.arange(len(operators)), size=(n_boot, len(operators)), replace=True)
    values = sums[selected].sum(axis=1) / counts[selected].sum(axis=1)
    return (float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)))

def winkler_score(actual: np.ndarray, prediction: np.ndarray, radius: float) -> np.ndarray:
    return stage3.winkler_score(actual, prediction, radius)

def evaluate_facility(daily: pd.DataFrame, source: pd.DataFrame, fitted: dict[str, object], radii: dict[str, float], row: pd.Series) -> tuple[list[dict[str, object]], list[pd.DataFrame]]:
    support_mask, support_meta = baseline.matched_support_mask(source, daily)
    matched = daily.loc[support_mask].copy()
    metrics: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    for model_index, model_name in enumerate(MODELS):
        all_prediction = baseline.predict_model(model_name, fitted[model_name], daily)
        matched_prediction = all_prediction[support_mask]
        if len(matched) < 3:
            continue
        actual = matched['gross_heat_rate_mmbtu_per_mwh'].to_numpy(float)
        error = matched_prediction - actual
        radius = radii[model_name]
        coverage = float(np.mean(np.abs(error) <= radius))
        coverage_ci_low, coverage_ci_high = week_cluster_coverage_ci(matched, matched_prediction, radius, seed=SEED + int(row['facility_id']) * 10 + model_index)
        mae = float(np.mean(np.abs(error)))
        metrics.append({'facility_id': int(row['facility_id']), 'facility_name': str(row['facility_name']), 'operator_id': int(row['operator_id']), 'operator_name': str(row['operator_name']), 'state': str(row['state']), 'model': model_name, 'daily_endpoint_days': int(len(daily)), 'matched_support_days': int(len(matched)), 'support_fraction': float(support_meta['test_support_fraction']), 'nn_distance_threshold': float(support_meta['nn_distance_threshold']), 'coverage_90': coverage, 'coverage_ci_low': coverage_ci_low, 'coverage_ci_high': coverage_ci_high, 'coverage_direction': 'undercoverage' if coverage < NOMINAL_LOW else 'overcoverage' if coverage > NOMINAL_HIGH else 'nominal_band', 'source_radius_mmbtu_per_mwh': float(radius), 'interval_width_mmbtu_per_mwh': float(2 * radius), 'mae_mmbtu_per_mwh': mae, 'interval_width_over_local_mae': float(2 * radius / mae) if mae else np.inf, 'signed_bias_mmbtu_per_mwh': float(np.mean(error)), 'mean_winkler_score': float(np.mean(winkler_score(actual, matched_prediction, radius)))})
        carrier = matched[['date', 'week', 'gross_heat_rate_mmbtu_per_mwh']].copy()
        carrier['facility_id'] = int(row['facility_id'])
        carrier['operator_id'] = int(row['operator_id'])
        carrier['state'] = str(row['state'])
        carrier['model'] = model_name
        carrier['prediction'] = matched_prediction
        carrier['residual_prediction_minus_observed'] = error
        carrier['source_radius_mmbtu_per_mwh'] = float(radius)
        carrier['covered'] = np.abs(error) <= radius
        predictions.append(carrier)
    return (metrics, predictions)

def summarize_national(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_index, (model, group) in enumerate(metrics.groupby('model')):
        coverage_low, coverage_high = operator_cluster_mean_ci(group, 'coverage_90', seed=SEED + model_index)
        rows.append({'model': model, 'facilities': int(group['facility_id'].nunique()), 'operators': int(group['operator_id'].nunique()), 'states': int(group['state'].nunique()), 'facility_equal_mean_coverage': float(group['coverage_90'].mean()), 'operator_cluster_ci_low': coverage_low, 'operator_cluster_ci_high': coverage_high, 'facilities_undercoverage': int(group['coverage_direction'].eq('undercoverage').sum()), 'facilities_nominal': int(group['coverage_direction'].eq('nominal_band').sum()), 'facilities_overcoverage': int(group['coverage_direction'].eq('overcoverage').sum()), 'median_width_over_local_mae': float(group['interval_width_over_local_mae'].median()), 'median_mae_mmbtu_per_mwh': float(group['mae_mmbtu_per_mwh'].median()), 'median_absolute_signed_bias': float(group['signed_bias_mmbtu_per_mwh'].abs().median()), 'operator_equal_mean_coverage': float(group.groupby('operator_id')['coverage_90'].mean().mean())})
    summary = pd.DataFrame(rows)
    return summary

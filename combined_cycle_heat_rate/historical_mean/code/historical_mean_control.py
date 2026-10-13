'Explicit-input, post hoc comparison of historical levels and centered curves.'
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ['model', 'facility_id', 'date']
TOL = 1e-10
INPUTS = {
    'history': 'FULL_2024_BRIDGE_PREDICTIONS.csv.gz',
    'alternatives': 'FULL_MAIN_2025_ALTERNATIVES.csv.gz',
    'conditional_alternatives': 'FULL_CONDITIONAL_2025_ALTERNATIVES.csv.gz',
    'pairs': 'CONDITIONAL_DAY_PAIRS.csv.gz',
    'identity': 'FACILITY_IDENTITY_AND_OPERATOR_CHANGES.csv',
    'bootstrap': 'BOOTSTRAP_OPERATOR_WEIGHTS.npz',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def unique(frame, keys):
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError('null or duplicate identity: ' + str(keys))


def dates(frame, year):
    frame = frame.copy()
    parsed = pd.to_datetime(frame.date, errors='raise')
    if not frame.year.eq(year).all() or not parsed.dt.year.eq(year).all():
        raise ValueError('wrong year or target-year leakage')
    if not parsed.eq(parsed.dt.normalize()).all():
        raise ValueError('not a calendar date')
    unique(frame, KEY)
    if not np.isfinite(frame[['prediction', 'observed']].to_numpy(float)).all():
        raise ValueError('nonfinite value')
    return frame


def same_observations(frame):
    groups = [rows.sort_values(['facility_id', 'date']).reset_index(drop=True)
              for _, rows in frame.groupby('model')]
    if len(groups) != 2 or set(frame.model) != {'M1', 'M2'}:
        raise ValueError('wrong model identities')
    if not groups[0][['facility_id', 'date']].equals(groups[1][['facility_id', 'date']]):
        raise ValueError('model-specific support differs')
    if not np.allclose(groups[0].observed, groups[1].observed, rtol=0, atol=TOL):
        raise ValueError('model-specific observations differ')


def historical_levels(history, ids):
    history = dates(history, 2024)
    history = history[history.facility_id.isin(ids)].copy()
    same_observations(history)
    if set(history.model) != {'M1', 'M2'}:
        raise ValueError('wrong models')
    for model, rows in history.groupby('model'):
        if set(rows.facility_id) != set(ids):
            raise ValueError('missing historical facility')
    keys = history.groupby('model').apply(lambda x: set(zip(x.facility_id, x.date)), include_groups=False)
    if keys.iloc[0] != keys.iloc[1]:
        raise ValueError('model-specific historical support differs')
    history['offset'] = history.prediction - history.observed
    return history.groupby(['model', 'facility_id'], as_index=False).agg(
        historical_observed_mean=('observed', 'mean'), historical_prediction_mean=('prediction', 'mean'),
        offset_2024=('offset', 'mean'), historical_days=('date', 'size'))


def centered_predictions(target, levels):
    target = dates(target, 2025)
    same_observations(target)
    result = target.merge(levels, on=['model', 'facility_id'], validate='many_to_one')
    if len(result) != len(target):
        raise ValueError('missing history for evaluation facility')
    result['facility_centered'] = result.prediction - result.offset_2024
    result['historical_mean'] = result.historical_observed_mean
    alternative_form = result.historical_mean + result.prediction - result.historical_prediction_mean
    if not np.allclose(result.facility_centered, alternative_form, rtol=0, atol=TOL):
        raise ValueError('centering algebra failed')
    for alternative in ('facility_centered', 'historical_mean'):
        result['error_' + alternative] = result[alternative] - result.observed
        result['absolute_error_' + alternative] = result['error_' + alternative].abs()
    return result


def facility_metrics(days, scope):
    metrics = days.groupby(['model', 'facility_id'], as_index=False).agg(
        days=('date', 'size'), mae_facility_centered=('absolute_error_facility_centered', 'mean'),
        mae_historical_mean=('absolute_error_historical_mean', 'mean'),
        bias_facility_centered=('error_facility_centered', 'mean'),
        bias_historical_mean=('error_historical_mean', 'mean'))
    metrics['scope'] = scope
    metrics['mae_centered_minus_historical_mean'] = metrics.mae_facility_centered - metrics.mae_historical_mean
    metrics['bias_centered_minus_historical_mean'] = metrics.bias_facility_centered - metrics.bias_historical_mean
    metrics['improved'] = metrics.mae_centered_minus_historical_mean.lt(0)
    metrics['worsened'] = metrics.mae_centered_minus_historical_mean.gt(0)
    return metrics


def bootstrap_summary(metrics, weights_file, identities):
    z = np.load(weights_file)
    if z['operator_counts'].shape != (5000, 91) or not np.array_equal(z['replicate'], np.arange(1, 5001)):
        raise ValueError('wrong registered bootstrap')
    ids = identities.sort_values('facility_id')
    if not np.array_equal(z['facility_ids'], ids.facility_id) or not np.array_equal(z['facility_operator_ids'], ids.operator_id_2024):
        raise ValueError('bootstrap identity mismatch')
    op = np.sort(ids.operator_id_2024.unique())
    if not np.array_equal(z['operator_ids'], op):
        raise ValueError('operator identity mismatch')
    rng = np.random.default_rng(2026100736)
    expected = np.array([np.bincount(rng.integers(0, 91, 91), minlength=91) for _ in range(5000)])
    if not np.array_equal(expected, z['operator_counts']):
        raise ValueError('bootstrap seed/multiplicity mismatch')
    index = {int(v): i for i, v in enumerate(op)}
    summaries, replicates = [], []
    for (scope, model), rows in metrics.groupby(['scope', 'model']):
        rows = rows.merge(ids[['facility_id', 'operator_id_2024']], on='facility_id', validate='one_to_one').sort_values('facility_id')
        weights = z['operator_counts'][:, [index[int(v)] for v in rows.operator_id_2024]]
        denominator = weights.sum(1)
        if (denominator == 0).any():
            raise ValueError('empty bootstrap sample')
        values = weights @ rows.mae_centered_minus_historical_mean.to_numpy() / denominator
        low, high = np.quantile(values, [.025, .975])
        summaries.append(dict(scope=scope, model=model, facilities=len(rows), days=int(rows.days.sum()),
            operator_clusters_observed=int(rows.operator_id_2024.nunique()), registered_operator_clusters=91,
            mae_facility_centered=float(rows.mae_facility_centered.mean()), mae_historical_mean=float(rows.mae_historical_mean.mean()),
            paired_mae_delta=float(rows.mae_centered_minus_historical_mean.mean()), ci_low=float(low), ci_high=float(high),
            improved=int(rows.improved.sum()), worsened=int(rows.worsened.sum()), ties=int((~rows.improved & ~rows.worsened).sum()),
            valid_replicates=5000, units='MMBtu/MWh', estimand='facility_equal_delta_with_fixed_2024_operator_cluster_draws'))
        replicates.append(pd.DataFrame(dict(scope=scope, model=model, replicate=np.arange(1, 5001),
            facility_weight_sum=denominator, paired_mae_delta=values)))
    return pd.DataFrame(summaries), pd.concat(replicates, ignore_index=True)


def prepare(results, manifest):
    files = {k: dict(path=str((results / name).resolve()), sha256=sha(results / name)) for k, name in INPUTS.items()}
    save_json(manifest, dict(schema_version=1, files=files, spec=dict(history_year=2024, target_year=2025,
        models=['M1', 'M2'], primary_facilities=120, conditional_facilities=101, conditional_pairs=6723,
        prediction_tolerance=TOL, bootstrap_replicates=5000, bootstrap_seed=2026100736,
        historical_level='arithmetic_mean_of_complete_2024_matched_support_observations',
        primary_contrast='mae_facility_centered_minus_historical_mean', conditional_rematching=False,
        model_refits=0, interval_recalibration=False, post_hoc=True)))


def run(manifest, output):
    cfg = json.loads(manifest.read_text())
    if cfg.get('schema_version') != 1 or set(cfg['files']) != set(INPUTS):
        raise ValueError('invalid input manifest')
    expected_spec = dict(history_year=2024, target_year=2025, models=['M1', 'M2'], primary_facilities=120,
        conditional_facilities=101, conditional_pairs=6723, prediction_tolerance=TOL,
        bootstrap_replicates=5000, bootstrap_seed=2026100736, historical_level='arithmetic_mean_of_complete_2024_matched_support_observations',
        primary_contrast='mae_facility_centered_minus_historical_mean', conditional_rematching=False,
        model_refits=0, interval_recalibration=False, post_hoc=True)
    if cfg.get('spec') != expected_spec:
        raise ValueError('locked control specification changed')
    paths = {}
    for role, item in cfg['files'].items():
        path = Path(item['path'])
        if not path.is_absolute():
            path = manifest.parent / path
        if not path.is_file() or sha(path) != item['sha256']:
            raise ValueError('input identity mismatch: ' + role)
        paths[role] = path
    if output.exists() and any(output.iterdir()):
        raise ValueError('nonempty results must not be overwritten')
    output.mkdir(parents=True, exist_ok=True)
    identity = pd.read_csv(paths['identity'])
    unique(identity, ['facility_id'])
    if len(identity) != 120 or identity.operator_id_2024.nunique() != 91:
        raise ValueError('wrong primary panel')
    ids = set(identity.facility_id)
    levels = historical_levels(pd.read_csv(paths['history']), ids)
    alternatives = pd.read_csv(paths['alternatives'])
    if set(alternatives.alternative) != {'original', 'pooled', 'facility'}:
        raise ValueError('unknown alternative')
    for alternative, rows in alternatives.groupby('alternative'):
        dates(rows, 2025)
        if set(rows.facility_id) != ids or set(rows.model) != {'M1', 'M2'}:
            raise ValueError('missing target facility/model')
    target = alternatives[alternatives.alternative.eq('original')].copy()
    days = centered_predictions(target.drop(columns=['offset_2024', 'historical_days']), levels)
    saved = alternatives[alternatives.alternative.eq('facility')][KEY + ['prediction']]
    joined = days.merge(saved, on=KEY, suffixes=('', '_saved'), validate='one_to_one')
    if len(joined) != len(saved) or not np.allclose(joined.facility_centered, joined.prediction_saved, rtol=0, atol=TOL):
        raise ValueError('new centering disagrees with frozen predictions')
    pairs = pd.read_csv(paths['pairs'])
    pairs = pairs[pairs.eligible.eq(True)].copy()
    for year in (2024, 2025):
        unique(pairs, ['facility_id', 'date_' + str(year)])
        if not pd.to_datetime(pairs['date_' + str(year)]).dt.year.eq(year).all():
            raise ValueError('conditional year mismatch')
    if len(pairs) != 6723 or pairs.facility_id.nunique() != 101:
        raise ValueError('conditional support identity changed')
    subset = days.merge(pairs[['facility_id', 'date_2025']].rename(columns={'date_2025': 'date'}), on=['facility_id', 'date'], validate='many_to_one')
    if len(subset) != 2 * 6723:
        raise ValueError('missing conditional date')
    saved_cond = pd.read_csv(paths['conditional_alternatives'])
    saved_cond = saved_cond[saved_cond.alternative.eq('facility')]
    if set(map(tuple, subset[KEY].to_numpy())) != set(map(tuple, saved_cond[KEY].to_numpy())):
        raise ValueError('conditional dates disagree with frozen branch')
    all_metrics = pd.concat([facility_metrics(days, 'main'), facility_metrics(subset, 'conditional')], ignore_index=True)
    summary, draws = bootstrap_summary(all_metrics, paths['bootstrap'], identity)
    levels.to_csv(output / 'HISTORICAL_LEVELS.csv', index=False)
    all_metrics.to_csv(output / 'FACILITY_MEAN_CONTROL.csv', index=False)
    summary.to_csv(output / 'SUMMARY_MEAN_CONTROL.csv', index=False)
    draws.to_csv(output / 'BOOTSTRAP_MEAN_CONTROL.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    columns = KEY + ['year', 'observed', 'prediction', 'historical_days', 'historical_observed_mean', 'historical_prediction_mean', 'offset_2024', 'facility_centered', 'historical_mean', 'error_facility_centered', 'error_historical_mean']
    days[columns].to_csv(output / 'DAILY_MEAN_CONTROL.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    for role, path in paths.items():
        if sha(path) != cfg['files'][role]['sha256']:
            raise ValueError('input changed while evaluating')
    save_json(output / 'COMPLETE.json', dict(status='POINT_COMPARISON_COMPLETE',
        post_hoc=True, no_2025_data_in_historical_level=True, existing_bootstrap_reused=True,
        new_model_fits=0, no_new_intervals=True, algebra_max_abs_error=float(np.max(np.abs(days.facility_centered - (days.historical_mean + days.prediction - days.historical_prediction_mean)))),
        summaries=summary.to_dict('records'), input_manifest_sha256=sha(manifest),
        outputs={p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}))
    print(summary.to_string(index=False))
    print('POINT_COMPARISON_COMPLETE')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-manifest', type=Path, required=True)
    parser.add_argument('--prepare-from', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.prepare_from:
        prepare(args.prepare_from, args.input_manifest)
    else:
        if args.output is None:
            parser.error('--output is required for evaluation')
        run(args.input_manifest, args.output)

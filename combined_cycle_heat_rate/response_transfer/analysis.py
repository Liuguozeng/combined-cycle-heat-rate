"""Estimate declared response alternatives from historical records and evaluate 2025."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from core import RunGuard,require_year,fit_within_response,fit_local_response,feature_matrix,save_json,sha
CANDIDATES=['historical_mean','historical_median','original_centered','within_source_centered','local_response']

def weighted_quantiles(values, weights, probabilities):
    order = np.argsort(values, kind='stable')
    x = np.asarray(values)[order]
    w = np.asarray(weights)[order]
    cumulative = np.cumsum(w) / w.sum()
    return np.array([x[min(np.searchsorted(cumulative, q, side='left'), len(x) - 1)] for q in probabilities])

def main():
    guard = RunGuard('response')
    OUT.mkdir(exist_ok=False)
    spec_path = PACKAGE / 'followup2025_evidence/reference/model_spec/FIXED_RELATION_SPECIFICATION.json'
    spec = json.loads(spec_path.read_text())
    source_path = PACKAGE / 'reproduction/reference_inputs/source_daily/SIX_FACILITY_DAILY_ENDPOINT.csv.gz'
    source = pd.read_csv(source_path)
    require_year(source, 2024)
    assert len(source) == 1828 and source.facility_id.nunique() == 6
    history = pd.read_csv(RESULTS / 'FULL_2024_BRIDGE_PREDICTIONS.csv.gz')
    require_year(history, 2024)
    future = pd.read_csv(RESULTS / 'FULL_MAIN_2025_ALTERNATIVES.csv.gz')
    future = future[future.alternative.eq('original')].copy()
    require_year(future, 2025)
    matched = pd.read_csv(RESULTS / 'FULL_CONDITIONAL_2025_ALTERNATIVES.csv.gz')
    matched = matched[matched.alternative.eq('original')].copy()
    require_year(matched, 2025)
    identity = pd.read_csv(RESULTS / 'FACILITY_IDENTITY_AND_OPERATOR_CHANGES.csv')
    ids = sorted(identity.facility_id.astype(int))
    assert len(ids) == 120
    history = history[history.facility_id.isin(ids)].copy()
    for frame in [history, future, matched]:
        if frame.duplicated(['facility_id', 'date', 'model']).any():
            raise ValueError('Duplicated prediction key')
    assert set(future.facility_id) == set(ids) and len(future) == 2 * 29443
    assert matched.facility_id.nunique() == 101 and len(matched) == 2 * 6723
    source_y = source.gross_heat_rate_mmbtu_per_mwh.to_numpy(float)
    source_groups = source.facility_id.to_numpy()
    params = {}
    calibration = []
    predictions = []
    checks = []
    for model in ['M1', 'M2']:
        ms = spec['models'][model]['pipeline']
        xs = feature_matrix(ms, source)
        coef = np.asarray(ms['ridge']['coefficients_after_scaling'])
        intercept = ms['ridge']['intercept']
        within = fit_within_response(xs, source_y, source_groups, alpha=ms['ridge']['alpha'])
        hist = history[history.model.eq(model)].copy().sort_values(['facility_id', 'date'])
        target = future[future.model.eq(model)].copy().sort_values(['facility_id', 'date'])
        xh = feature_matrix(ms, hist)
        xt = feature_matrix(ms, target)
        p_old = xh @ coef + intercept
        q_old = xt @ coef + intercept
        error = max(float(np.max(abs(p_old - hist.prediction.to_numpy()))), float(np.max(abs(q_old - target.prediction.to_numpy()))))
        if error > 1e-12:
            raise ValueError('Original frozen prediction changed')
        checks.append(dict(model=model, maximum_original_prediction_difference=error))
        hist['within_prediction'] = xh @ within
        target['within_prediction'] = xt @ within
        hist['original_replayed'] = p_old
        for facility, h in hist.groupby('facility_id', sort=True):
            t = target[target.facility_id.eq(facility)].copy()
            mean = float(h.observed.mean())
            median = float(h.observed.median())
            oldmean = float(h.original_replayed.mean())
            withinmean = float(h.within_prediction.mean())
            alpha = fit_local_response(h.original_replayed.to_numpy() - oldmean, h.observed.to_numpy() - mean)
            calibration.append(dict(model=model, facility_id=int(facility), history_year=2024, history_days=len(h), observed_mean=mean, observed_median=median, original_prediction_mean=oldmean, within_prediction_mean=withinmean, response_coefficient=alpha, response_identifiable=bool(np.any(h.original_replayed.to_numpy() != oldmean))))
            delta = t.prediction.to_numpy() - oldmean
            candidates = dict(historical_mean=np.full(len(t), mean), historical_median=np.full(len(t), median), original_centered=mean + delta, within_source_centered=mean + t.within_prediction.to_numpy() - withinmean, local_response=mean + alpha * delta)
            for candidate, pred in candidates.items():
                r = t[['facility_id', 'date', 'model', 'observed', 'ambient_temperature_c', 'facility_load_fraction']].copy()
                r['candidate'] = candidate
                r['prediction'] = pred
                r['absolute_error'] = abs(pred - r.observed)
                r['signed_error'] = pred - r.observed
                predictions.append(r)
        params[model] = dict(within_coefficients=within.tolist(), original_coefficients=coef.tolist(), ridge_alpha=1.0, training_rows=len(source), training_facilities=sorted(source_groups.astype(int).tolist() and np.unique(source_groups).astype(int).tolist()), source_feature_means=source.assign(**{f'feature_{i}': xs[:, i] for i in range(xs.shape[1])}).groupby('facility_id')[[f'feature_{i}' for i in range(xs.shape[1])]].mean().to_dict('index'))
        guard.check()
    predictions = pd.concat(predictions, ignore_index=True)
    predictions.to_csv(OUT / 'daily_predictions.csv.gz', index=False, float_format='%.17g')
    pd.DataFrame(calibration).to_csv(OUT / 'historical_calibration.csv', index=False, float_format='%.17g')
    save_json(OUT / 'fitted_response_parameters.json', params)
    save_json(OUT / 'original_prediction_checks.json', checks)
    h = history[history.model.eq('M1')].copy()
    weights = 1 / h.groupby('facility_id').facility_id.transform('size').to_numpy()
    boundaries = {c: weighted_quantiles(h[c].to_numpy(), weights, [1 / 3, 2 / 3]).tolist() for c in ['ambient_temperature_c', 'facility_load_fraction']}
    save_json(OUT / 'historical_stratum_boundaries.json', dict(history_year=2024, facility_equal=True, boundaries=boundaries))
    predictions['temperature_band'] = np.searchsorted(boundaries['ambient_temperature_c'], predictions.ambient_temperature_c, side='right') + 1
    predictions['load_band'] = np.searchsorted(boundaries['facility_load_fraction'], predictions.facility_load_fraction, side='right') + 1
    predictions['quarter'] = pd.to_datetime(predictions.date).dt.quarter
    matchkeys = set(zip(matched.facility_id, matched.date, matched.model))
    mask = np.array([(f, d, m) in matchkeys for f, d, m in zip(predictions.facility_id, predictions.date, predictions.model)])
    main_metrics = predictions.groupby(['model', 'candidate', 'facility_id']).agg(mae=('absolute_error', 'mean'), bias=('signed_error', 'mean'), days=('date', 'size')).reset_index()
    matched_metrics = predictions.loc[mask].groupby(['model', 'candidate', 'facility_id']).agg(mae=('absolute_error', 'mean'), bias=('signed_error', 'mean'), days=('date', 'size')).reset_index()
    frames = [main_metrics.assign(scope='main'), matched_metrics.assign(scope='matched_dates')]
    metrics = pd.concat(frames, ignore_index=True)
    metrics.to_csv(OUT / 'facility_metrics.csv', index=False, float_format='%.17g')
    with np.load(RESULTS / 'BOOTSTRAP_OPERATOR_WEIGHTS.npz', allow_pickle=False) as n:
        operators = n['operator_ids']
        counts = n['operator_counts']
        fid = n['facility_ids']
        ops = n['facility_operator_ids']
    assert list(fid) == ids and len(operators) == 91 and (counts.shape == (5000, 91))
    op_index = {int(op): k for k, op in enumerate(operators)}
    weights = counts[:, [op_index[int(op)] for op in ops]].astype(float)
    summaries = []
    for (scope, model), group in metrics.groupby(['scope', 'model']):
        wide = group.pivot(index='facility_id', columns='candidate', values='mae').sort_index()
        assert wide.notna().all().all()
        use = np.array([ids.index(int(f)) for f in wide.index])
        w = weights[:, use]
        b = {candidate: w @ wide[candidate].to_numpy() / w.sum(axis=1) for candidate in CANDIDATES}
        for candidate in CANDIDATES:
            values = wide[candidate]
            diff = values - wide.historical_mean
            dm = values - wide.historical_median
            row = dict(scope=scope, model=model, candidate=candidate, facilities=len(wide), facility_days=int(group[group.candidate.eq(candidate)].days.sum()), mae=float(values.mean()), minus_historical_mean=float(diff.mean()), minus_historical_median=float(dm.mean()), improved_vs_mean=int((diff < 0).sum()), worsened_vs_mean=int((diff > 0).sum()), tied_vs_mean=int((diff == 0).sum()))
            for name, values0 in [('mae', b[candidate]), ('minus_historical_mean', b[candidate] - b['historical_mean']), ('minus_historical_median', b[candidate] - b['historical_median']), ('minus_original_centered', b[candidate] - b['original_centered'])]:
                lo, hi = np.quantile(values0, [0.025, 0.975])
                row[name + '_ci95_lower'] = float(lo)
                row[name + '_ci95_upper'] = float(hi)
            summaries.append(row)
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / 'candidate_summaries.csv', index=False, float_format='%.17g')
    original_expected = {('main', 'M1'): 0.26, ('main', 'M2'): 0.256, ('matched_dates', 'M1'): 0.215, ('matched_dates', 'M2'): 0.211}
    for (scope, model), val in original_expected.items():
        actual = float(summary.query('scope==@scope and model==@model and candidate=="original_centered"').mae.iloc[0])
        assert abs(actual - val) < 0.0005
    strata = []
    for kind, columns in [('conditions', ['temperature_band', 'load_band']), ('quarter', ['quarter'])]:
        fm = predictions.groupby(['model', 'candidate', 'facility_id'] + columns).agg(mae=('absolute_error', 'mean'), days=('date', 'size')).reset_index()
        for grouping, local in fm.groupby(['model'] + columns):
            wide = local.pivot(index='facility_id', columns='candidate', values='mae')
            for candidate in CANDIDATES:
                strata.append(dict(kind=kind, model=grouping[0], stratum='|'.join(map(str, grouping[1:])), candidate=candidate, facilities=len(wide), facility_days=int(local[local.candidate.eq(candidate)].days.sum()), mae=float(wide[candidate].mean()), minus_historical_mean=float((wide[candidate] - wide.historical_mean).mean()), minus_historical_median=float((wide[candidate] - wide.historical_median).mean())))
    pd.DataFrame(strata).to_csv(OUT / 'stratum_summaries.csv', index=False, float_format='%.17g')
    save_json(OUT / 'RESULTS.json', dict(protocol_sha256=sha(PACKAGE / 'response_transfer/protocol.json'), source_sha256=sha(source_path), specification_sha256=sha(spec_path), calibration_uses_2025=False, original_models_overwritten=False, summary=summary.to_dict('records'), runtime=guard.finish()))
    print(summary[['scope', 'model', 'candidate', 'mae', 'minus_historical_mean', 'minus_historical_mean_ci95_lower', 'minus_historical_mean_ci95_upper', 'improved_vs_mean']].to_string(index=False), flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    PACKAGE=a.package_root.resolve();RESULTS=PACKAGE/'historical_transfer/results';OUT=a.output.resolve()
    if OUT.exists() or OUT.is_relative_to(PACKAGE):raise ValueError('A new output outside the input package is required')
    main()

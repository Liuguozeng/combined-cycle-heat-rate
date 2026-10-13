'Bounded, post hoc combined_cycle_heat_rate offset-transfer and source-omission analysis. No fits.'
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import resource
import signal
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

METRICS = ['mae', 'bias', 'coverage', 'radius', 'width', 'winkler']
KEYS = ['facility_id', 'year', 'date']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False, default=str) + '\n')


def save_csv(path, data):
    data.to_csv(path, index=False, compression={'method': 'gzip', 'mtime': 0} if str(path).endswith('.gz') else None)


def check_manifest(manifest):
    for role, item in manifest['files'].items():
        p = Path(item['path'])
        if not p.is_absolute():
            raise ValueError(f"input path must be absolute: {role}")
        if not p.is_file():
            raise ValueError(f"missing input: {role}")
        if sha(p) != item['sha256']:
            raise ValueError(f"input hash mismatch: {role}")
    for item in manifest.get('archives', []):
        if not Path(item['path']).is_file() or sha(item['path']) != item['sha256']:
            raise ValueError('archive hash mismatch')


def unique(frame, keys):
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError(f"null or duplicate identity: {keys}")


def validate_dates(frame, years, keys):
    d = frame.copy()
    if not set(keys + ['date', 'year']).issubset(d.columns):
        raise ValueError('missing date/year/identity input columns')
    d['date'] = pd.to_datetime(d.date, errors='raise')
    if not d.year.isin(years).all() or not d.date.dt.year.eq(d.year).all():
        raise ValueError('unknown or inconsistent year')
    if not d.date.eq(d.date.dt.normalize()).all():
        raise ValueError('not a calendar date')
    unique(d, keys)
    return d


def require_supported(source, target, support):
    cols = support['features_in_order']
    x = target[cols].to_numpy(float)
    if not np.isfinite(x).all():
        raise ValueError('nonfinite support coordinates')
    mean, scale = np.array(support['mean']), np.array(support['scale'])
    z = (x - mean) / scale
    tree = cKDTree((source[cols].to_numpy(float) - mean) / scale)
    distances = tree.query(z, k=1, workers=1)[0]
    mask = ((x >= support['minimum']) & (x <= support['maximum'])).all(1)
    mask &= distances <= support['nn_distance_threshold'] + 1e-12
    if not mask.all():
        raise ValueError('prediction outside frozen full-source support')
    return distances


def historical_offsets(frame, ids):
    h = frame.loc[frame.year.eq(2024) & frame.facility_id.isin(ids)].copy()
    if h.empty or set(h.facility_id) != set(ids):
        raise ValueError('2024 historical offsets missing facility identity')
    h['residual'] = h.prediction - h.observed
    offsets = h.groupby(['model', 'facility_id'], as_index=False).agg(
        offset_2024=('residual', 'mean'), historical_days=('residual', 'size'))
    for model, g in offsets.groupby('model'):
        if set(g.facility_id) != set(ids):
            raise ValueError('2024 offsets incomplete by model')
    pooled = offsets.groupby('model').offset_2024.mean().to_dict()
    return offsets, pooled


def alternatives(frame, offsets, pooled, radii):
    unique(offsets, ['facility_id', 'model'])
    d = frame.merge(offsets, on=['facility_id', 'model'], validate='many_to_one')
    if len(d) != len(frame):
        raise ValueError('missing historical offset')
    parts = []
    for name in ('original', 'pooled', 'facility'):
        x = d.copy()
        x['alternative'] = name
        x['applied_offset'] = 0. if name == 'original' else x.model.map(pooled) if name == 'pooled' else x.offset_2024
        x['original_prediction'] = x.prediction
        x['original_residual_prediction_minus_observed'] = x.original_prediction - x.observed
        x['prediction'] = x.prediction - x.applied_offset
        x['radius'] = x.model.map(radii)
        x['lower'] = x.prediction - x.radius
        x['upper'] = x.prediction + x.radius
        x['residual'] = x.prediction - x.observed
        x['residual_prediction_minus_observed'] = x.residual
        x['absolute_error'] = x.residual.abs()
        x['covered'] = x.observed.ge(x.lower) & x.observed.le(x.upper)
        x['width'] = 2 * x.radius
        x['winkler_day'] = x.width + 20 * np.maximum(x.absolute_error - x.radius, 0.)
        parts.append(x)
    return pd.concat(parts, ignore_index=True)


def facility_metrics(days):
    group = [x for x in ('case', 'scope', 'model', 'facility_id', 'alternative') if x in days]
    return days.groupby(group, as_index=False, sort=True).agg(
        days=('residual', 'size'), mae=('absolute_error', 'mean'), bias=('residual', 'mean'),
        coverage=('covered', 'mean'), radius=('radius', 'mean'), width=('width', 'mean'),
        winkler=('winkler_day', 'mean'))


def metric_changes(metrics):
    keys = [x for x in ('case', 'scope', 'model', 'facility_id') if x in metrics]
    out = metrics.query("alternative == 'facility'").drop(columns='alternative').copy()
    for baseline in ('original', 'pooled'):
        ref = metrics.loc[metrics.alternative.eq(baseline), keys + METRICS + ['days']]
        j = out.merge(ref, on=keys, suffixes=('', '_reference'), validate='one_to_one')
        if not j.days.eq(j.days_reference).all():
            raise ValueError('different evaluation days across alternatives')
        for metric in METRICS:
            out[f"{metric}_facility_minus_{baseline}"] = (j[metric] - j[metric + '_reference']).to_numpy()
    out['mae_improved_vs_original'] = out.mae_facility_minus_original.lt(0)
    out['mae_improved_vs_pooled'] = out.mae_facility_minus_pooled.lt(0)
    return out


def cardinality_assignment(distances, threshold):
    d = np.asarray(distances, dtype=float)
    n, m = d.shape
    if not n or not m:
        return np.empty((0, 2), dtype=int)
    # Losing one edge costs C > the total distance of ANY feasible matching.
    # Thus every higher-cardinality solution beats every lower-cardinality one;
    # at fixed cardinality the objective differs only by summed real distance.
    penalty = (min(n, m) + 1) * threshold + 1.
    cost = np.full((n, m + n), penalty)
    cost[:, :m] = np.where(d <= threshold, d, np.inf)
    rows, cols = linear_sum_assignment(cost)
    keep = cols < m
    return np.column_stack([rows[keep], cols[keep]])


def facility_cluster_weights(operator_counts, facility_operators, operator_ids):
    index = {int(op): i for i, op in enumerate(operator_ids)}
    return np.asarray(operator_counts)[:, [index[int(op)] for op in facility_operators]]


def weighted_facility_stat(values, weights):
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    numerator = w @ v
    denominator = w.sum(1)
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)


def validate_cache_entry(path, entry, sid, model, sources):
    if (entry['sha256'] != sha(path) or entry['model'] != model or
            entry['subset'] != f"LOO_{sid}" or set(entry['included_sources']) != set(sources) - {sid}):
        raise ValueError(f"cache hash or identity mismatch: {sid} {model}")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Runner:
    def __init__(self, manifest_path, spec_path, output):
        self.manifest_path, self.spec_path = Path(manifest_path), Path(spec_path)
        self.manifest = json.loads(self.manifest_path.read_text())
        self.spec = json.loads(self.spec_path.read_text())
        self.out = Path(output).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        if (self.out / 'COMPLETE.json').exists():
            raise ValueError('completed output exists; choose a fresh output directory')
        self.stages = []
        self.started = time.time()
        check_manifest(self.manifest)
        self.frozen = json.loads(self.path('frozen_spec').read_text())
        self.radii = {m: self.frozen['models'][m]['source_residual_interval']['half_width_mmbtu_per_mwh'] for m in self.spec['models']}
        for key, value in {'offset_year': 2024, 'evaluation_year': 2025, 'primary_facilities': 120,
                           'permutations': 1000, 'bootstrap_replicates': 5000,
                           'bootstrap_seed': 2026100736, 'permutation_seed': 2026100737,
                           'conditional_distance': .095, 'conditional_min_pairs': 30,
                           'conditional_min_months': 3, 'model_refits': 0, 'interval_recalibration': False}.items():
            if self.spec[key] != value:
                raise ValueError(f"locked analysis specification mismatch: {key}")
        if self.spec['years'] != [2024, 2025] or self.spec['models'] != ['M1', 'M2']:
            raise ValueError('unknown analysis year or model')
        for var in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            if os.environ.get(var) != '1':
                raise ValueError(f"one-thread environment required: {var}")

    def path(self, role):
        return Path(self.manifest['files'][role]['path'])

    def read(self, role):
        return pd.read_csv(self.path(role))

    def csv(self, name, frame):
        save_csv(self.out / name, frame)

    @contextmanager
    def stage(self, name):
        start = time.time()
        def exceeded(*_):
            raise TimeoutError(f"stage exceeded locked time limit: {name}")
        old = signal.signal(signal.SIGALRM, exceeded)
        signal.alarm(self.spec['stage_seconds'])
        print(f"STAGE START {name}", flush=True)
        try:
            yield
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            peak_gib = peak / (1024 ** 3 if sys.platform == 'darwin' else 1024 ** 2)
            if peak_gib > self.spec['memory_target_gib']:
                raise MemoryError('peak memory exceeded registered target')
            self.stages.append({'stage': name, 'seconds': time.time() - start, 'peak_gib': peak_gib})
            save_json(self.out / 'CHECKPOINT.json', {'status': 'running', 'completed_stages': self.stages})
            print(f"STAGE DONE {name} {time.time()-start:.2f}s {peak_gib:.3f}GiB", flush=True)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)

    def validate_inputs(self):
        pred = validate_dates(self.read('predictions'), [2024, 2025], KEYS + ['model'])
        weather = validate_dates(self.read('weather'), [2024, 2025], KEYS)
        if set(pred.model) != {'M1', 'M2'}:
            raise ValueError('wrong prediction models')
        pred['observed'] = pred.gross_heat_rate_mmbtu_per_mwh
        features = ['ambient_temperature_c', 'facility_load_fraction', 'doy_sin', 'doy_cos', 'month']
        d = pred.merge(weather[KEYS + features + ['operator_id', 'gross_heat_rate_mmbtu_per_mwh']],
                       on=KEYS, validate='many_to_one', suffixes=('', '_weather'))
        if len(d) != len(pred) or not d.operator_id.eq(d.operator_id_weather).all():
            raise ValueError('weather join identity mismatch')
        if not np.allclose(d.observed, d.gross_heat_rate_mmbtu_per_mwh_weather, atol=1e-12, rtol=0):
            raise ValueError('weather observation mismatch')
        if not np.allclose(d.prediction - d.observed, d.residual_prediction_minus_observed, atol=1e-12, rtol=0):
            raise ValueError('residual sign/value mismatch')
        for m in self.spec['models']:
            if not np.allclose(d.loc[d.model.eq(m), 'source_radius_mmbtu_per_mwh'], self.radii[m], atol=1e-12, rtol=0):
                raise ValueError('wrong original radius')
        x = d.loc[d.model.eq('M1'), KEYS].sort_values(KEYS).reset_index(drop=True)
        y = d.loc[d.model.eq('M2'), KEYS].sort_values(KEYS).reset_index(drop=True)
        pd.testing.assert_frame_equal(x, y)
        self.source = self.read('source_daily')
        d['support_nn_distance'] = require_supported(self.source, d, self.frozen['input_support'])
        self.replay = load_module(self.path('vendor_replay'), 'heat_rate_frozen_replay')
        self.replay.SPEC = self.frozen
        self.baseline = load_module(self.path('vendor_baseline'), 'heat_rate_frozen_baseline')
        self.full_models = {m: self.replay.SerializedRelation(m) for m in self.spec['models']}
        for year in (2024, 2025):
            evaluated = set(d.loc[d.year.eq(year), 'facility_id'])
            w = weather.loc[weather.year.eq(year) & weather.facility_id.isin(evaluated)]
            mask, _ = self.replay.fixed_support(self.source, w)
            expected = set(zip(w.loc[mask].facility_id, w.loc[mask].date))
            actual = set(zip(d.loc[d.year.eq(year)].facility_id, d.loc[d.year.eq(year)].date))
            if expected != actual:
                raise ValueError('full-source support membership changed')
        checks = []
        for m in self.spec['models']:
            a = d.loc[d.model.eq(m)]
            error = float(np.max(np.abs(self.full_models[m].predict(a) - a.prediction)))
            if error > self.spec['prediction_tolerance']:
                raise ValueError('frozen full-source prediction mismatch')
            checks.append({'model': m, 'days': len(a), 'max_prediction_error': error})
        paired = self.read('paired')
        unique(paired, ['facility_id', 'model'])
        self.ids = sorted(paired.facility_id.unique().astype(int))
        annual_ids = {y: set(d.loc[d.year.eq(y), 'facility_id']) for y in [2024, 2025]}
        if set(self.ids) != annual_ids[2024] & annual_ids[2025] or len(self.ids) != 120:
            raise ValueError('primary paired120 identity mismatch')
        legacy = set(self.read('legacy_panel').facility_id)
        if len(annual_ids[2025]) != 121 or len(legacy) != 120 or len(legacy & set(self.ids)) != 106:
            raise ValueError('annual/legacy/intersection identity mismatch')
        self.identity = paired.query("model == 'M1'")[['facility_id', 'operator_id_2024', 'operator_id_2025',
             'facility_name_2024', 'operator_name_2024', 'operator_name_2025']].sort_values('facility_id')
        self.identity['operator_changed'] = self.identity.operator_id_2024.ne(self.identity.operator_id_2025)
        self.identity['in_legacy_panel'] = self.identity.facility_id.isin(legacy)
        self.csv('FACILITY_IDENTITY_AND_OPERATOR_CHANGES.csv', self.identity)
        self.csv('OPERATOR_CHANGES_ONLY.csv', self.identity.loc[self.identity.operator_changed])
        self.d = d.loc[d.facility_id.isin(self.ids)].sort_values(['model', 'facility_id', 'year', 'date']).reset_index(drop=True)
        s = self.frozen['input_support']
        self.d['z_temperature'] = (self.d.ambient_temperature_c - s['mean'][0]) / s['scale'][0]
        self.d['z_load'] = (self.d.facility_load_fraction - s['mean'][1]) / s['scale'][1]
        self.d['day_id'] = self.d.facility_id.astype(str) + '|' + self.d.date.dt.strftime('%Y-%m-%d')
        self.csv('PRIMARY_FULL_SUPPORT_DAYS.csv.gz', self.d)
        self.csv('INPUT_REPRODUCTION_CHECKS.csv', pd.DataFrame(checks))
        save_json(self.out / 'PANEL_IDENTITY.json', {'annual_2024': len(annual_ids[2024]), 'annual_2025': 121,
            'paired': 120, 'legacy': 120, 'intersection': 106, 'operators_2024': int(self.identity.operator_id_2024.nunique()),
            'full_source_support_unchanged': True, 'main_days_by_year_model': {f"{y}_{m}": int(n) for (y, m), n in self.d.groupby(['year', 'model']).size().items()}})

    def cache_compatibility(self):
        import joblib
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import SplineTransformer, StandardScaler
        def forbidden_fit(*_, **__):
            raise RuntimeError('model training forbidden in bounded analysis')
        for cls in (Pipeline, Ridge, SplineTransformer, StandardScaler):
            cls.fit = forbidden_fit
        entries = {x['file']: x for x in json.loads(self.path('cache_manifest').read_text())}
        self.caches = {}
        cache_checks = []
        cache_parameters = []
        for sid in self.spec['source_ids']:
            for model in self.spec['models']:
                p = self.path(f"cache_{sid}_{model}")
                entry = entries[p.name]
                validate_cache_entry(p, entry, sid, model, self.spec['source_ids'])
                obj = joblib.load(p)
                if not isinstance(obj, Pipeline):
                    raise ValueError('unknown cache model type')
                self.caches[(sid, model)] = obj
                a = self.d.loc[self.d.model.eq(model)]
                direct = obj.predict(a)
                vendor = self.baseline.predict_model(model, obj, a)
                error = float(np.max(np.abs(direct - vendor)))
                if error > 1e-12 or not np.isfinite(direct).all():
                    raise ValueError('cache incompatible with vendor baseline predictor')
                cache_checks.append({'source_omitted': sid, 'model': model, 'sha256': sha(p),
                    'type': type(obj).__name__, 'vendor_prediction_max_error': error,
                    'included_sources': '|'.join(map(str, entry['included_sources'])), 'new_fits': 0})
                cache_parameters.append({'source_omitted': sid, 'model': model,
                    'pipeline_repr': repr(obj), 'steps': [{'name': name, 'class': type(step).__name__,
                    'learned_attributes': {key: np.asarray(getattr(step, key)).tolist() for key in
                    ('scale_', 'mean_', 'var_', 'n_samples_seen_', 'coef_', 'intercept_') if hasattr(step, key)}}
                    for name, step in obj.steps]})
        self.csv('CACHE_COMPATIBILITY.csv', pd.DataFrame(cache_checks))
        save_json(self.out / 'CACHE_LEARNED_PREPROCESSING.json', cache_parameters)
        historical = self.read('source_metrics').query("subset_type == 'leave_one_out'").copy()
        unique(historical, ['subset_id', 'year', 'model', 'facility_id'])
        if set(historical.subset_id) != {f"LOO_{sid}" for sid in self.spec['source_ids']} or set(historical.year) != {2022, 2023, 2024}:
            raise ValueError('missing historical single-source cases')
        self.csv('SOURCE_HISTORICAL_ORIGINAL_2022_2024.csv', historical)
        replayed = []
        difference_rows = []
        for year in (2022, 2023, 2024):
            raw_daily = self.read(f"legacy_daily_{year}")
            raw_pred = self.read(f"legacy_predictions_{year}")
            # Year is explicit in the manifest role for historical yearless CSVs.
            if 'year' not in raw_daily:
                raw_daily['year'] = year
            if 'year' not in raw_pred:
                raw_pred['year'] = year
            daily = validate_dates(raw_daily, [year], KEYS)
            pred = validate_dates(raw_pred, [year], KEYS + ['model'])
            cols = KEYS + ['ambient_temperature_c', 'facility_load_fraction', 'doy_sin', 'doy_cos']
            joined = pred.merge(daily[cols], on=KEYS, validate='many_to_one')
            if len(joined) != len(pred):
                raise ValueError('historical feature/support join missing input')
            for sid in self.spec['source_ids']:
                for model in self.spec['models']:
                    j = joined.loc[joined.model.eq(model)].copy()
                    j['observed'] = j.gross_heat_rate_mmbtu_per_mwh
                    p = self.baseline.predict_model(model, self.caches[(sid, model)], j)
                    r = p - j.observed.to_numpy()
                    j['absolute_error'], j['residual'] = np.abs(r), r
                    radius = self.radii[model]
                    j['radius'], j['width'] = radius, 2 * radius
                    j['covered'] = np.abs(r) <= radius
                    j['winkler_day'] = 2 * radius + 20 * np.maximum(np.abs(r) - radius, 0)
                    j['alternative'] = 'original'
                    mm = facility_metrics(j).assign(subset_id=f"LOO_{sid}", year=year)
                    replayed.append(mm)
                    ref = historical.loc[historical.subset_id.eq(f"LOO_{sid}") & historical.year.eq(year) & historical.model.eq(model)]
                    pair = mm.merge(ref, on=['subset_id', 'year', 'model', 'facility_id'], validate='one_to_one')
                    if len(pair) != len(ref) or len(mm) != len(ref):
                        raise ValueError('historical cache facility support identity mismatch')
                    for got, expected in [('days', 'matched_support_days'), ('mae', 'mae_mmbtu_per_mwh'),
                                          ('bias', 'signed_bias_mmbtu_per_mwh'), ('coverage', 'coverage_90'),
                                          ('winkler', 'mean_winkler_score')]:
                        delta = pair[got] - pair[expected]
                        maximum = float(delta.abs().max())
                        difference_rows.append({'subset_id': f"LOO_{sid}", 'year': year, 'model': model,
                            'metric': got, 'facilities': len(pair), 'max_absolute_difference': maximum})
                        if maximum > self.spec['prediction_tolerance']:
                            self.csv('CACHE_HISTORICAL_FAILURE.csv', pair)
                            raise ValueError(f"historical cache mismatch: {sid} {model} {year} {got} {maximum}")
        self.csv('SOURCE_HISTORICAL_CACHE_REPLAY.csv', pd.concat(replayed, ignore_index=True))
        self.csv('CACHE_HISTORICAL_METRIC_DIFFERENCES.csv', pd.DataFrame(difference_rows))
        hs = historical.groupby(['subset_id', 'year', 'model'], as_index=False).agg(
            facilities=('facility_id', 'nunique'), days=('matched_support_days', 'sum'),
            mae=('mae_mmbtu_per_mwh', 'mean'), bias=('signed_bias_mmbtu_per_mwh', 'mean'),
            coverage=('coverage_90', 'mean'), winkler=('mean_winkler_score', 'mean'))
        for metric in ('mae', 'bias', 'coverage', 'winkler'):
            hs[metric + '_rank'] = hs.groupby(['year', 'model'])[metric].rank(method='average')
        self.csv('SOURCE_HISTORICAL_SUMMARIES_RANKS.csv', hs)

    def conditional_match(self):
        d = self.d.query("model == 'M1'")
        pairs, monthly = [], []
        for fid in self.ids:
            f = d.loc[d.facility_id.eq(fid)]
            for month in range(1, 13):
                a = f.loc[f.year.eq(2024) & f.month.eq(month)].sort_values('date').reset_index(drop=True)
                b = f.loc[f.year.eq(2025) & f.month.eq(month)].sort_values('date').reset_index(drop=True)
                distance = cdist(a[['z_temperature', 'z_load']], b[['z_temperature', 'z_load']])
                matched = cardinality_assignment(distance, self.spec['conditional_distance'])
                monthly.append({'facility_id': fid, 'month': month, 'days_2024': len(a), 'days_2025': len(b),
                                'pairs': len(matched), 'candidate_edges': int((distance <= .095).sum())})
                for i, j in matched:
                    row = {'facility_id': fid, 'month': month, 'distance': distance[i, j],
                           'pair_id': f"{fid}|{a.date.iloc[i]:%Y-%m-%d}|{b.date.iloc[j]:%Y-%m-%d}"}
                    for y, source in [(2024, a.iloc[i]), (2025, b.iloc[j])]:
                        for col in ('date', 'day_id', 'operator_id', 'observed', 'ambient_temperature_c',
                                    'facility_load_fraction', 'z_temperature', 'z_load', 'support_nn_distance'):
                            row[f"{col}_{y}"] = source[col]
                    pairs.append(row)
        self.matches = pd.DataFrame(pairs)
        if self.matches.empty:
            raise ValueError('no conditional matches; explicit scientific stop required')
        for y in (2024, 2025):
            unique(self.matches, ['facility_id', f"date_{y}"])
        self.csv('CONDITIONAL_MATCH_COUNTS_MONTH.csv', pd.DataFrame(monthly))
        counts = pd.DataFrame(monthly).groupby('facility_id', as_index=False).agg(
            pairs=('pairs', 'sum'), months=('pairs', lambda x: int((x > 0).sum())),
            full_support_2024=('days_2024', 'sum'), full_support_2025=('days_2025', 'sum'))
        counts['eligible'] = counts.pairs.ge(30) & counts.months.ge(3)
        counts['exclusion_reason'] = counts.apply(lambda r: 'included' if r.eligible else
            '|'.join((['below_30_pairs'] if r.pairs < 30 else []) + (['below_3_months'] if r.months < 3 else [])), axis=1)
        self.conditional_ids = set(counts.loc[counts.eligible, 'facility_id'])
        self.matches['eligible'] = self.matches.facility_id.isin(self.conditional_ids)
        self.csv('CONDITIONAL_MATCH_COUNTS_FACILITY.csv', counts)
        self.csv('CONDITIONAL_DAY_PAIRS.csv.gz', self.matches)
        balance = []
        for fid, g in self.matches.groupby('facility_id'):
            for variable in ('z_temperature', 'z_load'):
                delta = g[variable + '_2025'] - g[variable + '_2024']
                balance.append({'facility_id': fid, 'variable': variable, 'eligible': fid in self.conditional_ids,
                    'pairs': len(g), 'mean_2024': g[variable + '_2024'].mean(), 'mean_2025': g[variable + '_2025'].mean(),
                    'mean_delta_in_frozen_sd_units': delta.mean(), 'mean_absolute_delta': delta.abs().mean(),
                    'max_absolute_delta': delta.abs().max(), 'mean_euclidean_distance': g.distance.mean()})
        self.csv('CONDITIONAL_BALANCE_FACILITY.csv', pd.DataFrame(balance))
        self.csv('CONDITIONAL_BALANCE_SUMMARY.csv', pd.DataFrame(balance).query('eligible').groupby('variable', as_index=False).agg(
            facilities=('facility_id', 'nunique'), mean_delta_in_frozen_sd_units=('mean_delta_in_frozen_sd_units', 'mean'),
            mean_absolute_delta=('mean_absolute_delta', 'mean'), max_absolute_delta=('max_absolute_delta', 'max')))

    def evaluate_cases(self):
        metrics, offsets_all, changes, summaries, transfer, residual_pairs, residual_facility = [], [], [], [], [], [], []
        source_year_metrics = []
        cases = [('FULL', None)] + [(f"LOO_{sid}", sid) for sid in self.spec['source_ids']]
        for name, sid in cases:
            d = self.d.copy()
            if sid is not None:
                for model in self.spec['models']:
                    mask = d.model.eq(model)
                    d.loc[mask, 'prediction'] = self.baseline.predict_model(model, self.caches[(sid, model)], d.loc[mask])
            d['residual_prediction_minus_observed'] = d.prediction - d.observed
            for year in (2024, 2025):
                original = d.loc[d.year.eq(year)].copy()
                original['case'], original['alternative'] = name, 'original'
                original['residual'] = original.prediction-original.observed
                original['radius'] = original.model.map(self.radii)
                original['width'] = 2*original.radius
                original['lower'], original['upper'] = original.prediction-original.radius, original.prediction+original.radius
                original['absolute_error'] = original.residual.abs()
                original['covered'] = original.absolute_error <= original.radius
                original['winkler_day'] = original.width+20*np.maximum(original.absolute_error-original.radius, 0)
                mm = facility_metrics(original).assign(year=year)
                mm['mae_rank_within_case_year_model'] = mm.groupby('model').mae.rank(method='average')
                mm['bias_rank_within_case_year_model'] = mm.groupby('model').bias.rank(method='average')
                source_year_metrics.append(mm)
                if year == 2024:
                    self.csv(f"{name}_2024_BRIDGE_PREDICTIONS.csv.gz", original)
            off, pool = historical_offsets(d, self.ids)
            off['case'] = name
            off['pooled_120_offset'] = off.model.map(pool)
            offsets_all.append(off)
            for scope in ('main', 'conditional'):
                target = d.loc[d.year.eq(2025)].copy()
                if scope == 'conditional':
                    selected = self.matches.loc[self.matches.eligible, ['facility_id', 'date_2025']].rename(columns={'date_2025': 'date'})
                    target = target.merge(selected, on=['facility_id', 'date'], validate='many_to_one')
                alt = alternatives(target, off.drop(columns=['case', 'pooled_120_offset']), pool, self.radii)
                alt['case'], alt['scope'] = name, scope
                self.csv(f"{name}_{scope.upper()}_2025_ALTERNATIVES.csv.gz", alt)
                mm = facility_metrics(alt)
                delta = metric_changes(mm)
                metrics.append(mm); changes.append(delta)
                summary = mm.groupby(['case', 'scope', 'model', 'alternative'], as_index=False)[METRICS].mean()
                counts = mm.groupby(['case', 'scope', 'model', 'alternative'], as_index=False).agg(facilities=('facility_id', 'nunique'), days=('days', 'sum'))
                summary = summary.merge(counts, validate='one_to_one')
                summaries.append(summary)
            for model in self.spec['models']:
                x = d.loc[d.model.eq(model)].copy()
                x['residual'] = x.prediction - x.observed
                annual = x.groupby(['facility_id', 'year']).residual.mean().unstack()
                annual['case'], annual['model'] = name, model
                annual['residual_delta_2025_minus_2024'] = annual[2025] - annual[2024]
                annual['same_sign'] = np.sign(annual[2025]).eq(np.sign(annual[2024]))
                annual['absolute_offset_transfer_error'] = (annual[2025] - annual[2024]).abs()
                annual = annual.rename(columns={2024: 'mean_residual_2024', 2025: 'mean_residual_2025'}).reset_index()
                annual['bias_rank_2024'] = annual.mean_residual_2024.rank(method='average')
                annual['bias_rank_2025'] = annual.mean_residual_2025.rank(method='average')
                transfer.append(annual)
                j = self.matches.copy()
                for year in (2024, 2025):
                    a = x.loc[x.year.eq(year), ['facility_id', 'date', 'prediction', 'observed', 'residual']]
                    a = a.rename(columns={col: f"{col}_{year}" for col in ('date', 'prediction', 'observed', 'residual')})
                    j = j.merge(a, on=['facility_id', f"date_{year}"], validate='one_to_one', suffixes=('', '_verify'))
                j['case'], j['model'] = name, model
                j['paired_residual_delta'] = j.residual_2025 - j.residual_2024
                residual_pairs.append(j)
                fm = j.groupby(['case', 'model', 'facility_id', 'eligible'], as_index=False).agg(
                    pairs=('pair_id', 'size'), conditional_residual_2024=('residual_2024', 'mean'),
                    conditional_residual_2025=('residual_2025', 'mean'), paired_residual_delta=('paired_residual_delta', 'mean'),
                    mean_absolute_paired_residual_delta=('paired_residual_delta', lambda v: v.abs().mean()))
                residual_facility.append(fm)
        self.metrics = pd.concat(metrics, ignore_index=True)
        self.changes = pd.concat(changes, ignore_index=True)
        self.offsets = pd.concat(offsets_all, ignore_index=True)
        self.summaries = pd.concat(summaries, ignore_index=True)
        self.transfer = pd.concat(transfer, ignore_index=True)
        self.residual_facility = pd.concat(residual_facility, ignore_index=True)
        self.csv('FACILITY_ALL_METRICS.csv', self.metrics)
        self.csv('FACILITY_ALL_CHANGES.csv', self.changes)
        self.csv('FACILITY_FULL_2024_OFFSETS.csv', self.offsets)
        self.csv('FACILITY_EQUAL_SUMMARIES.csv', self.summaries)
        self.csv('OFFSET_TRANSFER_DIRECTIONS_RANKS.csv', self.transfer)
        transfer_summary = self.transfer.groupby(['case', 'model'], as_index=False).agg(
            facilities=('facility_id', 'nunique'), same_sign_facilities=('same_sign', 'sum'),
            mean_residual_2024=('mean_residual_2024', 'mean'), mean_residual_2025=('mean_residual_2025', 'mean'),
            mean_residual_drift=('residual_delta_2025_minus_2024', 'mean'),
            mean_absolute_offset_transfer_error=('absolute_offset_transfer_error', 'mean'))
        self.csv('SOURCE_OFFSET_TRANSFER_SUMMARY.csv', transfer_summary)
        self.csv('CONDITIONAL_RESIDUAL_DAY_PAIRS_ALL_CASES.csv.gz', pd.concat(residual_pairs, ignore_index=True))
        self.csv('CONDITIONAL_RESIDUAL_FACILITY_ALL_CASES.csv', self.residual_facility)
        source_summary = self.changes.groupby(['case', 'scope', 'model'], as_index=False).agg(
            facilities=('facility_id', 'nunique'), mae_facility_minus_original=('mae_facility_minus_original', 'mean'),
            mae_facility_minus_pooled=('mae_facility_minus_pooled', 'mean'),
            improved_vs_original=('mae_improved_vs_original', 'sum'), improved_vs_pooled=('mae_improved_vs_pooled', 'sum'))
        source_summary['rank_gain_vs_original'] = source_summary.groupby(['scope', 'model']).mae_facility_minus_original.rank(method='average')
        source_summary['rank_gain_vs_pooled'] = source_summary.groupby(['scope', 'model']).mae_facility_minus_pooled.rank(method='average')
        full = source_summary.query("case == 'FULL'")[['scope', 'model', 'mae_facility_minus_original', 'mae_facility_minus_pooled']]
        source_summary = source_summary.merge(full, on=['scope', 'model'], suffixes=('', '_full'), validate='many_to_one')
        for base in ('original', 'pooled'):
            col = f"mae_facility_minus_{base}"
            source_summary[f"same_direction_vs_full_{base}"] = np.sign(source_summary[col]).eq(np.sign(source_summary[col + '_full']))
        self.csv('SOURCE_CASE_SUMMARIES_DIRECTIONS_RANKS.csv', source_summary)
        year_metrics = pd.concat(source_year_metrics, ignore_index=True)
        self.csv('SOURCE_ORIGINAL_2024_2025_FACILITY_METRICS.csv', year_metrics)
        ref = year_metrics.query("case == 'FULL'").drop(columns=['case', 'alternative'])
        yc = year_metrics.merge(ref, on=['year', 'model', 'facility_id'], suffixes=('', '_full'), validate='many_to_one')
        if not yc.days.eq(yc.days_full).all():
            raise ValueError('source case day support changed')
        for metric in METRICS:
            yc[metric+'_minus_full'] = yc[metric]-yc[metric+'_full']
            yc[metric+'_direction_minus_full'] = np.sign(yc[metric+'_minus_full'])
        yc['mae_rank_change_vs_full'] = yc.mae_rank_within_case_year_model-yc.mae_rank_within_case_year_model_full
        yc['bias_rank_change_vs_full'] = yc.bias_rank_within_case_year_model-yc.bias_rank_within_case_year_model_full
        self.csv('SOURCE_ORIGINAL_YEAR_COMPARISONS.csv', yc)
        ys = year_metrics.groupby(['case', 'year', 'model'], as_index=False)[METRICS].mean()
        for metric in METRICS:
            ys[metric+'_rank_across_cases'] = ys.groupby(['year', 'model'])[metric].rank(method='average')
        self.csv('SOURCE_ORIGINAL_2024_2025_EQUAL_FACILITY_SUMMARY.csv', ys)

    def permutations(self):
        rng = np.random.default_rng(self.spec['permutation_seed'])
        permutation = np.array([rng.permutation(len(self.ids)) for _ in range(1000)])
        donor_ids = np.array(self.ids)[permutation]
        mapping = pd.DataFrame({'replicate': np.repeat(np.arange(1, 1001), len(self.ids)),
            'target_facility_id': np.tile(self.ids, 1000), 'offset_donor_facility_id': donor_ids.ravel()})
        self.csv('PERMUTATION_FACILITY_LABELS_SHARED_M1_M2.csv.gz', mapping)
        summaries, controls = [], []
        for scope in ('main', 'conditional'):
            for model in self.spec['models']:
                d = self.d.loc[self.d.year.eq(2025) & self.d.model.eq(model)].copy()
                if scope == 'conditional':
                    selected = self.matches.loc[self.matches.eligible, ['facility_id', 'date_2025']].rename(columns={'date_2025': 'date'})
                    d = d.merge(selected, on=['facility_id', 'date'], validate='one_to_one')
                offsets = self.offsets.loc[self.offsets.case.eq('FULL') & self.offsets.model.eq(model)].set_index('facility_id').loc[self.ids].offset_2024.to_numpy()
                fids = sorted(d.facility_id.unique())
                lookup = {fid: i for i, fid in enumerate(self.ids)}
                cubes = {k: [] for k in METRICS}
                day_counts = []
                for fid in fids:
                    f = d.loc[d.facility_id.eq(fid)]
                    residual = (f.prediction - f.observed).to_numpy()[:, None] - offsets[None, :]
                    ae = np.abs(residual); rad = self.radii[model]
                    day_counts.append(len(f))
                    for metric, values in {'mae': ae.mean(0), 'bias': residual.mean(0), 'coverage': (ae <= rad).mean(0),
                        'radius': np.full(len(self.ids), rad), 'width': np.full(len(self.ids), 2 * rad),
                        'winkler': (2 * rad + 20 * np.maximum(ae - rad, 0)).mean(0)}.items():
                        cubes[metric].append(values)
                target_index = [lookup[f] for f in fids]
                assigned = permutation[:, target_index]
                data = {'replicate': np.repeat(np.arange(1, 1001), len(fids)), 'facility_id': np.tile(fids, 1000),
                        'donor_facility_id': np.array(self.ids)[assigned].ravel(), 'days': np.tile(day_counts, 1000)}
                summary = pd.DataFrame({'replicate': np.arange(1, 1001), 'scope': scope, 'model': model, 'facilities': len(fids)})
                for metric in METRICS:
                    matrix = np.array(cubes[metric])
                    vals = matrix[np.arange(len(fids))[None, :], assigned]
                    data[metric] = vals.ravel()
                    summary[metric] = vals.mean(1)
                pf = pd.DataFrame(data).assign(scope=scope, model=model)
                self.csv(f"PERMUTATION_FACILITY_METRICS_{scope.upper()}_{model}.csv.gz", pf)
                actual = self.summaries.query("case == 'FULL' and scope == @scope and model == @model").set_index('alternative')
                for baseline in ('original', 'pooled'):
                    summary[f"mae_minus_{baseline}"] = summary.mae - actual.loc[baseline, 'mae']
                summary['mae_minus_correct_facility'] = summary.mae - actual.loc['facility', 'mae']
                summaries.append(summary)
                controls.append({'scope': scope, 'model': model, 'facilities': len(fids), 'permutations': 1000,
                    'correct_facility_mae': actual.loc['facility', 'mae'], 'permuted_mean_mae': summary.mae.mean(),
                    'permuted_mae_q025': summary.mae.quantile(.025), 'permuted_mae_median': summary.mae.median(),
                    'permuted_mae_q975': summary.mae.quantile(.975), 'permuted_mae_min': summary.mae.min(),
                    'permuted_mae_max': summary.mae.max(), 'descriptive_control_not_p_value': True})
        self.csv('PERMUTATION_1000_SUMMARIES.csv', pd.concat(summaries, ignore_index=True))
        self.csv('PERMUTATION_DESCRIPTIVE_CONTROL.csv', pd.DataFrame(controls))

    def bootstrap(self):
        ids = self.identity.sort_values('facility_id')
        operator_ids = np.sort(ids.operator_id_2024.unique().astype(int))
        rng = np.random.default_rng(self.spec['bootstrap_seed'])
        counts = np.array([np.bincount(rng.integers(0, len(operator_ids), len(operator_ids)), minlength=len(operator_ids)) for _ in range(5000)])
        np.savez_compressed(self.out / 'BOOTSTRAP_OPERATOR_WEIGHTS.npz', replicate=np.arange(1, 5001), operator_ids=operator_ids,
            operator_counts=counts, facility_ids=ids.facility_id.to_numpy(), facility_operator_ids=ids.operator_id_2024.to_numpy())
        self.csv('BOOTSTRAP_OPERATOR_MULTIPLICITIES.csv.gz', pd.DataFrame({'replicate': np.repeat(np.arange(1, 5001), len(operator_ids)),
            'operator_id_2024': np.tile(operator_ids, 5000), 'multiplicity': counts.ravel()}))
        replicates, cis = [], []
        for (case, scope, model), g in self.changes.groupby(['case', 'scope', 'model']):
            g = g.sort_values('facility_id').merge(ids[['facility_id', 'operator_id_2024']], on='facility_id', validate='one_to_one').sort_values('facility_id')
            weights = facility_cluster_weights(counts, g.operator_id_2024, operator_ids)
            rep = pd.DataFrame({'replicate': np.arange(1, 5001), 'case': case, 'scope': scope, 'model': model,
                'facility_weight_sum': weights.sum(1), 'distinct_selected_facilities': (weights > 0).sum(1)})
            for baseline in ('original', 'pooled'):
                for metric in METRICS:
                    col = f"{metric}_facility_minus_{baseline}"
                    vals = weighted_facility_stat(g[col].to_numpy(), weights)
                    rep[col] = vals
                    valid = vals[np.isfinite(vals)]
                    cis.append({'case': case, 'scope': scope, 'model': model, 'quantity': col,
                        'estimate': g[col].mean(), 'ci_low': np.quantile(valid, .025), 'ci_high': np.quantile(valid, .975),
                        'facilities': len(g), 'operator_clusters_registered': len(operator_ids), 'valid_replicates': len(valid),
                        'invalid_zero_facility_weight': 5000 - len(valid), 'estimand': 'facility_equal_delta_operator_cluster_resampling'})
            replicates.append(rep)
        self.ci = pd.DataFrame(cis)
        self.csv('BOOTSTRAP_5000_ALL_CASES_REPLICATES.csv.gz', pd.concat(replicates, ignore_index=True))
        self.csv('BOOTSTRAP_CI_ALL_CASES.csv', self.ci)
        residual_reps, residual_cis = [], []
        for (case, model), g in self.residual_facility.query('eligible').groupby(['case', 'model']):
            g = g.merge(ids[['facility_id', 'operator_id_2024']], on='facility_id', validate='one_to_one').sort_values('facility_id')
            weights = facility_cluster_weights(counts, g.operator_id_2024, operator_ids)
            rep = pd.DataFrame({'replicate': np.arange(1, 5001), 'case': case, 'model': model})
            for col in ('conditional_residual_2024', 'conditional_residual_2025', 'paired_residual_delta', 'mean_absolute_paired_residual_delta'):
                vals = weighted_facility_stat(g[col].to_numpy(), weights)
                rep[col] = vals
                residual_cis.append({'case': case, 'model': model, 'quantity': col, 'facilities': len(g),
                    'estimate': g[col].mean(), 'ci_low': np.nanquantile(vals, .025), 'ci_high': np.nanquantile(vals, .975), 'replicates': 5000})
            residual_reps.append(rep)
        self.csv('CONDITIONAL_RESIDUAL_BOOTSTRAP_5000.csv.gz', pd.concat(residual_reps, ignore_index=True))
        self.csv('CONDITIONAL_RESIDUAL_CI.csv', pd.DataFrame(residual_cis))

    def finalize(self):
        import joblib
        import scipy
        import sklearn
        from threadpoolctl import threadpool_info
        check_manifest(self.manifest)
        results = self.ci.query("case == 'FULL' and quantity.str.startswith('mae_')", engine='python')
        counts = pd.read_csv(self.out / 'CONDITIONAL_MATCH_COUNTS_FACILITY.csv')
        summary = {
            'status': 'complete', 'post_hoc': True, 'new_model_fits': 0, 'interval_recalibration': False,
            'conditional_eligible_facilities': int(counts.eligible.sum()), 'conditional_excluded_facilities': int((~counts.eligible).sum()),
            'conditional_pairs_all': int(counts.pairs.sum()), 'conditional_pairs_eligible': int(counts.loc[counts.eligible, 'pairs'].sum()),
            'primary_and_conditional_mae_ci': results.to_dict('records'),
            'source_omissions': self.spec['source_ids'], 'permutations': 1000, 'bootstrap_replicates': 5000,
        }
        save_json(self.out / 'SCIENTIFIC_RESULTS.json', summary)
        source_overview = pd.read_csv(self.out / 'SOURCE_CASE_SUMMARIES_DIRECTIONS_RANKS.csv')
        permutation_overview = pd.read_csv(self.out / 'PERMUTATION_DESCRIPTIVE_CONTROL.csv')
        overview = {'status': 'complete', 'post_hoc': True, 'panel': {'annual_2025': 121, 'paired': 120, 'legacy': 120,
            'intersection': 106, 'operator_clusters_2024': int(self.identity.operator_id_2024.nunique())},
            'conditional': {'eligible_facilities': summary['conditional_eligible_facilities'], 'excluded_facilities': summary['conditional_excluded_facilities'],
                'all_pairs': summary['conditional_pairs_all'], 'eligible_pairs': summary['conditional_pairs_eligible']},
            'mae_comparisons': results[['scope', 'model', 'quantity', 'estimate', 'ci_low', 'ci_high', 'facilities']].to_dict('records'),
            'source_cases': source_overview[['case', 'scope', 'model', 'mae_facility_minus_original', 'mae_facility_minus_pooled', 'same_direction_vs_full_original', 'same_direction_vs_full_pooled']].to_dict('records'),
            'descriptive_permutation_controls': permutation_overview.to_dict('records'),
            'cache_models_verified': 12, 'source_omissions_complete': 6, 'historical_years_replayed': [2022, 2023, 2024],
            'adverse_facility_counts': self.changes.query("case == 'FULL'").groupby(['scope', 'model']).apply(
                lambda g: {'scope': g.name[0], 'model': g.name[1], 'facilities': len(g),
                    'worse_vs_original': int((g.mae_facility_minus_original>0).sum()),
                    'worse_vs_pooled': int((g.mae_facility_minus_pooled>0).sum())}, include_groups=False).tolist(),
            'coverage_after_facility_correction': self.summaries.query("case == 'FULL' and alternative == 'facility'")[['scope', 'model', 'coverage', 'width']].to_dict('records'),
            'conditional_residual_comparisons': pd.read_csv(self.out / 'CONDITIONAL_RESIDUAL_CI.csv').query("case == 'FULL'").to_dict('records'),
            'bootstrap_replicates': 5000, 'permutations': 1000, 'new_fits': 0, 'interval_recalibration': False,
            'skipped_registered_branches': [],
            'runtime_seconds': time.time()-self.started, 'peak_gib': max(x['peak_gib'] for x in self.stages)}
        save_json(self.out / 'RESULTS_OVERVIEW.json', overview)
        software = {'python': sys.version, 'platform': platform.platform(), 'numpy': np.__version__, 'pandas': pd.__version__,
            'scipy': scipy.__version__, 'scikit_learn': sklearn.__version__, 'joblib': joblib.__version__}
        provenance = {'utc': datetime.now(timezone.utc).isoformat(),
            'input_manifest_sha256': sha(self.manifest_path), 'analysis_spec_sha256': sha(self.spec_path),
            'inputs': {**{key: self.manifest[key] for key in ('schema', 'scope') if key in self.manifest},
                       'files': {role: {key: value for key, value in item.items() if key != 'path'}
                                 for role, item in self.manifest['files'].items()},
                       'archives': [{key: value for key, value in item.items() if key != 'path'}
                                    for item in self.manifest.get('archives', [])]},
            'software': software, 'stages': self.stages, 'runtime_seconds': time.time()-self.started,
            'threadpools': [{key: value for key, value in item.items() if key != 'filepath'} for item in threadpool_info()],
            'environment': {v: os.environ.get(v) for v in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS']},
            'source_code_sha256': sha(Path(__file__)), 'test_code_sha256': sha(Path(__file__).with_name('test_historical_transfer.py')),
            'implementation_hashes': {name: sha(Path(__file__).with_name(name)) for name in ['historical_transfer_analysis.py', 'test_historical_transfer.py', 'package_tools.py', 'check_prepared.py']},
            'new_fits': 0, 'skipped_registered_branches': [],
            'historical_reference': 'corrected_measurement_GSOD_reference_distinct_from_corrected_weather_bridge',
            'historical_year_assignment': 'explicit_manifest_year_role_checked_against_calendar_dates'}
        save_json(self.out / 'PROVENANCE.json', provenance)
        save_json(self.out / 'CHECKPOINT.json', {'status': 'complete', 'completed_stages': self.stages})
        outputs = [{'path': str(p.relative_to(self.out)), 'sha256': sha(p), 'bytes': p.stat().st_size} for p in sorted(self.out.iterdir()) if p.is_file() and p.name not in ('OUTPUT_HASHES.json', 'COMPLETE.json')]
        save_json(self.out / 'OUTPUT_HASHES.json', outputs)
        save_json(self.out / 'COMPLETE.json', {'status': 'complete', 'registered_stages_finished': True,
            'outputs': len(outputs), 'output_hash_manifest_sha256': sha(self.out / 'OUTPUT_HASHES.json'), 'runtime_seconds': time.time()-self.started})
        print(json.dumps(summary, indent=2), flush=True)

    def run(self):
        with self.stage('input_identity_and_frozen_prediction'):
            self.validate_inputs()
        with self.stage('all_12_cache_compatibility_and_historical_replay'):
            self.cache_compatibility()
        with self.stage('max_cardinality_min_distance_conditional_matching'):
            self.conditional_match()
        with self.stage('all_source_cases_offset_and_interval_evaluation'):
            self.evaluate_cases()
        with self.stage('1000_shared_facility_label_permutations'):
            self.permutations()
        with self.stage('5000_operator_cluster_bootstrap_facility_equal'):
            self.bootstrap()
        with self.stage('178mm_figure'):
            from plot_upgrade import make_figure
            make_figure(self.out)
        self.finalize()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input-manifest', required=True)
    p.add_argument('--analysis-spec', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    runner = None
    try:
        runner = Runner(a.input_manifest, a.analysis_spec, a.output)
        runner.run()
    except BaseException as e:
        if runner is not None:
            save_json(runner.out / 'STOPPED.json', {'status': 'stopped', 'reason': f"{type(e).__name__}: {e}", 'completed_stages': runner.stages,
                'runtime_seconds': time.time()-runner.started})
        raise


if __name__ == '__main__':
    main()

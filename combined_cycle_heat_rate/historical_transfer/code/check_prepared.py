'Read-only checks of frozen predictions and registered derived tables.'
import argparse
import json
from pathlib import Path
import resource
import signal
import socket
import sys
import time

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching

import historical_transfer_analysis as c
from package_tools import prepare, privacy_scan, verify_payload, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--package-root', type=Path, required=True)
    p.add_argument('--input-manifest', type=Path, required=True)
    p.add_argument('--analysis-spec', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root = args.package_root.resolve()
    if args.output.resolve().is_relative_to(root):
        raise ValueError('check report must be outside immutable package')
    def blocked(*_args, **_kwargs):
        raise RuntimeError('network/model fitting disabled during prepared check')
    socket.socket.connect = blocked
    socket.create_connection = blocked
    socket.getaddrinfo = blocked
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('3600 second check limit')))
    signal.alarm(3600)
    import joblib
    import scipy
    import sklearn
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import SplineTransformer, StandardScaler
    for cls in (Pipeline, ColumnTransformer, Ridge, SplineTransformer, StandardScaler):
        cls.fit = blocked
        if hasattr(cls, 'fit_transform'):
            cls.fit_transform = blocked
    from threadpoolctl import threadpool_info
    start = time.monotonic()
    checks = []
    def check(value, label):
        if not value:
            raise ValueError('prepared check failure: ' + label)
        checks.append(label)
    def equal(a, b, label, tol=1e-10):
        check(np.shape(a) == np.shape(b) and np.allclose(a, b, atol=tol, rtol=0, equal_nan=True), label)
    payload_files = verify_payload(root)
    check(privacy_scan(root) == payload_files + 1, 'privacy_scan_all_payloads')
    manifest = json.loads(args.input_manifest.read_text())
    c.check_manifest(manifest)
    internal = prepare(root)
    check(manifest['files'] == internal['files'], 'only_extracted_internal_registered_inputs')
    spec = json.loads(args.analysis_spec.read_text())
    check(c.sha(args.analysis_spec) == c.sha(root / 'analysis_spec.json'), 'locked_analysis_spec')
    for env in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        import os
        check(os.environ.get(env) == '1', 'single_thread_env:' + env)
    check(all(item['num_threads'] == 1 for item in threadpool_info()), 'single_thread_loaded_blas')
    def inp(role):
        return Path(manifest['files'][role]['path'])
    out = root / 'results'
    def read(name):
        return pd.read_csv(out / name)
    frozen = json.loads(inp('frozen_spec').read_text())
    pred = c.validate_dates(pd.read_csv(inp('predictions')), [2024, 2025], c.KEYS + ['model'])
    weather = c.validate_dates(pd.read_csv(inp('weather')), [2024, 2025], c.KEYS)
    features = ['ambient_temperature_c', 'facility_load_fraction', 'doy_sin', 'doy_cos', 'month']
    joined = pred.merge(weather[c.KEYS + features], on=c.KEYS, validate='many_to_one')
    check(len(joined) == len(pred), 'registered_weather_join_membership')
    baseline = c.load_module(inp('vendor_baseline'), 'prepared_vendor_baseline')
    replay = c.load_module(inp('vendor_replay'), 'prepared_vendor_replay')
    replay.SPEC = frozen
    source = pd.read_csv(inp('source_daily'))
    c.require_supported(source, joined, frozen['input_support'])
    d = read('PRIMARY_FULL_SUPPORT_DAYS.csv.gz')
    d = c.validate_dates(d, [2024, 2025], c.KEYS + ['model'])
    ids = sorted(d.facility_id.unique())
    check(len(ids) == 120, 'primary120_identity')
    registered = joined.loc[joined.facility_id.isin(ids)].sort_values(['model'] + c.KEYS)
    stored = d.sort_values(['model'] + c.KEYS)
    check(list(zip(registered.model, registered.facility_id, registered.year, registered.date)) ==
          list(zip(stored.model, stored.facility_id, stored.year, stored.date)), 'exact_registered_primary_day_identity')
    equal(registered.prediction, stored.prediction, 'registered_predictions_unchanged')
    equal(registered.gross_heat_rate_mmbtu_per_mwh, stored.observed, 'registered_observations_unchanged')
    for i, col in enumerate(['z_temperature', 'z_load']):
        equal(d[col], (d[features[i]] - frozen['input_support']['mean'][i]) / frozen['input_support']['scale'][i], 'frozen_population_coordinates:' + col)
    radii = {m: frozen['models'][m]['source_residual_interval']['half_width_mmbtu_per_mwh'] for m in spec['models']}
    for model in spec['models']:
        f = joined.loc[joined.model.eq(model)]
        equal(replay.SerializedRelation(model).predict(f), f.prediction, 'frozen_full_model:' + model)
    entries = {e['file']: e for e in json.loads(inp('cache_manifest').read_text())}
    caches, cache_checks = {}, []
    for sid in spec['source_ids']:
        for model in spec['models']:
            path = inp(f"cache_{sid}_{model}")
            c.validate_cache_entry(path, entries[path.name], sid, model, spec['source_ids'])
            obj = joblib.load(path)
            check(isinstance(obj, Pipeline), f"frozen_pipeline:{sid}:{model}")
            a = d.loc[d.model.eq(model)]
            direct = obj.predict(a)
            equal(direct, baseline.predict_model(model, obj, a), f"cache_vendor_compatibility:{sid}:{model}", 1e-12)
            caches[(sid, model)] = obj
            cache_checks.append({'source_omitted': sid, 'model': model, 'sha256': c.sha(path), 'new_fits': 0})
    offsets = read('FACILITY_FULL_2024_OFFSETS.csv')
    metrics = read('FACILITY_ALL_METRICS.csv')
    recomputed = []
    cases = ['FULL'] + [f"LOO_{sid}" for sid in spec['source_ids']]
    for case in cases:
        ds = d.copy()
        for model in spec['models']:
            mask = ds.model.eq(model)
            if case != 'FULL':
                ds.loc[mask, 'prediction'] = caches[(int(case[4:]), model)].predict(ds.loc[mask])
        for year in (2024, 2025):
            ref = read(f"{case}_2024_BRIDGE_PREDICTIONS.csv.gz") if year == 2024 else read(f"{case}_MAIN_2025_ALTERNATIVES.csv.gz").query("alternative == 'original'")
            ref = ref.sort_values(['model'] + c.KEYS)
            actual = ds.loc[ds.year.eq(year)].sort_values(['model'] + c.KEYS)
            check(list(zip(ref.model, ref.facility_id, ref.date)) == list(zip(actual.model, actual.facility_id, actual.date.dt.strftime('%Y-%m-%d'))), f"case_day_identity:{case}:{year}")
            equal(actual.prediction, ref.prediction, f"case_frozen_prediction:{case}:{year}")
        off, pool = c.historical_offsets(ds, ids)
        saved_off = offsets.loc[offsets.case.eq(case)].sort_values(['model', 'facility_id'])
        equal(off.sort_values(['model', 'facility_id']).offset_2024, saved_off.offset_2024, 'full2024_offset:' + case)
        equal(saved_off.pooled_120_offset, saved_off.model.map(pool), 'equal120_pooled:' + case)
        for scope in ('main', 'conditional'):
            alt = read(f"{case}_{scope.upper()}_2025_ALTERNATIVES.csv.gz")
            expected_offset = alt.merge(off, on=['model', 'facility_id'], validate='many_to_one', suffixes=('', '_recomputed')).offset_2024_recomputed
            applied = np.where(alt.alternative.eq('facility'), expected_offset, np.where(alt.alternative.eq('pooled'), alt.model.map(pool), 0))
            equal(alt.applied_offset, applied, f"full_historical_offsets_no_rematching:{case}:{scope}")
            equal(alt.prediction, alt.original_prediction - applied, f"center_shift:{case}:{scope}")
            equal(alt.lower, alt.original_prediction - alt.radius - applied, f"lower_shift:{case}:{scope}")
            equal(alt.upper, alt.original_prediction + alt.radius - applied, f"upper_shift:{case}:{scope}")
            equal(alt.radius, alt.model.map(radii), f"original_radius:{case}:{scope}")
            equal(alt.width, 2 * alt.radius, f"fixed_width:{case}:{scope}")
            residual = alt.prediction - alt.observed
            equal(alt.residual, residual, f"residual_sign:{case}:{scope}")
            equal(alt.absolute_error, abs(residual), f"absolute_error:{case}:{scope}")
            equal(alt.winkler_day, 2 * alt.radius + 20 * np.maximum(abs(residual) - alt.radius, 0), f"winkler:{case}:{scope}")
            equal(alt.covered.astype(float), ((alt.observed >= alt.lower) & (alt.observed <= alt.upper)).astype(float), f"coverage:{case}:{scope}")
            keys = {name: set(zip(g.model, g.facility_id, g.date)) for name, g in alt.groupby('alternative')}
            check(keys['original'] == keys['facility'] == keys['pooled'], f"same_2025_days:{case}:{scope}")
            mm = c.facility_metrics(alt).sort_values(['model', 'facility_id', 'alternative'])
            saved = metrics.loc[metrics.case.eq(case) & metrics.scope.eq(scope)].sort_values(['model', 'facility_id', 'alternative'])
            equal(mm[c.METRICS], saved[c.METRICS], f"all_six_metrics:{case}:{scope}")
            recomputed.append(mm)
    historical = read('SOURCE_HISTORICAL_ORIGINAL_2022_2024.csv')
    for year in (2022, 2023, 2024):
        daily = pd.read_csv(inp(f"legacy_daily_{year}"))
        pr = pd.read_csv(inp(f"legacy_predictions_{year}"))
        for frame in (daily, pr):
            if 'year' not in frame:
                frame['year'] = year
        daily = c.validate_dates(daily, [year], c.KEYS)
        pr = c.validate_dates(pr, [year], c.KEYS + ['model'])
        joined_hist = pr.merge(daily[c.KEYS + features[:4]], on=c.KEYS, validate='many_to_one')
        for sid in spec['source_ids']:
            for model in spec['models']:
                a = joined_hist.loc[joined_hist.model.eq(model)].copy()
                a['residual'] = caches[(sid, model)].predict(a) - a.gross_heat_rate_mmbtu_per_mwh
                a['absolute_error'] = abs(a.residual)
                a['covered'] = a.absolute_error <= radii[model]
                a['winkler'] = 2 * radii[model] + 20 * np.maximum(a.absolute_error - radii[model], 0)
                re = a.groupby('facility_id')[['residual', 'absolute_error', 'covered', 'winkler']].mean()
                ref = historical.loc[historical.subset_id.eq(f"LOO_{sid}") & historical.year.eq(year) & historical.model.eq(model)].set_index('facility_id').sort_index()
                equal(re, ref[['signed_bias_mmbtu_per_mwh', 'mae_mmbtu_per_mwh', 'coverage_90', 'mean_winkler_score']], f"original_2022_24_cache_metrics:{sid}:{model}:{year}")
    pairs = read('CONDITIONAL_DAY_PAIRS.csv.gz')
    for year in (2024, 2025):
        check(not pairs.duplicated(['facility_id', f"date_{year}"]).any(), f"no_date_reuse:{year}")
    equal(np.hypot(pairs.z_temperature_2025 - pairs.z_temperature_2024, pairs.z_load_2025 - pairs.z_load_2024), pairs.distance, 'pair_frozen_distance')
    check(pairs.distance.le(.095 + 1e-12).all(), 'pair_threshold')
    base = d.query("model == 'M1'")
    for (fid, month), block in base.groupby(['facility_id', 'month']):
        a = block.loc[block.year.eq(2024)].sort_values('date')
        b = block.loc[block.year.eq(2025)].sort_values('date')
        dist = cdist(a[['z_temperature', 'z_load']], b[['z_temperature', 'z_load']])
        matched = c.cardinality_assignment(dist, .095)
        selected = pairs.loc[pairs.facility_id.eq(fid) & pairs.month.eq(month)]
        maximum = int((maximum_bipartite_matching(csr_matrix(dist <= .095), perm_type='column') >= 0).sum()) if len(a) and len(b) else 0
        check(len(selected) == maximum == len(matched), f"independent_maximum_cardinality:{fid}:{month}")
        equal(selected.distance.sum(), dist[matched[:, 0], matched[:, 1]].sum() if len(matched) else 0., f"minimum_total_distance:{fid}:{month}")
    changes = read('FACILITY_ALL_CHANGES.csv')
    re_changes = c.metric_changes(pd.concat(recomputed, ignore_index=True)).sort_values(['case', 'scope', 'model', 'facility_id'])
    saved_changes = changes.sort_values(['case', 'scope', 'model', 'facility_id'])
    delta_cols = [name for name in changes if '_facility_minus_' in name]
    equal(re_changes[delta_cols], saved_changes[delta_cols], 'complete_facility_deltas_no_outcome_filtering')
    weights = np.load(out / 'BOOTSTRAP_OPERATOR_WEIGHTS.npz')
    rng = np.random.default_rng(spec['bootstrap_seed'])
    ops = weights['operator_ids']
    expected_counts = np.array([np.bincount(rng.integers(0, len(ops), len(ops)), minlength=len(ops)) for _ in range(5000)])
    equal(expected_counts, weights['operator_counts'], 'bootstrap_seed_all5000_weights', 0)
    identity = dict(zip(weights['facility_ids'], weights['facility_operator_ids']))
    reps = read('BOOTSTRAP_5000_ALL_CASES_REPLICATES.csv.gz')
    cis = read('BOOTSTRAP_CI_ALL_CASES.csv')
    for (case, scope, model), g in changes.groupby(['case', 'scope', 'model']):
        g = g.sort_values('facility_id')
        w = c.facility_cluster_weights(expected_counts, [identity[fid] for fid in g.facility_id], ops)
        ref = reps.loc[reps.case.eq(case) & reps.scope.eq(scope) & reps.model.eq(model)].sort_values('replicate')
        for col in delta_cols:
            vals = c.weighted_facility_stat(g[col], w)
            equal(vals, ref[col], f"facility_equal_operator_bootstrap:{case}:{scope}:{model}:{col}")
            ci = cis.loc[cis.case.eq(case) & cis.scope.eq(scope) & cis.model.eq(model) & cis.quantity.eq(col)].iloc[0]
            equal([g[col].mean(), np.nanquantile(vals, .025), np.nanquantile(vals, .975)], [ci.estimate, ci.ci_low, ci.ci_high], f"saved_CI:{case}:{scope}:{model}:{col}")
    labels = read('PERMUTATION_FACILITY_LABELS_SHARED_M1_M2.csv.gz')
    rng = np.random.default_rng(spec['permutation_seed'])
    donors = np.array([rng.permutation(ids) for _ in range(1000)])
    equal(labels.offset_donor_facility_id, donors.ravel(), 'permutation_seed_shared_complete1000', 0)
    summaries = read('PERMUTATION_1000_SUMMARIES.csv')
    for scope in ('main', 'conditional'):
        for model in spec['models']:
            pf = read(f"PERMUTATION_FACILITY_METRICS_{scope.upper()}_{model}.csv.gz")
            mapping = labels.rename(columns={'target_facility_id': 'facility_id'})
            joined_labels = pf.merge(mapping, on=['replicate', 'facility_id'], validate='one_to_one')
            equal(joined_labels.donor_facility_id, joined_labels.offset_donor_facility_id, f"permutation_same_mapping:{scope}:{model}", 0)
            ref = summaries.loc[summaries.scope.eq(scope) & summaries.model.eq(model)].sort_values('replicate')
            equal(pf.groupby('replicate')[c.METRICS].mean(), ref[c.METRICS], f"permutation_facility_equal_summaries:{scope}:{model}")
    adverse = changes.query("case == 'FULL' and scope == 'main'").groupby('model').mae_facility_minus_original.apply(lambda v: int((v > 0).sum())).to_dict()
    check(adverse == {'M1': 24, 'M2': 23}, 'adverse_facilities_preserved')
    check(set(changes.case) == set(cases) and len(cache_checks) == 12, 'all6_source_cases_12_caches')
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3 if sys.platform == 'darwin' else 1024 ** 2)
    check(peak <= 4, 'registered_memory_limit')
    result = {'status': 'pass', 'scope': 'offline_packaged_derived_and_frozen_prediction_check',
        'payload_files': payload_files, 'checks': len(checks), 'check_names': checks,
        'software': {'python': sys.version.split()[0], 'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__, 'joblib': joblib.__version__},
        'runtime_seconds': time.monotonic() - start, 'peak_gib': peak,
        'new_fits': 0, 'network_disabled': True, 'cache_checks': cache_checks,
        'worsened_main_facility_counts': adverse,
        'not_executed': ['raw_EPA_EIA_NOAA_reconstruction', 'full_analysis_rerun', 'new_permutation_day_cube', 'new_figures']}
    write_json(args.output, result)
    signal.alarm(0)
    print(json.dumps({k: v for k, v in result.items() if k not in ('check_names', 'cache_checks')}, indent=2))


if __name__ == '__main__':
    main()

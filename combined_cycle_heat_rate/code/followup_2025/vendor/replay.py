from __future__ import annotations
import json
import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import SplineTransformer

class SerializedRelation:
    'Rehydrate the published coefficients and basis, without fitting.'

    def __init__(self, name):
        self.spec = SPEC['models'][name]['pipeline']
        self.transforms = []
        for t in self.spec['transformers']:
            if t['class'] == 'passthrough':
                self.transforms.append((t['input_columns'], None))
                continue
            basis = t['learned_bspline_basis'][0]
            transformer = SplineTransformer(**t['parameters'])
            transformer.bsplines_ = [BSpline(basis['knot_vector'], np.eye(basis['coefficient_matrix_shape'][0]), basis['degree'], extrapolate=True)]
            transformer.n_features_in_ = 1
            transformer.n_features_out_ = basis['coefficient_matrix_shape'][0] - 1
            self.transforms.append((t['input_columns'], transformer))

    def predict(self, frame):
        blocks = [frame[cols].to_numpy(float) if t is None else t.transform(frame[cols].to_numpy(float)) for cols, t in self.transforms]
        x = np.hstack(blocks) / np.array(self.spec['standard_scaler']['scale'])
        assert np.isfinite(x).all()
        prediction = np.sum(x * np.array(self.spec['ridge']['coefficients_after_scaling']), axis=1) + self.spec['ridge']['intercept']
        assert np.isfinite(prediction).all()
        return prediction

def fixed_support(source, target):
    s = SPEC['input_support']
    cols = s['features_in_order']
    x = target[cols].to_numpy(float)
    train = source[cols].to_numpy(float)
    z = (train - np.array(s['mean'])) / np.array(s['scale'])
    q = (x - np.array(s['mean'])) / np.array(s['scale'])
    distance = NearestNeighbors(n_neighbors=1).fit(z).kneighbors(q)[0][:, 0]
    mask = ((x >= np.array(s['minimum'])) & (x <= np.array(s['maximum']))).all(axis=1) & (distance <= s['nn_distance_threshold'] + 1e-12)
    return (mask, {'nn_distance_threshold': s['nn_distance_threshold'], 'test_support_fraction': float(mask.mean()) if len(mask) else 0.0})

def compare_numeric(old, new, keys, name, *, tol=1e-12, require_equal=False):
    old = old.sort_values(keys).reset_index(drop=True)
    new = new.sort_values(keys).reset_index(drop=True)
    if old.duplicated(keys).any() or new.duplicated(keys).any():
        raise ValueError(name + ': duplicate identity')
    both = old.merge(new, on=keys, how='outer', suffixes=('_old', '_new'), indicator=True, validate='one_to_one')
    changes = []
    maximum = 0.0
    for _, row in both.iterrows():
        identity = '|'.join((str(row[k]) for k in keys))
        if row['_merge'] != 'both':
            changes.append({'artifact': name, 'identity': identity, 'field': '__membership__', 'old': row['_merge'] == 'left_only', 'new': row['_merge'] == 'right_only', 'delta': ''})
            continue
        for c in sorted(set(old.columns) & set(new.columns) - set(keys)):
            a, b = (row[c + '_old'], row[c + '_new'])
            if pd.isna(a) and pd.isna(b):
                continue
            if isinstance(a, (int, float, np.number)) and isinstance(b, (int, float, np.number)):
                delta = float(b) - float(a)
                maximum = max(maximum, abs(delta))
                same = abs(delta) <= tol or (np.isnan(a) and np.isnan(b))
            else:
                same = str(a) == str(b)
                delta = ''
            if not same:
                changes.append({'artifact': name, 'identity': identity, 'field': c, 'old': a, 'new': b, 'delta': delta})
    if require_equal and changes:
        csvsave(CHECK / (name + '_FAILURES.csv'), changes)
        raise ValueError(f'{name}: original reproduction failed ({len(changes)} fields; max {maximum})')
    return (changes, {'artifact': name, 'old_rows': len(old), 'new_rows': len(new), 'changed_fields': len(changes), 'maximum_numeric_difference': maximum, 'tolerance': tol})

def primary():
    gate = json.loads((OUT / 'INVENTORY_RESULT.json').read_text())
    assert gate['proceed']
    CHECK.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    DATA.mkdir(exist_ok=True)
    hits = pd.read_csv(OUT / 'ACTUAL_HITS.csv')
    assert set(hits.role) == {'external'}
    pairs = hits[['year', 'facility_id', 'path', 'source_sha256']].drop_duplicates()
    if not pairs.year.eq(2024).all():
        raise RuntimeError('Additional-year correction requires its original year-specific evaluator; do not apply 2024 seeds blindly.')
    m = original_national()
    source = pd.read_csv(SOURCE)
    fitted = {n: SerializedRelation(n) for n in ('M1', 'M2')}
    radii = {n: SPEC['models'][n]['source_residual_interval']['half_width_mmbtu_per_mwh'] for n in fitted}
    checks = []
    changes = []
    masks = []
    unit_rows = []
    endpoint_diffs = []
    original_metrics = pd.read_csv(SUP / 'results/FACILITY_MODEL_METRICS.csv')
    metrics = original_metrics.copy()
    endpoints = {y: pd.read_csv(p) for y, p in ENDPOINTS.items()}
    predictions = {y: pd.read_csv(p) for y, p in PREDS.items()}
    for y in endpoints:
        e = endpoints[y]
        for name in fitted:
            f = predictions[y].loc[predictions[y].model.eq(name)]
            j = f.merge(e, on=['facility_id', 'date'], suffixes=('_frozen', ''), validate='one_to_one')
            delta = np.abs(fitted[name].predict(j) - j.prediction)
            maximum = float(delta.max())
            assert maximum <= 1e-12, (y, name, maximum)
            expected = set(zip(f.facility_id, f.date))
            mask, _ = fixed_support(source, e)
            actual = set(zip(e.loc[mask, 'facility_id'], e.loc[mask, 'date']))
            assert expected == actual, (y, name, 'original support identity')
            checks.append({'year': y, 'model': name, 'rows': len(f), 'prediction_max_difference': maximum, 'support_identity': True, 'fitted': False})
    csvsave(CHECK / 'SERIALIZED_PRIMARY_REPRODUCTION.csv', checks)
    cohort = pd.read_csv(REPO / 'data/eligibility/PREHOURLY_COHORT.csv')
    oldmanifest = pd.read_csv(SUP / 'source_provenance/CAMPD_2024_SOURCE_MANIFEST.csv').set_index('facility_id')
    for hit in pairs.itertuples(index=False):
        fid, y = (int(hit.facility_id), int(hit.year))
        assert sha(hit.path) == oldmanifest.loc[fid, 'carrier_sha256'] == hit.source_sha256
        row = cohort.loc[cohort.facility_id.eq(fid)].iloc[0]
        raw = pd.read_csv(hit.path, dtype={'Unit ID': str}, low_memory=False)
        tagged, old_ok = eligible_rows(raw, normalized=False)
        _, new_ok = eligible_rows(raw, normalized=True)
        csvsave(DATA / f'combined_cycle_heat_rate_{fid}_{y}_FLAG_AUDIT.csv', tagged[['Facility ID', 'Unit ID', 'Date', 'Hour', 'raw_heat_input_flag', 'normalized_heat_input_flag']].assign(strict_included=old_ok, corrected_included=new_ok))
        weather = m.baseline.load_weather(weather_for(row))
        meta = {'nameplate_mw': float(row.qualifying_nameplate_mw), 'name': row.facility_name, 'operator_id': int(row.operator_id), 'cooling_type': 'not_used_by_m1_m2', 'station_distance_km': float(row.selected_station_distance_km)}
        local = {}
        for mode in ('old', 'corrected'):
            r = raw.loc[raw['Unit Type'].str.strip().str.casefold().eq('combined cycle')].copy()
            if mode == 'corrected':
                r['Heat Input Measure Indicator'] = r['Heat Input Measure Indicator'].map(canonical_flag)
            hourly = m.baseline.aggregate_facility_hours(r, meta['nameplate_mw'])
            daily = m.baseline.aggregate_facility_days(hourly, weather, fid, meta)
            if len(daily) < 150:
                raise RuntimeError('Corrected 150-day eligibility no longer met; record scope before integration.')
            met, pred = m.evaluate_facility(daily, source, fitted, radii, row)
            d = daily.copy()
            d['date'] = d.date.dt.strftime('%Y-%m-%d')
            pr = pd.concat(pred, ignore_index=True)
            pr['date'] = pr.date.dt.strftime('%Y-%m-%d')
            local[mode] = (d, pd.DataFrame(met).assign(year=y), pr)
        old_d, old_m, old_p = local['old']
        new_d, new_m, new_p = local['corrected']
        for label, old, actual, keys in [('daily', endpoints[y].loc[endpoints[y].facility_id.eq(fid)], old_d, ['facility_id', 'date']), ('metrics', original_metrics.loc[original_metrics.year.eq(y) & original_metrics.facility_id.eq(fid)], old_m, ['year', 'facility_id', 'model']), ('predictions', predictions[y].loc[predictions[y].facility_id.eq(fid)], old_p, ['facility_id', 'date', 'model'])]:
            _, check = compare_numeric(old, actual, keys, f'{fid}_{y}_{label}_baseline', tol=1e-10 if label == 'daily' else 1e-12, require_equal=True)
            checks.append(check)
        for label, a, b, keys in [('daily', old_d, new_d, ['facility_id', 'date']), ('metrics', old_m, new_m, ['year', 'facility_id', 'model']), ('predictions', old_p, new_p, ['facility_id', 'date', 'model'])]:
            diff, check = compare_numeric(a, b, keys, f'{fid}_{y}_{label}')
            changes += diff
            checks.append(check)
        oldset = set(old_p.date)
        newset = set(new_p.date)
        masks += [{'year': y, 'facility_id': fid, 'date': date, 'old_supported': date in oldset, 'new_supported': date in newset} for date in sorted(set(old_d.date) | set(new_d.date))]
        unit_rows.append({'facility_id': fid, 'year': y, 'historical_raw_sha_matches_manifest': True, 'raw_rows': len(raw), 'additional_measured_rows': int((new_ok & ~old_ok).sum()), 'old_endpoint_days': len(old_d), 'new_endpoint_days': len(new_d), 'old_matched_days': len(oldset), 'new_matched_days': len(newset), 'added_support_dates': sorted(newset - oldset), 'removed_support_dates': sorted(oldset - newset)})
        for name, frame, replacement in [('metrics', metrics, new_m)]:
            selected = frame.year.eq(y) & frame.facility_id.eq(fid)
            for c in frame.columns:
                if c not in replacement:
                    replacement[c] = frame.loc[selected, c].iloc[0]
            result = pd.concat([frame.loc[~selected], replacement[frame.columns]], ignore_index=True).sort_values(['year', 'facility_id', 'model']).reset_index(drop=True)
            if name == 'metrics':
                metrics = result
        endpoints[y] = pd.concat([endpoints[y].loc[~endpoints[y].facility_id.eq(fid)], new_d], ignore_index=True).sort_values(['facility_id', 'date']).reset_index(drop=True)
        predictions[y] = pd.concat([predictions[y].loc[~predictions[y].facility_id.eq(fid)], new_p], ignore_index=True).sort_values(['facility_id', 'model', 'date']).reset_index(drop=True)
    for y in endpoints:
        csvsave(DATA / f'combined_cycle_heat_rate_{y}_CORRECTED_DAILY_ENDPOINT.csv', endpoints[y])
        csvsave(DATA / f'combined_cycle_heat_rate_{y}_CORRECTED_MATCHED_PREDICTIONS.csv', predictions[y])
    csvsave(RESULTS / 'FACILITY_MODEL_METRICS.csv', metrics)
    csvsave(OUT / 'OLD_NEW_PRIMARY_DIFFERENCES.csv', changes)
    csvsave(OUT / 'SUPPORT_MEMBERSHIP_COMPARISON.csv', masks)
    jsave(CHECK / 'PRIMARY_REPLAY.json', {'checks': checks, 'affected': unit_rows, 'raw_version_resolved': 'original archived bytes match historical manifest', 'original_model_fit_calls': 0})
    summaries = []
    old_summaries = pd.read_csv(SUP / 'results/MULTIYEAR_NATIONAL_SUMMARY.csv')
    for y, old_summary in old_summaries.groupby('year'):
        if y not in set(pairs.year):
            summaries.append(old_summary)
            continue
        year_metrics = metrics.loc[metrics.year.eq(y)].copy()
        summary = m.summarize_national(year_metrics)
        for c in old_summary:
            if c not in summary:
                summary[c] = old_summary[c].iloc[0]
        summaries.append(summary[old_summaries.columns])
    national = pd.concat(summaries, ignore_index=True)
    csvsave(RESULTS / 'MULTIYEAR_NATIONAL_SUMMARY.csv', national)
    equal = national.groupby('model', as_index=False).agg(final_years=('year', 'nunique'), equal_year_mean_coverage=('facility_equal_mean_coverage', 'mean'), equal_year_mean_operator_coverage=('operator_equal_mean_coverage', 'mean'), equal_year_mean_median_width_over_local_mae=('median_width_over_local_mae', 'mean'), equal_year_mean_median_mae_mmbtu_per_mwh=('median_mae_mmbtu_per_mwh', 'mean'))
    csvsave(RESULTS / 'FINAL_2022_2024_EQUAL_YEAR_SUMMARY.csv', equal)
    table = pd.read_csv(SUP / 'results/RESULTS_TABLES.csv')
    parts = []
    for tid, f in [('annual_national_summary', national), ('final_2022_2024_equal_year_summary', equal)]:
        old = table.loc[table.table_id.eq(tid)]
        f = f.copy()
        for c in table:
            if c not in f:
                f[c] = old[c].iloc[0]
        parts.append(f[table.columns])
    csvsave(RESULTS / 'RESULTS_TABLES.csv', pd.concat(parts, ignore_index=True))
    common_panel(metrics, predictions)
    print(json.dumps({'affected': unit_rows, 'corrected_metrics': metrics.loc[metrics.facility_id.eq(56349) & metrics.year.eq(2024)].to_dict('records')}, indent=2), flush=True)

def common_panel(metrics, predictions):
    panel, identity = panel_analysis.common_facility_panel(metrics)
    allpred = pd.concat([d.assign(year=y) for y, d in predictions.items()], ignore_index=True)
    sub = allpred.loc[allpred.facility_id.isin(panel.facility_id.unique())]
    facility = panel_analysis.supported_and_covered_facility_rows(panel, sub)
    outputs = {'COMMON_FACILITY_PANEL': panel, 'COMMON_PANEL_GATE': panel_analysis.common_panel_gate_rows(panel), 'COMMON_PANEL_SUMMARY': panel_analysis.summarize_common_panel(panel), 'SUPPORTED_AND_COVERED_FACILITY': facility, 'SUPPORTED_AND_COVERED_SUMMARY': panel_analysis.summarize_supported_and_covered(facility)}
    for label, frame in outputs.items():
        csvsave(RESULTS / f'{label}.csv', frame)
    bias = annual_persistence.bias_persistence_rows(panel)
    recurrence = annual_persistence.coverage_recurrence_rows(panel)
    csvsave(RESULTS / 'BIAS_PERSISTENCE.csv', bias)
    csvsave(RESULTS / 'COVERAGE_RECURRENCE.csv', recurrence)
    persistence_identity = {'rows': len(panel), 'facilities': int(panel.facility_id.nunique()), 'years': [2022, 2023, 2024], 'relations': ['M1', 'M2']}
    jsave(RESULTS / 'PERSISTENCE_SUMMARY.json', annual_persistence.persistence_summary(persistence_identity, bias, recurrence))
    jsave(CHECK / 'COMMON_PANEL_IDENTITY.json', {**identity, 'rows': len(panel), 'complete_cross_product': len(panel) == panel.facility_id.nunique() * 6, 'corrected_historical_gate': panel_analysis.evaluate_common_panel_gate(outputs['COMMON_PANEL_GATE'])})

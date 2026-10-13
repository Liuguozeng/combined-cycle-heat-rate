'Manifest-only, offline dependency injection for the frozen combined_cycle_heat_rate follow-up.'
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def resolve_inputs(manifest_path, contract):
    manifest_path = Path(manifest_path).resolve()
    obj = read(manifest_path)
    if obj.get('paper') != 'combined_cycle_heat_rate' or contract.get('paper') != 'combined_cycle_heat_rate':
        raise ValueError('Cross-paper input manifest')
    if obj.get('schema_version') != 1:
        raise ValueError('Unsupported input manifest schema')
    supplied = {}
    for row in obj['inputs']:
        if row['id'] in supplied:
            raise ValueError('Duplicate input identity: ' + row['id'])
        supplied[row['id']] = row
    requirements = {r['id']: r for r in contract['inputs']}
    if len(requirements) != len(contract['inputs']):
        raise ValueError('Duplicate contract identity')
    if set(supplied) - set(requirements):
        raise ValueError('Undeclared input identity')
    result = {}
    for key, required in requirements.items():
        row = supplied.get(key)
        if row is None:
            raise FileNotFoundError('NOT_EXECUTED: missing lawful input ' + key)
        if 'sha256' in row and row['sha256'] != required['sha256']:
            raise ValueError('Input manifest hash cannot override pinned hash: ' + key)
        path = (manifest_path.parent / row['path']).resolve()
        if not path.is_file():
            raise FileNotFoundError('NOT_EXECUTED: missing lawful input ' + key)
        if digest(path) != required['sha256']:
            raise ValueError('Input/cache hash mismatch: ' + key)
        result[key] = path
    if not any(r['role'] == 'expected' for r in requirements.values()):
        raise FileNotFoundError('NOT_EXECUTED: missing expected contract')
    return result


def validate_output(path, inputs):
    path = path.resolve()
    if path == HERE or HERE in path.parents or path in HERE.parents:
        raise ValueError('Output overlaps code package')
    if any(path == p or path in p.parents or p in path.parents for p in inputs.values()):
        raise ValueError('Output overlaps an input')
    if path.exists() and any(path.iterdir()):
        raise ValueError('Output directory must be empty; never overwrite a run')
    return path


def load_vendor(name, injected=None):
    provenance = read(HERE / 'CODE_PROVENANCE.json')
    expected = next(r['export_sha256'] for r in provenance if r['file'] == name + '.py')
    path = HERE / 'vendor' / (name + '.py')
    if digest(path) != expected:
        raise ValueError('Implementation hash mismatch: ' + name)
    module = types.ModuleType('heat_rate_portable_' + name)
    module.__file__ = str(path)
    module.__dict__.update(injected or {})
    exec(compile(path.read_text(), str(path), 'exec'), module.__dict__)
    return module


def check_environment(contract):
    actual = {k: importlib.metadata.version(k) for k in contract['software'] if k != 'python'}
    actual['python'] = '.'.join(map(str, sys.version_info[:3]))
    if actual != contract['software']:
        raise RuntimeError('Locked environment mismatch: ' + json.dumps(actual))
    return actual


def install_guard(output, inputs, extra_read_roots=()):
    'Defense in depth: Python I/O audit plus no network/process escape.'
    allowed_files = {p.resolve() for p in inputs.values()}
    read_roots = [HERE, output, Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve(),
                  *(Path(p).resolve() for p in extra_read_roots)]
    counts = dict(read_events=0, write_events=0, denied_events=0, network_attempts=0)

    def allowed(path, write):
        if isinstance(path, int) or path is None:
            return True
        p = Path(os.fsdecode(path)).resolve()
        if write:
            return p == output or output in p.parents
        return p in allowed_files or any(p == root or root in p.parents for root in read_roots) or str(p) in {'/dev/null', '/dev/urandom'}

    def audit(event, args):
        if event.startswith(('socket.', 'subprocess.')) or event in {'os.system', 'os.fork', 'os.posix_spawn'}:
            counts['denied_events'] += 1
            counts['network_attempts'] += int(event.startswith('socket.'))
            raise PermissionError('Offline guard blocked network/process operation')
        if event == 'open':
            path, mode, flags = args
            write = bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            counts['write_events' if write else 'read_events'] += 1
            if not allowed(path, write):
                counts['denied_events'] += 1
                raise PermissionError('I/O guard blocked undeclared path')
        elif event in {'os.listdir', 'os.scandir'}:
            if not allowed(args[0], False):
                counts['denied_events'] += 1
                raise PermissionError('I/O guard blocked undeclared directory')
        elif event in {'os.mkdir', 'os.remove', 'os.rmdir', 'os.chmod', 'os.utime'}:
            if not allowed(args[0], True):
                counts['denied_events'] += 1
                raise PermissionError('I/O guard blocked external mutation')
        elif event in {'os.rename', 'os.link', 'os.symlink'}:
            targets = args[:2] if event == 'os.rename' else args[1:2]
            if not all(allowed(p, True) for p in targets):
                counts['denied_events'] += 1
                raise PermissionError('I/O guard blocked external mutation')
    sys.addaudithook(audit)
    return counts


def compare_tables(expected, actual, label, tolerance, keys=None):
    import numpy as np
    import pandas as pd
    if not actual.is_file():
        raise ValueError('Missing recomputed output: ' + label)
    a = pd.read_csv(expected)
    b = pd.read_csv(actual)
    # Timestamp is execution metadata, not a scientific comparison field.
    ignored = [c for c in ['scan_utc'] if c in a.columns]
    a = a.drop(columns=ignored)
    b = b.drop(columns=ignored)
    if keys:
        for frame in (a, b):
            if frame[keys].isna().any().any() or frame.duplicated(keys).any():
                raise ValueError('Duplicate/missing result identity: ' + label)
        a = a.sort_values(keys).reset_index(drop=True)
        b = b.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_exact=False, check_dtype=False, atol=tolerance, rtol=0)
    differences = {}
    for col in a.select_dtypes(include='number').columns:
        values = (a[col] - b[col]).abs()
        differences[col] = float(values.max()) if values.notna().any() else 0.0
    return dict(artifact=label, status='PASS', rows=len(a), columns=len(a.columns),
                tolerance=tolerance, rtol=0, maximum_numeric_differences=differences,
                excluded_execution_metadata=ignored, expected_sha256=digest(expected), actual_sha256=digest(actual))


def table_signature(frame):
    ignored = [c for c in ['scan_utc'] if c in frame.columns]
    scientific = frame.drop(columns=ignored)
    content = scientific.to_csv(index=False).encode('utf-8')
    return dict(rows=len(scientific), columns=list(scientific.columns),
                canonical_csv_sha256=hashlib.sha256(content).hexdigest(),
                excluded_execution_metadata=ignored)


def run(contract, inputs, output):
    import pandas as pd
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import SplineTransformer, StandardScaler

    def forbidden_fit(*args, **kwargs):
        raise RuntimeError('Predictive model fitting is forbidden')
    for cls in [Ridge, SplineTransformer, StandardScaler]:
        cls.fit = forbidden_fit
    work = output / 'work'
    work.mkdir()
    for row in contract['inputs']:
        if 'stage' not in row or row['id'] == 'FIXED_FOLLOWUP_135.csv':
            continue
        rel = Path(row['stage'])
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('Unsafe staged path')
        target = work / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(inputs[row['id']])
    followup = work / 'followup'
    annual_root = followup / 'annual_evaluation'
    weather_root = followup / 'weather_processing'
    sup = work / 'supplementary'
    q = followup / 'qualification'
    bridge_root = followup / 'bridge'
    bridge_root.mkdir()
    annual_root.mkdir(exist_ok=True)
    # Intake regenerates the fixed table; its expected carrier is never a write target.
    inp = load_vendor('inputs')
    intake = load_vendor('intake', dict(HERE=followup, RAW=followup / 'raw', SUP=sup, ARCH=work / 'archive',
        locate_header=inp.locate_header, linked_capacity=inp.linked_capacity, unique_keys=inp.unique_keys,
        annual_boundary_screen=inp.annual_boundary_screen))
    print('STAGE EIA_2025_RAW_QUALIFICATION', flush=True)
    intake.main()
    spec = read(inputs['frozen_model'])
    base = load_vendor('baseline')
    flags = load_vendor('flags', dict(DICTIONARY=inputs['dictionary']))
    replay = load_vendor('replay', dict(SPEC=spec))
    stage3 = load_vendor('stage3')
    base.matched_support_mask = replay.fixed_support
    national = load_vendor('national', dict(baseline=base, stage3=stage3))
    fitted = {n: replay.SerializedRelation(n) for n in ['M1', 'M2']}
    radii = {n: spec['models'][n]['source_residual_interval']['half_width_mmbtu_per_mwh'] for n in fitted}
    source = pd.read_csv(inputs['source_daily'])
    modules = lambda: (base, flags, replay, national, fitted, radii, source)
    bridge = load_vendor('bridge', dict(Q=q, OUT=bridge_root, SUP=sup, ARCH=work / 'archive',
        EXP=work / 'reference_exp', sha=digest, unique_keys=inp.unique_keys, modules=modules))
    print('STAGE EPA_2024_ORIGINAL_BRIDGE', flush=True)
    bridge.original_bridge()
    annual = load_vendor('annual', dict(HERE=annual_root, Q=q, OUT=annual_root / 'results',
        SUP=sup, ARCH=work / 'archive', sha=digest, unique_keys=inp.unique_keys,
        modules=modules, metadata=bridge.metadata, load_frozen=load_vendor))
    print('STAGE EPA_2025_PREWEATHER', flush=True)
    annual.prepare()
    wx = load_vendor('weather')
    dc = load_vendor('daily_consistency')
    (weather_root / 'weather.py').symlink_to(HERE / 'vendor/weather.py')
    correction = load_vendor('run_correction', dict(HERE=weather_root, BASE=followup, RAW=weather_root / 'raw',
        TABLES=weather_root / 'tables', CACHE=weather_root / 'scan', VALID=weather_root / 'validation', sha=digest,
        HOURLY_RULE=wx.RULE, RULE=dc.DAILY_RULE, FIELDS=dc.FIELDS,
        check_observation=wx.check_observation, select_hours=wx.select_hours, daily_from_hours=wx.daily_from_hours,
        inspect_ssod=dc.inspect_ssod, apply_daily_consistency=dc.apply_daily_consistency))
    print('STAGE WEATHER_RAW_REPLAY', flush=True)
    correction.run(finalize=True)
    status = read(weather_root / 'STATUS.json')
    if status['partial'] or status['unknown_qc_rows'] or any(r['scan_status'] == 'scan_failed' for r in status['scan_failures']):
        raise ValueError('Weather replay did not reach the frozen qualified state')
    decisions = pd.read_csv(weather_root / 'station_year_decisions.csv')
    if decisions.ssod_product_status.eq('read_error').any():
        raise ValueError('Unadjudicated SSOD parse error')
    independent_weather = load_vendor('verify_weather').check(weather_root)
    save(output / 'WEATHER_INDEPENDENT.json', independent_weather)
    # Preserve the original identity guard, now against injected immutable inputs.
    save(annual_root / 'BASELINE_IDENTITY.json', dict(candidate_root=str(sup), files=[
        dict(path='model_spec/FIXED_RELATION_SPECIFICATION.json', actual=digest(inputs['frozen_model']))]))
    print('STAGE FROZEN_ANNUAL_EVALUATION', flush=True)
    annual.weather_adapter(weather_root)
    annual.evaluate(annual.OUT / 'WEATHER_EVALUATION_DAILY.csv.gz', annual.OUT / 'WEATHER_EVALUATION_DECISIONS.csv')
    verifier = load_vendor('verify', dict(SUP=sup))
    independent, counts = verifier.check(annual.OUT)
    counts.to_csv(annual.OUT / 'INDEPENDENT_COUNTS.csv', index=False)
    save(output / 'INDEPENDENT_VERIFICATION.json', independent)
    comparisons = accept_results(contract, inputs, work, output)
    return dict(status='PASS', paper='combined_cycle_heat_rate', mode='raw_replay', raw_inputs_recomputed_this_command=True,
        frozen_predictions_recomputed_this_command=True, stages=dict(epa_2024='recomputed_from_raw', epa_2025='recomputed_from_raw',
        eia_2025='recomputed_from_raw_workbooks_and_annual_EPA', eia_2024='frozen_qualification_validated_not_rebuilt',
        weather='raw_products_replayed_with_original_missing_carrier_decisions',
        station_mapping='hash_validated_adjudications_not_re_adjudicated', frozen_evaluation='recomputed',
        historical_2022_2023='not_rerun_not_modified'), new_model_fits=0,
        bootstrap_replicates=5000, independent=independent, independent_weather=independent_weather,
        weather_scope={k: status[k] for k in ['station_years', 'hourly_scanned', 'hourly_missing_station_years',
            'official_daily_product_only_station_years', 'approved_station_years', 'not_approved_station_years', 'correction_rows', 'unresolved_rows']},
        comparisons=len(comparisons), expected_substituted_for_computation=False)


def accept_results(contract, inputs, work, output):
    import pandas as pd
    comparisons = []
    for row in contract['inputs']:
        if 'compare' not in row:
            continue
        try:
            comparison = compare_tables(inputs[row['id']], work / row['compare'], row['compare'], row['tolerance'], row['keys'])
        except Exception as exc:
            save(output / 'RESULT_DIFFERENCES.json', dict(status='FAIL_STOPPED', failed_artifact=row['compare'],
                detail=str(exc), completed=comparisons, original_results_overwritten=False))
            raise ValueError('Result difference; stopped without changing originals: ' + row['compare']) from exc
        comparisons.append(comparison)
    for expected in read(inputs['expected_weather_table_signatures']):
        actual = table_signature(pd.read_csv(work / expected['artifact']))
        if actual != expected['signature']:
            save(output / 'RESULT_DIFFERENCES.json', dict(status='FAIL_STOPPED', failed_artifact=expected['artifact'],
                reason='Full table signature differs', expected_signature=expected['signature'], actual_signature=actual,
                completed=comparisons, original_results_overwritten=False))
            raise ValueError('Private-table result difference; stopped: ' + expected['artifact'])
        comparisons.append(dict(artifact=expected['artifact'], status='PASS', verification='exact_canonical_table_signature',
            rows=actual['rows'], columns=len(actual['columns']), excluded_execution_metadata=actual['excluded_execution_metadata'],
            canonical_csv_sha256=actual['canonical_csv_sha256'], restricted_raw_table_redistributed=False))
    save(output / 'RESULT_DIFFERENCES.json', dict(status='PASS', comparisons=comparisons))
    return comparisons


def verify_existing(contract, inputs, previous, output):
    import pandas as pd
    prior = read(previous / 'REPLAY_STATUS.json')
    if prior.get('status') != 'PASS' or prior.get('paper') != 'combined_cycle_heat_rate':
        raise ValueError('Existing output lacks a successful combined_cycle_heat_rate raw-replay receipt')
    work = previous / 'work'
    weather_root = work / 'followup/weather_processing'
    decisions = pd.read_csv(weather_root / 'station_year_decisions.csv')
    if decisions.ssod_product_status.eq('read_error').any():
        raise ValueError('Unadjudicated SSOD parse error')
    independent_weather = load_vendor('verify_weather').check(weather_root)
    independent, counts = load_vendor('verify', dict(SUP=work / 'supplementary')).check(work / 'followup/annual_evaluation/results')
    counts.to_csv(output / 'INDEPENDENT_COUNTS.csv', index=False)
    comparisons = accept_results(contract, inputs, work, output)
    save(output / 'WEATHER_INDEPENDENT.json', independent_weather)
    save(output / 'INDEPENDENT_VERIFICATION.json', independent)
    return dict(status='PASS', paper='combined_cycle_heat_rate', mode='verify_existing_output', raw_inputs_recomputed_this_command=False,
                frozen_predictions_recomputed_this_command=False, new_model_fits=0,
                prior_raw_replay_receipt_sha256=digest(previous / 'REPLAY_STATUS.json'),
                independent=independent, independent_weather=independent_weather, comparisons=len(comparisons))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--verify-existing-output', type=Path,
                        help='Verify completed results only; never label this as raw replay')
    args = parser.parse_args()
    contract = read(HERE / 'INPUT_CONTRACT.json')
    environment = check_environment(contract)
    inputs = resolve_inputs(args.input_manifest, contract)
    previous = args.verify_existing_output.resolve() if args.verify_existing_output else None
    output = validate_output(args.output_dir, {**inputs, '_previous': previous} if previous else inputs)
    output.mkdir(parents=True)
    os.environ['TMPDIR'] = str(output / 'tmp')
    (output / 'tmp').mkdir()
    sys.dont_write_bytecode = True
    # Import dependencies before closing filesystem access to their installation.
    import numpy
    import pandas
    import scipy.spatial
    import sklearn.linear_model
    import sklearn.preprocessing
    import sklearn.neighbors
    import openpyxl
    io_counts = install_guard(output, {**inputs, '_manifest': args.input_manifest.resolve()}, [previous] if previous else [])
    try:
        report = verify_existing(contract, inputs, previous, output) if previous else run(contract, inputs, output)
        if any(digest(inputs[r['id']]) != r['sha256'] for r in contract['inputs']):
            raise ValueError('Read-only input changed during run')
        report.update(environment=environment, input_manifest_sha256=digest(args.input_manifest),
                      input_contract_sha256=digest(HERE / 'INPUT_CONTRACT.json'),
                      inputs_read_only=True, undeclared_history_reads=0, offline_guard=io_counts)
        save(output / 'REPLAY_STATUS.json', report)
        output_files = [dict(path=str(p.relative_to(output)), sha256=digest(p), bytes=p.stat().st_size)
                        for p in output.rglob('*') if p.is_file() and not p.is_symlink()]
        save(output / 'OUTPUT_MANIFEST.json', dict(paper='combined_cycle_heat_rate', files=output_files))
        print(json.dumps(report, indent=2))
    except Exception as exc:
        save(output / 'REPLAY_STATUS.json', dict(status='FAIL_STOPPED', reason=str(exc), inputs_read_only=True,
             original_results_overwritten=False, offline_guard=io_counts))
        raise


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(json.dumps(dict(status='NOT_EXECUTED' if isinstance(exc, FileNotFoundError) else 'FAIL', reason=str(exc))), file=sys.stderr)
        raise SystemExit(2)

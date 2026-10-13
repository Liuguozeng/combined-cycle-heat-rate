'Read-only inputs and isolated frozen-method replay for this paper only.'
import argparse
import collections
import csv
import hashlib
import json
import shutil
import sys
import tempfile
import types
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT/'reproduction'

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())

def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def csvsave(path,frame):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if hasattr(frame,'to_csv'):frame.to_csv(path,index=False)
    elif frame:
        with path.open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(frame[0]));w.writeheader();w.writerows(frame)

def load(name, injected=None):
    path=REP/'code'/(name+'.py')
    expected=next(x['export_sha256'] for x in read(REP/'CODE_PROVENANCE.json') if x['file']==path.name)
    if digest(path)!=expected:raise ValueError('Implementation hash mismatch: '+name)
    m=types.ModuleType('paper_replay_'+name);m.__file__=str(path)
    m.__dict__.update(injected or {})
    exec(compile(path.read_text(),str(path),'exec'),m.__dict__)
    return m

def required_artifacts(primary_only=False):
    names=read(REP/'RESULT_CONTRACT.json')['required_artifacts']
    if len(names)!=len(set(names)):raise ValueError('Duplicate expected artifact')
    declared={r['artifact'] for r in read(ROOT/'REPRODUCTION_SCOPE_MAP.json') if r['verification_mode']=='recomputed'}
    if set(names)!=declared:raise ValueError('Scope map and replay artifact contract differ')
    if primary_only:
        names=[n for n in names if 'SOURCE_COMPOSITION' not in n and
               ('_' not in n or n.startswith('results/'))]
    for name in names:
        path=Path(name)
        if path.is_absolute() or '..' in path.parts or not name.startswith('results/'):
            raise ValueError('Invalid result path: '+name)
        if not (ROOT/path).is_file():raise FileNotFoundError('Required expected result missing: '+name)
    return names

def resolve_inputs(args):
    requirements=read(REP/'INPUT_REQUIREMENTS.json'); supplied={}
    if args.input_manifest:
        mf=args.input_manifest.resolve();obj=read(mf)
        if obj.get('paper')!=requirements['paper']:raise ValueError('Cross-paper input manifest')
        for row in obj['inputs']:
            if row['id'] in supplied:raise ValueError('Duplicate input identity: '+row['id'])
            p=Path(row['path']);supplied[row['id']]=(mf.parent/p).resolve() if not p.is_absolute() else p
    else:
        base=args.source_workspace.resolve()
        # Compatibility only: discover the original carriers by file identity, never execute old code.
        byname=collections.defaultdict(list)
        for p in base.rglob('*'):
            if p.is_file():byname[p.name].append(p)
        hit=base/'original_source_records/ACTUAL_HITS.csv'
        if hit.is_file():
            for row in csv.DictReader(hit.open()):
                p=Path(row['path'])
                if p.is_file():
                    byname[p.name].append(p)
                    for q in p.parent.parent.rglob('*.csv*'):byname[q.name].append(q)
        for row in requirements['inputs']:
            if 'bundled_path' not in row:
                supplied[row['id']]=next((p for p in byname[row['filename']] if digest(p)==row['sha256']),None)
    allowed={r['id'] for r in requirements['inputs'] if 'bundled_path' not in r}
    if set(supplied)-allowed:raise ValueError('Undeclared external input IDs: '+str(sorted(set(supplied)-allowed)))
    resolved={}
    for row in requirements['inputs']:
        p=ROOT/row['bundled_path'] if 'bundled_path' in row else supplied.get(row['id'])
        if p is None or not p.is_file():raise FileNotFoundError('NOT EXECUTED: missing lawful input '+row['id'])
        if digest(p)!=row['sha256']:raise ValueError('Input version/hash mismatch: '+row['id'])
        resolved[row['id']]=p.resolve()
    return requirements,resolved

def compare_outputs(actual_root,names):
    import pandas as pd
    results=[]
    for name in names:
        expected=ROOT/name;actual=actual_root/Path(name).name
        if not expected.is_file() or not actual.is_file():raise FileNotFoundError('Required artifact absent: '+name)
        if expected.suffix in {'.csv','.gz'}:
            a=pd.read_csv(expected);b=pd.read_csv(actual)
            pd.testing.assert_frame_equal(a,b,check_exact=True)
        elif read(expected)!=read(actual):raise ValueError('JSON result mismatch: '+name)
        results.append({'artifact':name,'expected_sha256':digest(expected),'actual_sha256':digest(actual),'serialized_cells_equal':True})
    return results


def heat_rate(temp,inputs,primary_only=False):
    import numpy as np
    import pandas as pd
    flags=load('flags',dict(DICTIONARY=inputs['dictionary']))
    baseline=load('baseline');stage3=load('stage3');national=load('national',dict(baseline=baseline,stage3=stage3))
    panel_analysis=load('panel');annual_persistence=load('persistence')
    old=temp/'reference';(old/'results').mkdir(parents=True);(old/'source_provenance').mkdir()
    for key,p in inputs.items():
        if key.startswith('prior_combined_cycle_heat_rate_'):(old/'results'/p.name).symlink_to(p)
    (old/'source_provenance/CAMPD_2024_SOURCE_MANIFEST.csv').symlink_to(inputs['prior_campd_manifest'])
    repo=temp/'repository';(repo/'data/eligibility').mkdir(parents=True)
    (repo/'data/eligibility/PREHOURLY_COHORT.csv').symlink_to(inputs['cohort'])
    hits=[dict(role='external',year=int(k.split('_')[-1]),facility_id=int(k.split('_')[-2]),path=str(p),source_sha256=digest(p))
          for k,p in inputs.items() if k.startswith('raw_')]
    csvsave(temp/'ACTUAL_HITS.csv',hits);save(temp/'INVENTORY_RESULT.json',{'proceed':True})
    source_ids=(7153,55042,55107,55294,55322,56047)
    r=load('replay',dict(OUT=temp,WORK=temp,SUP=old,REPO=repo,SOURCE=inputs['source_daily'],
        ENDPOINTS={y:inputs['endpoints_'+str(y)] for y in (2022,2023,2024)},PREDS={y:inputs['predictions_'+str(y)] for y in (2022,2023,2024)},
        SPEC=read(ROOT/'model_spec/FIXED_RELATION_SPECIFICATION.json'),
        RESULTS=temp/'corrected_results',DATA=temp/'corrected_data',CHECK=temp/'validation',
        SOURCE_IDS=source_ids,original_national=lambda:national,
        weather_for=lambda row:inputs['weather_'+str(int(row.facility_id))],
        panel_analysis=panel_analysis,annual_persistence=annual_persistence,
        canonical_flag=flags.canonical_flag,eligible_rows=flags.eligible_rows,sha=digest,jsave=save,csvsave=csvsave))
    baseline.matched_support_mask=r.fixed_support
    r.primary()
    diagnostics={'scope':'Incremental endpoint correction, unchanged frozen references and complete current reductions','new_model_fits':0,'warning_counts':{},'cached_prediction_independent_max_difference':0.0}
    if primary_only:return r.RESULTS,diagnostics
    import joblib
    s=load('subsets');a=load('assemble')
    cache=REP/'model_cache';cm=read(cache/'MODEL_CACHE_MANIFEST.json')
    for row in cm:
        if digest(cache/row['file'])!=row['sha256']:raise ValueError('Subset cache identity mismatch')
    ends={y:pd.read_csv(r.DATA/f'combined_cycle_heat_rate_{y}_CORRECTED_DAILY_ENDPOINT.csv') for y in (2022,2023,2024)}
    preds={y:pd.read_csv(r.DATA/f'combined_cycle_heat_rate_{y}_CORRECTED_MATCHED_PREDICTIONS.csv') for y in (2022,2023,2024)}
    joined={(y,n):s.join_frozen_support_rows(preds[y],ends[y],year=y,model=n) for y in ends for n in ('M1','M2')}
    panel=set(pd.read_csv(r.RESULTS/'COMMON_FACILITY_PANEL.csv').facility_id)
    metrics=[];full=[];common=[];counts=collections.Counter();maximum=0.0
    allowed={'divide by zero encountered in matmul','overflow encountered in matmul','invalid value encountered in matmul'}
    for subset in s.source_subsets(source_ids):
        for n in ('M1','M2'):
            model=joblib.load(cache/(subset['subset_id']+'_'+n+'.joblib'))
            radius=r.SPEC['models'][n]['source_residual_interval']['half_width_mmbtu_per_mwh']
            for y in ends:
                rows=joined[y,n]
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always');values=model.predict(rows)
                for w in caught:
                    if type(w.message) is not RuntimeWarning or str(w.message) not in allowed:raise RuntimeError('Unadjudicated prediction warning: '+str(w.message))
                    counts[str(w.message)]+=1
                transformed=model.named_steps['features'].transform(rows)
                scaled=model.named_steps['scale'].transform(transformed);ridge=model.named_steps['ridge']
                independent=np.sum(scaled*ridge.coef_,axis=1)+ridge.intercept_
                delta=float(np.max(np.abs(values-independent)));maximum=max(maximum,delta)
                if not np.isfinite(scaled).all() or not np.isfinite(values).all() or delta>1e-12:
                    raise ValueError('Cached prediction disagrees with independent coefficient contraction')
                if not np.array_equal(np.abs(values-rows.gross_heat_rate_mmbtu_per_mwh.to_numpy())<=radius,
                                      np.abs(independent-rows.gross_heat_rate_mmbtu_per_mwh.to_numpy())<=radius):
                    raise ValueError('Independent contraction changed coverage membership')
                fm=s.facility_metrics(rows,values,radius,subset,y,n);metrics.append(fm)
                full.append(s.add_subset_identity(s.summarize_facilities(fm),subset,y,n))
                common.append(s.add_subset_identity(s.summarize_facilities(fm[fm.facility_id.isin(panel)]),subset,y,n))
    for name,f in a.assemble(s,metrics,full,common).items():csvsave(r.RESULTS/name,f)
    diagnostics.update(warning_counts=dict(counts),cached_prediction_independent_max_difference=maximum,
                       warning_interpretation='Recorded matmul warnings; independently checked finite transformed inputs, coefficient contraction and exact covered-day membership; no claim of a fully identified platform cause')
    return r.RESULTS,diagnostics

def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--input-manifest',type=Path);g.add_argument('--source-workspace',type=Path)
    p.add_argument('--primary-only',action='store_true');p.add_argument('--output-dir',type=Path)
    args=p.parse_args();names=required_artifacts(args.primary_only);req,inputs=resolve_inputs(args)
    if req['paper']!='combined_cycle_heat_rate':raise ValueError('Cross-paper replay configuration')
    import importlib.metadata
    actual={k:importlib.metadata.version(k) for k in req['software'] if k!='python'}
    if any(actual[k]!=v for k,v in req['software'].items() if k!='python'):
        raise RuntimeError('Use the locked software versions: '+json.dumps({'expected':req['software'],'actual':actual}))
    if '.'.join(map(str,sys.version_info[:3]))!=req['software']['python']:
        raise RuntimeError('Use the locked Python version: '+req['software']['python'])
    if args.output_dir and (args.output_dir.exists() and any(args.output_dir.iterdir())):raise ValueError('Output directory must be empty')
    if args.output_dir:
        target=args.output_dir.resolve()
        if ROOT==target or ROOT in target.parents or any(p==target or target in p.parents for p in inputs.values()):
            raise ValueError('Output must be separate from the package and input carriers')
    before={str(p):digest(p) for p in inputs.values()}
    with tempfile.TemporaryDirectory(prefix='paper_portable_') as name:
        temp=Path(name)
        out,diagnostics=heat_rate(temp,inputs,args.primary_only)
        comparisons=compare_outputs(out,names)
        if before!={str(p):digest(p) for p in inputs.values()}:raise ValueError('Input changed during replay')
        report={'paper':req['paper'],'status':'PASS','comparisons':comparisons,'diagnostics':diagnostics,
                'inputs_read_only':True,'required_artifact_count':len(names)}
        if args.output_dir:
            args.output_dir.mkdir(parents=True,exist_ok=True)
            for n in names:shutil.copyfile(out/Path(n).name,args.output_dir/Path(n).name)
            save(args.output_dir/'REPLAY_RESULT.json',report)
        print(json.dumps(report,indent=2))

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(json.dumps({'status':'NOT_EXECUTED' if isinstance(exc,FileNotFoundError) else 'FAIL','reason':str(exc)},indent=2),file=sys.stderr)
        raise SystemExit(2)

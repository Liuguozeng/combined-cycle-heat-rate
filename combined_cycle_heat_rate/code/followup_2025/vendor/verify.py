import argparse

import json

import sys

from pathlib import Path

import numpy as np

import pandas as pd

def identities(frame,keys,expected,label):
    if frame[keys].isna().any().any() or frame.duplicated(keys).any() or set(map(tuple,frame[keys].to_numpy()))!=set(expected):
        raise ValueError(label+' identity set mismatch')

def independent_support(frame):
    from scipy.spatial import cKDTree
    spec=json.loads((SUP/'model_spec/FIXED_RELATION_SPECIFICATION.json').read_text())['input_support']
    source=pd.read_csv(SUP/'reproduction/reference_inputs/source_daily/SIX_FACILITY_DAILY_ENDPOINT.csv.gz')
    cols=spec['features_in_order'];center=np.asarray(spec['mean']);scale=np.asarray(spec['scale'])
    x=frame[cols].to_numpy(float);z=(source[cols].to_numpy(float)-center)/scale
    distance=cKDTree(z).query((x-center)/scale,k=1)[0]
    return ((x>=np.asarray(spec['minimum']))&(x<=np.asarray(spec['maximum']))).all(axis=1)&(distance<=spec['nn_distance_threshold']+1e-12)

def check(path):
    required=['FACILITY_MODEL_METRICS.csv','MATCHED_PREDICTIONS.csv.gz','QUALIFIED_DAILY.csv.gz',
              'ANNUAL_SUMMARIES.csv','COMMON_FACILITY_PAIRS.csv','PERSISTENCE.csv',
              'SUPPORTED_AND_COVERED_FACILITIES.csv','SUPPORTED_AND_COVERED_SUMMARY.csv',
              'WEATHER_JOINED_DAILY.csv.gz','WEATHER_SUPPORT_KEYS.csv.gz']
    for f in required:
        if not (path/f).is_file():raise ValueError('Missing required result: '+f)
    m=pd.read_csv(path/required[0]);p=pd.read_csv(path/required[1]);d=pd.read_csv(path/required[2])
    for frame,keys in [(m,['facility_id','year','model']),(p,['facility_id','year','model','date']),(d,['facility_id','year','date'])]:
        if frame[keys].isna().any().any() or frame.duplicated(keys).any():raise ValueError('Invalid or duplicate identity')
        if not frame.year.isin([2024,2025]).all():raise ValueError('Unexpected evaluation year')
        if 'date' in frame and not (pd.to_datetime(frame.date,errors='raise').dt.year==frame.year).all():raise ValueError('Date/year mismatch')
    if set(m.model)!={'M1','M2'} or not m.groupby(['facility_id','year']).size().eq(2).all():raise ValueError('Missing model pair')
    keys=['facility_id','year','date']
    if not set(zip(d.facility_id,d.year))==set(zip(m.facility_id,m.year)):raise ValueError('Endpoint/metric facility-year mismatch')
    expected_support=d.loc[independent_support(d),keys]
    expected_keys={(fid,year,date,model) for fid,year,date in expected_support.itertuples(index=False,name=None) for model in ['M1','M2']}
    identities(p,keys+['model'],expected_keys,'Prediction support')
    dates=pd.to_datetime(p.date)
    if not p.week.eq(dates.dt.to_period('W-SUN').astype(str)).all():raise ValueError('Prediction week/date mismatch')
    observed=p.merge(d[keys+['gross_heat_rate_mmbtu_per_mwh','operator_id']],on=keys,suffixes=('_p','_d'),validate='many_to_one')
    if not observed.operator_id_p.eq(observed.operator_id_d).all():raise ValueError('Prediction operator-year mismatch')
    if not np.allclose(observed.gross_heat_rate_mmbtu_per_mwh_p,observed.gross_heat_rate_mmbtu_per_mwh_d,atol=1e-12,rtol=1e-12):raise ValueError('Prediction observed endpoint mismatch')
    meta=d.groupby(['facility_id','year']).operator_id.agg(['first','nunique']).reset_index()
    mm=m.merge(meta,on=['facility_id','year'],validate='many_to_one')
    if not mm['nunique'].eq(1).all() or not mm.operator_id.eq(mm['first']).all():raise ValueError('Metric operator-year mismatch')
    joined=pd.read_csv(path/'WEATHER_JOINED_DAILY.csv.gz');support_keys=pd.read_csv(path/'WEATHER_SUPPORT_KEYS.csv.gz')
    if joined.duplicated(keys).any():raise ValueError('Duplicate weather-joined endpoint')
    identities(support_keys,keys,joined.loc[independent_support(joined),keys].itertuples(index=False,name=None),'Pre-eligibility weather support')
    groups=dict(tuple(p.groupby(['facility_id','year','model'])))
    if set(groups)!=set(zip(m.facility_id,m.year,m.model)):raise ValueError('Metric/prediction identities differ')
    errors=[];maxima={};counts=[]
    spec=json.loads((SUP/'model_spec/FIXED_RELATION_SPECIFICATION.json').read_text())
    for _,r in m.iterrows():
        g=groups[(r.facility_id,r.year,r.model)]
        err=g.prediction.to_numpy()-g.gross_heat_rate_mmbtu_per_mwh.to_numpy();radius=float(r.source_radius_mmbtu_per_mwh)
        frozen_radius=spec['models'][r.model]['source_residual_interval']['half_width_mmbtu_per_mwh']
        if not np.isclose(radius,frozen_radius,atol=1e-12,rtol=1e-12) or not np.allclose(g.source_radius_mmbtu_per_mwh,frozen_radius,atol=1e-12,rtol=1e-12):raise ValueError('Frozen interval radius mismatch')
        radius=frozen_radius
        if not np.isfinite(err).all():raise ValueError('Nonfinite predictions')
        covered=np.abs(err)<=radius
        if not np.array_equal(g.covered.to_numpy(),covered):raise ValueError('Covered-day membership differs')
        expected=dict(coverage_90=float(covered.mean()),mae_mmbtu_per_mwh=float(np.abs(err).mean()),
             signed_bias_mmbtu_per_mwh=float(err.mean()),mean_winkler_score=float((2*radius+20*np.maximum(np.abs(err)-radius,0)).mean()),
             interval_width_mmbtu_per_mwh=2*radius,matched_support_days=len(g))
        nd=int(((d.facility_id==r.facility_id)&(d.year==r.year)).sum())
        expected.update(daily_endpoint_days=nd,support_fraction=len(g)/nd)
        for col,val in expected.items():
            delta=abs(float(r[col])-val);maxima[col]=max(maxima.get(col,0),delta)
            if not np.isclose(r[col],val,atol=1e-12,rtol=1e-12):errors.append((int(r.facility_id),int(r.year),r.model,col,delta))
        category='undercoverage' if expected['coverage_90']<.85 else 'overcoverage' if expected['coverage_90']>.95 else 'nominal_band'
        if r.coverage_direction!=category:raise ValueError('Coverage class mismatch')
        # Independent week-cluster bootstrap reproduces the unchanged seed and indexing order.
        weeks=sorted(g.week.unique());cw=[];nw=[]
        for week in weeks:
            ix=g.week.eq(week).to_numpy();cw.append(int(covered[ix].sum()));nw.append(int(ix.sum()))
        rng=np.random.default_rng(20260827+int(r.facility_id)*10+(0 if r.model=='M1' else 1))
        draws=rng.integers(0,len(weeks),size=(5000,len(weeks)))
        ratios=np.asarray(cw)[draws].sum(axis=1)/np.asarray(nw)[draws].sum(axis=1)
        lo,hi=np.quantile(ratios,[.025,.975])
        if not np.allclose([lo,hi],[r.coverage_ci_low,r.coverage_ci_high],atol=1e-12,rtol=1e-12):raise ValueError('Week CI mismatch')
        counts.append(dict(facility_id=int(r.facility_id),year=int(r.year),model=r.model,days=nd,supported=len(g),covered=int(covered.sum()),weeks=len(weeks)))
    if errors:raise ValueError('Independent reductions differ: '+str(errors[:5]))
    sums=pd.read_csv(path/'ANNUAL_SUMMARIES.csv')
    identities(sums,['year','model','scope'],[(y,n,s) for y in [2024,2025] for n in ['M1','M2'] for s in ['all_eligible_followup','common_2024_2025']],'Annual summary')
    joint=pd.read_csv(path/'SUPPORTED_AND_COVERED_FACILITIES.csv')
    independent=pd.DataFrame(counts)
    j=joint.merge(independent,on=['facility_id','year','model'],how='outer',indicator=True,validate='one_to_one')
    if not j._merge.eq('both').all():raise ValueError('Joint-fraction identities differ')
    if not (j.supported_and_covered_days==j.covered).all():raise ValueError('Joint covered numerator differs')
    if not (j.daily_endpoint_days==j.days).all():raise ValueError('Joint denominator differs')
    if not np.allclose(j.supported_and_covered_fraction,j.covered/j.days,atol=1e-12,rtol=1e-12):raise ValueError('Joint fraction differs')
    pairids=set(m.loc[m.year.eq(2024),'facility_id'])&set(m.loc[m.year.eq(2025),'facility_id'])
    for _,r in sums.iterrows():
        if r.scope not in {'common_2024_2025','all_eligible_followup'}:raise ValueError('Unknown summary scope')
        g=m.loc[m.year.eq(r.year)&m.model.eq(r.model)]
        if r.scope=='common_2024_2025':g=g.loc[g.facility_id.isin(pairids)]
        if int(r.facilities)!=len(g):raise ValueError('Annual denominator mismatch')
        if not np.isclose(r.facility_equal_mean_coverage,g.coverage_90.mean(),atol=1e-12,rtol=1e-12):raise ValueError('Annual coverage mismatch')
        for cat,col in [('undercoverage','facilities_undercoverage'),('nominal_band','facilities_nominal'),('overcoverage','facilities_overcoverage')]:
            if int(r[col])!=int(g.coverage_direction.eq(cat).sum()):raise ValueError('Annual category mismatch')
        clusters=g.groupby('operator_id').coverage_90.agg(['sum','size','mean'])
        rng=np.random.default_rng(20260827+(0 if r.model=='M1' else 1))
        ix=rng.integers(0,len(clusters),size=(5000,len(clusters)))
        values=clusters['sum'].to_numpy()[ix].sum(axis=1)/clusters['size'].to_numpy()[ix].sum(axis=1)
        if not np.allclose(np.quantile(values,[.025,.975]),[r.operator_cluster_ci_low,r.operator_cluster_ci_high],atol=1e-12,rtol=1e-12):
            raise ValueError('Annual operator-cluster CI mismatch')
        if not np.isclose(clusters['mean'].mean(),r.operator_equal_mean_coverage,atol=1e-12,rtol=1e-12):raise ValueError('Operator-equal coverage mismatch')
    pairs=pd.read_csv(path/'COMMON_FACILITY_PAIRS.csv')
    persistence=pd.read_csv(path/'PERSISTENCE.csv')
    identities(pairs,['facility_id','model'],[(f,n) for f in pairids for n in ['M1','M2']],'Paired table')
    identities(persistence,['model'],[('M1',),('M2',)],'Persistence')
    expected=m.loc[m.year.eq(2024)&m.facility_id.isin(pairids)].merge(m.loc[m.year.eq(2025)&m.facility_id.isin(pairids)],on=['facility_id','model'],suffixes=('_2024','_2025'),validate='one_to_one')
    try:pd.testing.assert_frame_equal(pairs.sort_values(['facility_id','model']).reset_index(drop=True),expected.sort_values(['facility_id','model']).reset_index(drop=True),check_exact=False,atol=1e-12,rtol=1e-12)
    except AssertionError as exc:raise ValueError('Paired values differ from annual metrics') from exc
    for _,r in persistence.iterrows():
        a=m.loc[m.year.eq(2024)&m.model.eq(r.model)&m.facility_id.isin(pairids)].set_index('facility_id').sort_index()
        b=m.loc[m.year.eq(2025)&m.model.eq(r.model)&m.facility_id.isin(pairids)].set_index('facility_id').sort_index()
        for col,result in [('signed_bias_mmbtu_per_mwh','signed_bias_spearman'),('coverage_90','coverage_spearman')]:
            rho=a[col].rank().corr(b[col].rank())
            if not np.isclose(r[result],rho,atol=1e-12,rtol=1e-12):raise ValueError('Rank persistence mismatch')
        if int(r.common_facilities)!=len(pairids) or int(r.same_coverage_category)!=int((a.coverage_direction==b.coverage_direction).sum()):raise ValueError('Persistence denominator/category mismatch')
    jointsum=pd.read_csv(path/'SUPPORTED_AND_COVERED_SUMMARY.csv')
    identities(jointsum,['year','model'],[(y,n) for y in [2024,2025] for n in ['M1','M2']],'Joint summary')
    for _,r in jointsum.iterrows():
        g=joint.loc[joint.year.eq(r.year)&joint.model.eq(r.model)]
        op=g.groupby('operator_id').supported_and_covered_fraction.mean().to_numpy()
        rng=np.random.default_rng(20260831+5000+int(r.year)+(0 if r.model=='M1' else 100))
        draws=rng.integers(0,len(op),size=(5000,len(op)));bounds=np.quantile(op[draws].mean(axis=1),[.025,.975])
        expected=[g.supported_and_covered_fraction.mean(),op.mean(),*bounds]
        actual=[r.facility_equal_supported_and_covered,r.operator_equal_supported_and_covered,r.operator_cluster_ci95_lower,r.operator_cluster_ci95_upper]
        if not np.allclose(expected,actual,atol=1e-12,rtol=1e-12) or r.facilities!=len(g) or r.operator_identifiers!=len(op):raise ValueError('Joint summary values/denominators mismatch')
    return dict(status='PASS',facility_model_rows=len(m),matched_prediction_rows=len(p),daily_rows=len(d),
        common_facilities=len(pairids),maximum_independent_differences=maxima,checks='identity; daily denominator; support; signed error; MAE; coverage; category; Winkler; week CI; annual summary') ,pd.DataFrame(counts)

import importlib.util

import json

import sys

from pathlib import Path

import numpy as np

import pandas as pd

def metadata(row):
    return {'nameplate_mw':float(row.qualifying_nameplate_mw), 'name':row.facility_name,
            'operator_id':int(row.operator_id), 'cooling_type':'not_used_by_m1_m2',
            'station_distance_km':float(row.selected_station_distance_km)}

def compare(a, b, keys, numeric, tolerance):
    unique_keys(a, keys); unique_keys(b, keys)
    j = a.merge(b, on=keys, how='outer', suffixes=('_a','_b'), indicator=True, validate='one_to_one')
    same = bool(j._merge.eq('both').all())
    differences = {c:float((j[c+'_a']-j[c+'_b']).abs().max()) for c in numeric}
    passed = same and all(np.isfinite(x) and x <= tolerance for x in differences.values())
    return {'passed':passed, 'identities_equal':same, 'rows_a':len(a), 'rows_b':len(b),
            'maximum_differences':differences, 'tolerance':tolerance}

def original_bridge():
    base, flags, replay, national, fitted, radii, source = modules()
    cohort = pd.read_csv(Q/'FIXED_FOLLOWUP_135.csv', dtype={'selected_station_id':str})
    raw_manifest = pd.read_csv(SUP/'source_provenance/CAMPD_2024_SOURCE_MANIFEST.csv').set_index('facility_id')
    weather_manifest = pd.read_csv(SUP/'source_provenance/NOAA_2024_SOURCE_MANIFEST.csv').set_index('facility_id')
    daily_parts=[]; pre_parts=[]; met_parts=[]; pred_parts=[]; ledger=[]; hand=[]
    for _, row in cohort.iterrows():
        fid=int(row.facility_id)
        raw_path=ARCH/'data/raw/campd_hourly_2024'/raw_manifest.loc[fid,'carrier_id']
        weather_path=ARCH/'data/raw/noaa_gsod_2024'/weather_manifest.loc[fid,'weather_carrier_id']
        for path, expected in [(raw_path,raw_manifest.loc[fid,'carrier_sha256']),
                                (weather_path,weather_manifest.loc[fid,'weather_carrier_sha256'])]:
            if not path.is_file() or sha(path)!=expected:
                raise ValueError(f'Historical byte identity not established: {path}')
        raw=pd.read_csv(raw_path,dtype={'Unit ID':str},low_memory=False)
        tagged, eligible=flags.eligible_rows(raw,normalized=True)
        r=tagged.loc[tagged['Unit Type'].fillna('').str.strip().str.casefold().eq('combined cycle')].copy()
        r['Heat Input Measure Indicator']=r.normalized_heat_input_flag
        h=base.aggregate_facility_hours(r,float(row.qualifying_nameplate_mw))
        weather=base.load_weather(weather_path)
        unique_keys(weather,['date'])
        calendar=pd.DataFrame({'date':pd.date_range('2024-01-01','2024-12-31'), 'ambient_temperature_c':np.nan})
        pre=base.aggregate_facility_days(h,calendar,fid,metadata(row))
        pre=pre.drop(columns=['ambient_temperature_c','temp_x_recirculating','temp_x_once_through','temp_x_dry_cooling'])
        d=base.aggregate_facility_days(h,weather,fid,metadata(row))
        if len(d)<150:
            raise ValueError(f'Frozen annual eligibility changed: {fid}')
        mets,preds=national.evaluate_facility(d,source,fitted,radii,row)
        daily_parts.append(d); pre_parts.append(pre); met_parts.extend(mets); pred_parts.extend(preds)
        ledger.append({'facility_id':fid,'raw_sha256':sha(raw_path),'weather_sha256':sha(weather_path),
                       'raw_unit_hours':len(raw),'included_unit_hours':int(eligible.sum()),
                       'eligible_hours':len(h),'preweather_days':len(pre),'gsod_days':len(d)})
        # Independent single-day sums include operating time only in gross energy.
        # Prefer a retained partial-hour day when available; never alter membership.
        inc=tagged.loc[eligible].copy()
        partial=inc.loc[inc['Operating Time'].lt(1),'Date'].astype(str)
        chosen=next((x for x in partial if pd.Timestamp(x) in set(d.date)), d.date.iloc[0].strftime('%Y-%m-%d'))
        hours=set(h.loc[h.date.eq(pd.Timestamp(chosen)),'hour'])
        sample=inc.loc[pd.to_datetime(inc.Date).eq(pd.Timestamp(chosen)) & inc.Hour.isin(hours)]
        gross=float(sum(float(x['Gross Load (MW)'])*float(x['Operating Time']) for _,x in sample.iterrows()))
        heat=float(sum(float(x['Heat Input (mmBtu)']) for _,x in sample.iterrows()))
        actual=d.loc[d.date.eq(pd.Timestamp(chosen))].iloc[0]
        hand.append({'facility_id':fid,'date':chosen,'partial_hour_present':bool(sample['Operating Time'].lt(1).any()),
                     'gross_mwh_independent':gross,'heat_mmbtu_independent':heat,'unit_rows':len(sample),
                     'gross_delta':gross-actual.gross_mwh,'heat_delta':heat-actual.heat_input_mmbtu,
                     'heat_rate_delta':heat/gross-actual.gross_heat_rate_mmbtu_per_mwh})
        if len(ledger)%20==0: print(f'REBUILT {len(ledger)}/135',flush=True)
    daily=pd.concat(daily_parts,ignore_index=True); pre=pd.concat(pre_parts,ignore_index=True)
    metrics=pd.DataFrame(met_parts).assign(year=2024); pred=pd.concat(pred_parts,ignore_index=True)
    for name,frame in [('PREWEATHER_2024',pre),('GSOD_2024_DAILY',daily),('GSOD_2024_METRICS',metrics),('GSOD_2024_PREDICTIONS',pred),('RAW_IDENTITY',pd.DataFrame(ledger)),('INDEPENDENT_DAY_SUMS',pd.DataFrame(hand))]:
        frame.to_csv(OUT/(name+'.csv.gz'),index=False,compression={'method':'gzip','mtime':0})
    correct=EXP/'heat_rate_endpoint/measurement_processing/corrected_data'
    olddaily=pd.read_csv(correct/'2024_CORRECTED_DAILY_ENDPOINT.csv',parse_dates=['date'])
    oldpred=pd.read_csv(correct/'2024_CORRECTED_MATCHED_PREDICTIONS.csv',parse_dates=['date'])
    oldmetrics=pd.read_csv(SUP/'results/FACILITY_MODEL_METRICS.csv').query('year==2024')
    checks={
      'daily':compare(olddaily,daily,['facility_id','date'],['gross_mwh','heat_input_mmbtu','qualifying_hours','gross_heat_rate_mmbtu_per_mwh','facility_load_fraction','ambient_temperature_c'],1e-10),
      'predictions':compare(oldpred,pred,['facility_id','date','model'],['prediction','gross_heat_rate_mmbtu_per_mwh'],1e-12),
      'metrics':compare(oldmetrics,metrics,['facility_id','year','model'],['daily_endpoint_days','matched_support_days','coverage_90','coverage_ci_low','coverage_ci_high','mae_mmbtu_per_mwh','signed_bias_mmbtu_per_mwh','mean_winkler_score'],1e-12),
      'new_model_fits':0, 'weather_change_evaluated':False}
    checks['passed']=all(checks[k]['passed'] for k in ('daily','predictions','metrics'))
    (OUT/'GSOD_REPRODUCTION.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks,indent=2))
    if not checks['passed']: raise RuntimeError('Original branch must reproduce before SSOD comparison')

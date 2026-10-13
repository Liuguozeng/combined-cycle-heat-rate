import argparse

import importlib.util

import json

import sys

from datetime import datetime, timezone

from pathlib import Path

import numpy as np

import pandas as pd

def save(name, data):
    OUT.mkdir(exist_ok=True)
    p = OUT/name
    if isinstance(data, pd.DataFrame):
        data.to_csv(p,index=False,compression={'method':'gzip','mtime':0} if name.endswith('.gz') else None)
    else:
        p.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')

def validate_daily_weather(weather, year):
    w = weather.copy()
    w['date'] = pd.to_datetime(w.date,errors='raise')
    unique_keys(w,['date'])
    if not w.date.dt.year.eq(year).all(): raise ValueError('Weather year mismatch')
    w['temp_c'] = pd.to_numeric(w.temp_c,errors='raise')
    return w.loc[w.temp_c.between(-60,60)].copy()

def join_weather(pre, weather, year):
    if not pd.to_datetime(pre.date).dt.year.eq(year).all(): raise ValueError('Endpoint year mismatch')
    unique_keys(pre,['facility_id','date'])
    w = validate_daily_weather(weather,year)[['date','temp_c']].rename(columns={'temp_c':'ambient_temperature_c'})
    d = pre.merge(w,on='date',how='inner',validate='one_to_one')
    for name in ['recirculating','once_through','dry_cooling']:
        d['temp_x_'+name] = 0.
    return d

def independent_day(rows):
    gross = sum(float(p)*float(t) for p,t in zip(rows['Gross Load (MW)'],rows['Operating Time']))
    heat = sum(float(q) for q in rows['Heat Input (mmBtu)'])
    return gross,heat,heat/gross

def weather_date_comparison(old, joined, statuses):
    cols=['facility_id','date','ambient_temperature_c']
    comparison=old[cols].merge(joined[cols],on=['facility_id','date'],how='outer',
        suffixes=('_gsod','_new'),indicator=True,validate='one_to_one')
    comparison['new_analysis_status']=comparison.facility_id.map(statuses)
    return comparison

def cohort():
    c = pd.read_csv(Q/'FIXED_FOLLOWUP_135.csv',dtype={'selected_station_id':str})
    e = pd.read_csv(Q/'EIA_EPA_FINAL_2025_QUALIFICATION.csv')
    unique_keys(c,['facility_id']);unique_keys(e,['facility_id'])
    if set(c.facility_id)!=set(e.facility_id):raise ValueError('Follow-up identity mismatch')
    source=pd.read_csv(SUP/'results/DISCOVERY_FACILITY_CONTEXT.csv')
    overlap=e.loc[e.facility_id.isin(source.facility_id)|e.operator_id_2025.isin(source.operator_id)]
    if len(overlap):raise ValueError('Follow-up source-facility/operator overlap requires explicit adjudication')
    return c,e

def weather_adapter(folder):
    "Translate the supplied weather input contract without changing values."
    status=json.loads((folder/'STATUS.json').read_text())
    if status.get('partial',True) or status.get('pending_hourly_downloads',0):
        raise ValueError('Weather acquisition/scan is not finalized')
    daily=pd.read_csv(folder/'tables/WEATHER_DAILY_CORRECTED.csv.gz',parse_dates=['date'])
    mapping=pd.read_csv(folder/'tables/FACILITY_STATION_MAPPING.csv',dtype={'gsod_id':str})
    unique_keys(daily,['ghcn_id','date']);unique_keys(mapping,['facility_id','year'])
    dc=[];parts=[]
    for (sid,year),g in mapping.groupby(['gsod_id','year']):
        if g.ghcn_id.nunique()!=1 or g.mapping_status.nunique()!=1:
            raise ValueError('Inconsistent shared station mapping')
        gid=g.ghcn_id.iloc[0];approved=g.mapping_status.iloc[0]=='approved'
        wx=daily.loc[daily.ghcn_id.eq(gid)&daily.year.eq(year)].copy()
        usable=wx.weather_usable.map(lambda x:str(x).lower()=='true')
        good=wx.loc[usable].copy()
        approved=approved and not good.empty
        dc.append(dict(gsod_id=sid,ghcn_id=gid,year=int(year),approved_for_evaluation=approved,
            reason=g.mapping_status.iloc[0],usable_days=len(good)))
        if approved:
            good['gsod_id']=sid;good['weather_method']=good.daily_method
            good['temperature_observations']=good.selected_hour_count.where(good.daily_method.eq('project_hourly_reconstruction'),good.ssod_count)
            parts.append(good)
    save('WEATHER_EVALUATION_DECISIONS.csv',pd.DataFrame(dc))
    if not parts:raise ValueError('No usable weather')
    save('WEATHER_EVALUATION_DAILY.csv.gz',pd.concat(parts,ignore_index=True))

def prepare():
    base,flags,*_ = modules()
    c,e = cohort();e=e.set_index('facility_id')
    parts=[];ledger=[];hand=[];sources=[]
    for _,row in c.iterrows():
        fid=int(row.facility_id);info=e.loc[fid]
        entry={'facility_id':fid,'year':2025,'station':row.selected_station_id}
        if not bool(info.annual_screen_pass):
            ledger.append({**entry,'status':'annual_boundary_screen_failed','gross_net_ratio':info.gross_net_ratio});continue
        p=ARCH/f'data/raw/campd_hourly_2025/facility_{fid}_2025.csv.gz'
        if not p.exists():p=HERE/f'raw_epa/facility_{fid}_2025.csv.gz'
        if not p.exists():
            ledger.append({**entry,'status':'hourly_input_missing'});continue
        digest=sha(p)
        raw=pd.read_csv(p,dtype={'Unit ID':str},low_memory=False)
        if set(raw['Facility ID'])!={fid}:raise ValueError('Wrong facility: '+str(fid))
        if not pd.to_datetime(raw.Date,errors='raise').dt.year.eq(2025).all():raise ValueError('Wrong EPA year')
        tagged,eligible=flags.eligible_rows(raw,normalized=True)
        selected=tagged.loc[tagged['Unit Type'].fillna('').str.strip().str.casefold().eq('combined cycle')].copy()
        selected['Heat Input Measure Indicator']=selected.normalized_heat_input_flag
        capacity=float(info.qualifying_nameplate_mw_2025)
        row=row.copy();row['qualifying_nameplate_mw']=capacity;row['operator_id']=int(info.operator_id_2025)
        h=base.aggregate_facility_hours(selected,capacity)
        calendar=pd.DataFrame({'date':pd.date_range('2025-01-01','2025-12-31'),'ambient_temperature_c':np.nan})
        d=base.aggregate_facility_days(h,calendar,fid,metadata(row))
        if not d.empty:
            d=d.drop(columns=['ambient_temperature_c','temp_x_recirculating','temp_x_once_through','temp_x_dry_cooling'])
            parts.append(d)
            inc=tagged.loc[eligible].copy();inc['parsed_date']=pd.to_datetime(inc.Date)
            partial=inc.loc[inc['Operating Time'].lt(1),'parsed_date']
            date=next((x for x in partial if x in set(d.date)),d.date.iloc[0])
            hours=set(h.loc[h.date.eq(date),'hour'])
            sample=inc.loc[inc.parsed_date.eq(date)&inc.Hour.isin(hours)]
            g,q,r=independent_day(sample);actual=d.loc[d.date.eq(date)].iloc[0]
            hand.append(dict(facility_id=fid,date=str(date.date()),partial_hour=bool(sample['Operating Time'].lt(1).any()),
                gross_delta=g-actual.gross_mwh,heat_delta=q-actual.heat_input_mmbtu,
                heat_rate_delta=r-actual.gross_heat_rate_mmbtu_per_mwh))
        if sha(p)!=digest:raise ValueError('Raw input changed while reading')
        sources.append(dict(facility_id=fid,year=2025,path=str(p),sha256=digest,rows=len(raw),role='raw_unit_hours'))
        ledger.append({**entry,'status':'preweather_built','raw_rows':len(raw),'eligible_unit_rows':int(eligible.sum()),
             'eligible_hours':len(h),'preweather_days':len(d),'nameplate_mw':capacity})
        if len(ledger)%20==0:print('PREWEATHER',len(ledger),flush=True)
    if not parts:raise ValueError('No eligible inputs')
    save('PREWEATHER_2025.csv.gz',pd.concat(parts,ignore_index=True))
    save('EPA_QUALIFICATION_2025.csv',pd.DataFrame(ledger))
    save('INDEPENDENT_DAY_SUMS_2025.csv',pd.DataFrame(hand))
    save('RAW_INPUT_MANIFEST.json',sources)
    deltas=pd.DataFrame(hand)[['gross_delta','heat_delta','heat_rate_delta']].abs().max()
    if deltas.max()>1e-7:raise ValueError('Independent sums differ: '+str(deltas))
    save('PREPARE_STATUS.json',dict(status='PASS',facilities=len(parts),days=sum(map(len,parts)),independent_max_differences=deltas.to_dict(),new_fits=0))

def evaluate(weather_path, decision_path):
    prior=json.loads((HERE/'BASELINE_IDENTITY.json').read_text())
    for item in prior['files']:
        if '/model_spec/' in item['path'] or '/reproduction/code/' in item['path']:
            if sha(Path(prior['candidate_root'])/item['path'])!=item['actual']:
                raise ValueError('Frozen implementation or model changed')
    gate=json.loads((HERE.parent/'bridge/GSOD_REPRODUCTION.json').read_text())
    if not gate['passed']:raise ValueError('Original reproduction prerequisite not met')
    base,flags,replay,national,fitted,radii,source=modules()
    c,e=cohort();e=e.set_index('facility_id')
    w=pd.read_csv(weather_path,dtype={'gsod_id':str},parse_dates=['date'])
    decisions=pd.read_csv(decision_path,dtype={'gsod_id':str})
    unique_keys(w,['gsod_id','year','date']);unique_keys(decisions,['gsod_id','year'])
    needed={(str(s),y) for s in c.selected_station_id for y in [2024,2025]}
    if needed-set(zip(decisions.gsod_id,decisions.year)):raise ValueError('Incomplete weather decisions')
    approved=decisions.approved_for_evaluation.map(lambda x: str(x).lower()=='true')
    allowed=set(zip(decisions.loc[approved,'gsod_id'],decisions.loc[approved,'year']))
    allmetrics=[];allpred=[];alldays=[];ledger=[];monthly=[];weather_joined=[];weather_support=[]
    for year in (2024,2025):
        pp=HERE.parent/'bridge/PREWEATHER_2024.csv.gz' if year==2024 else OUT/'PREWEATHER_2025.csv.gz'
        pre=pd.read_csv(pp,parse_dates=['date'])
        for _,row in c.iterrows():
            fid=int(row.facility_id);sid=str(row.selected_station_id)
            rec={'facility_id':fid,'year':year,'station':sid}
            if year==2025:
                info=e.loc[fid]
                if not bool(info.annual_screen_pass):ledger.append({**rec,'status':'annual_boundary_screen_failed'});continue
                row=row.copy();row['operator_id']=int(info.operator_id_2025);row['operator_name']=info.operator_name_2025
                row['state']=info.state_2025;row['facility_name']=info.facility_name_2025
            p=pre.loc[pre.facility_id.eq(fid)].copy()
            if p.empty:ledger.append({**rec,'status':'no_eligible_preweather_input'});continue
            if (sid,year) not in allowed:ledger.append({**rec,'status':'weather_not_qualified','preweather_days':len(p)});continue
            wx=w.loc[w.gsod_id.eq(sid)&w.year.eq(year)]
            d=join_weather(p,wx,year)
            weather_joined.append(d.assign(year=year))
            if len(d):
                support_mask,_=replay.fixed_support(source,d)
                weather_support.append(d.loc[support_mask,['facility_id','date']].assign(year=year))
            monthly.extend(dict(facility_id=fid,year=year,month=m,eligible_days=int(d.date.dt.month.eq(m).sum())) for m in range(1,13))
            if len(d)<150:ledger.append({**rec,'status':'below_150_eligible_days','eligible_days':len(d),'preweather_days':len(p)});continue
            mets,preds=national.evaluate_facility(d,source,fitted,radii,row)
            if len(mets)!=2:ledger.append({**rec,'status':'below_3_supported_days','eligible_days':len(d)});continue
            allmetrics.extend({**m,'year':year} for m in mets)
            allpred.extend(x.assign(year=year) for x in preds);alldays.append(d.assign(year=year))
            ledger.append({**rec,'status':'evaluated','preweather_days':len(p),'eligible_days':len(d),
                'matched_support_days':mets[0]['matched_support_days'],'weather_method':'|'.join(sorted(wx.weather_method.unique()))})
        print('EVALUATED',year,sum(x['year']==year for x in allmetrics),flush=True)
    if not allmetrics:raise ValueError('No valid evaluations')
    metrics=pd.DataFrame(allmetrics);pred=pd.concat(allpred,ignore_index=True);days=pd.concat(alldays,ignore_index=True)
    unique_keys(metrics,['facility_id','year','model']);unique_keys(pred,['facility_id','year','date','model'])
    save('FACILITY_MODEL_METRICS.csv',metrics);save('MATCHED_PREDICTIONS.csv.gz',pred);save('QUALIFIED_DAILY.csv.gz',days)
    joined=pd.concat(weather_joined,ignore_index=True);support_keys=pd.concat(weather_support,ignore_index=True)
    save('WEATHER_JOINED_DAILY.csv.gz',joined);save('WEATHER_SUPPORT_KEYS.csv.gz',support_keys)
    save('FACILITY_QUALIFICATION.csv',pd.DataFrame(ledger));save('SEASONAL_COVERAGE.csv',pd.DataFrame(monthly))
    summaries=[national.summarize_national(m).assign(year=y,scope='all_eligible_followup') for y,m in metrics.groupby('year')]
    ids=set(metrics.loc[metrics.year.eq(2024),'facility_id'])&set(metrics.loc[metrics.year.eq(2025),'facility_id'])
    paired=[];recurrence=[]
    for model in ['M1','M2']:
        a=metrics.loc[metrics.year.eq(2024)&metrics.model.eq(model)&metrics.facility_id.isin(ids)]
        b=metrics.loc[metrics.year.eq(2025)&metrics.model.eq(model)&metrics.facility_id.isin(ids)]
        j=a.merge(b,on=['facility_id','model'],suffixes=('_2024','_2025'),validate='one_to_one')
        paired.append(j)
        recurrence.append(dict(model=model,common_facilities=len(j),
            signed_bias_spearman=float(j.signed_bias_mmbtu_per_mwh_2024.corr(j.signed_bias_mmbtu_per_mwh_2025,method='spearman')),
            coverage_spearman=float(j.coverage_90_2024.corr(j.coverage_90_2025,method='spearman')),
            same_coverage_category=int((j.coverage_direction_2024==j.coverage_direction_2025).sum())))
    for y,m in metrics.loc[metrics.facility_id.isin(ids)].groupby('year'):
        summaries.append(national.summarize_national(m).assign(year=y,scope='common_2024_2025'))
    save('ANNUAL_SUMMARIES.csv',pd.concat(summaries,ignore_index=True))
    save('COMMON_FACILITY_PAIRS.csv',pd.concat(paired,ignore_index=True));save('PERSISTENCE.csv',pd.DataFrame(recurrence))
    panel=load_frozen('panel')
    joint=panel.supported_and_covered_facility_rows(metrics,pred)
    save('SUPPORTED_AND_COVERED_FACILITIES.csv',joint)
    save('SUPPORTED_AND_COVERED_SUMMARY.csv',panel.summarize_supported_and_covered(joint))
    old=pd.read_csv(HERE.parent/'bridge/GSOD_2024_METRICS.csv.gz')
    now=metrics.loc[metrics.year.eq(2024)]
    comparison=old.merge(now,on=['facility_id','model'],suffixes=('_gsod','_new'),validate='one_to_one')
    save('2024_METRIC_PRODUCT_COMPARISON.csv',comparison)
    oldpred=pd.read_csv(HERE.parent/'bridge/GSOD_2024_PREDICTIONS.csv.gz',parse_dates=['date'])
    pnew=pred.loc[pred.year.eq(2024)]
    cp=oldpred[['facility_id','model','date','prediction']].merge(pnew[['facility_id','model','date','prediction']],
        on=['facility_id','model','date'],suffixes=('_gsod','_new'),validate='one_to_one')
    cp['prediction_delta']=cp.prediction_new-cp.prediction_gsod
    save('2024_COMMON_SUPPORT_PREDICTION_DIFFERENCES.csv.gz',cp)
    all_new_support=pd.concat([support_keys.loc[support_keys.year.eq(2024),['facility_id','date']].assign(model=n) for n in ['M1','M2']],ignore_index=True)
    membership=oldpred[['facility_id','model','date']].merge(all_new_support,
        on=['facility_id','model','date'],how='outer',indicator=True,validate='one_to_one')
    states=pd.DataFrame(ledger).loc[lambda x:x.year.eq(2024)].set_index('facility_id').status
    membership['new_analysis_status']=membership.facility_id.map(states)
    save('2024_SUPPORT_MEMBERSHIP.csv.gz',membership)
    olddays=pd.read_csv(HERE.parent/'bridge/GSOD_2024_DAILY.csv.gz',parse_dates=['date'])
    wd=weather_date_comparison(olddays,joined.loc[joined.year.eq(2024)],states)
    save('2024_WEATHER_DATE_COMPARISON.csv.gz',wd)
    save('EVALUATION_STATUS.json',dict(status='COMPLETED_PENDING_INDEPENDENT_VERIFICATION',utc=datetime.now(timezone.utc).isoformat(),
        new_fits=0,weather_sha256=sha(weather_path),decisions_sha256=sha(decision_path),
        model_spec_sha256=sha(SUP/'model_spec/FIXED_RELATION_SPECIFICATION.json'),
        preweather_2025_sha256=sha(OUT/'PREWEATHER_2025.csv.gz'),
        facilities_by_year={str(y):int(m.facility_id.nunique()) for y,m in metrics.groupby('year')},
        common_facilities=len(ids),model_rows=len(metrics),supported_model_days=len(pred),qualified_days=len(days)))

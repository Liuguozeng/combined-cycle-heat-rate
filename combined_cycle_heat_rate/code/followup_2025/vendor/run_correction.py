from pathlib import Path

from datetime import datetime, timezone

import argparse

import csv

import hashlib

import fcntl

import json

import math

import sys

import pandas as pd

COLS=['STATION','DATE','temperature','temperature_Measurement_Code','temperature_Quality_Code',
      'temperature_Report_Type','temperature_Source_Code','temperature_Source_Station_ID',
      'REM','REM_Quality_Code','REM_Report_Type','REM_Source_Code','REM_Source_Station_ID']

def dump(path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False))

def csv_out(frame,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(path,index=False,compression={'method':'gzip','mtime':0} if str(path).endswith('.gz') else None)

def verify_receipt(path):
    p=Path(str(path)+'.receipt.json')
    if not p.exists(): raise ValueError('missing official download receipt: '+str(path))
    obj=json.loads(p.read_text())
    if obj.get('sha256')!=sha(path): raise ValueError('official input hash mismatch: '+str(path))
    if not obj.get('url','').startswith('https://www.ncei.noaa.gov/'):
        raise ValueError('unrecognized official hourly/daily source')

def find_hourly(st,year):
    for root in (BASE/'raw/hourly',RAW/'hourly'):
        p=root/f'GHCNh_{st}_{year}.psv'
        if p.exists(): return p
    return None

def acquisition_status(st,year):
    if find_hourly(st,year): return 'available'
    p=RAW/'hourly'/f'GHCNh_{st}_{year}.psv.receipt.json'
    if not p.exists(): return 'not_yet_downloaded_or_in_transfer'
    receipt=json.loads(p.read_text())
    codes=[a['http'] for a in receipt.get('attempts',[])]
    if '429' in codes: return 'rate_limit_stopped'
    if '404' in codes or '410' in codes: return 'annual_endpoint_not_found'
    if len(receipt.get('attempts',[]))<2: return 'not_yet_downloaded_or_in_transfer'
    return 'bounded_transfer_failed'

def read_ssod(st,year):
    for root in (BASE/'raw/ssod',RAW/'ssod'):
        for p in (root/str(year)/f'SSOD_{st}_{year}.csv',root/f'SSOD_{st}_por.csv'):
            if not p.exists(): continue
            verify_receipt(p)
            f=pd.read_csv(p,dtype={'STATION':str})
            f['ssod_raw_line']=range(2,len(f)+2)
            f=f.loc[pd.to_numeric(f.Year).eq(year)].copy()
            if f.empty: continue
            if not f.STATION.eq(st).all(): raise ValueError('SSOD station identity mismatch')
            f['date']=pd.to_datetime(f.DATE).dt.strftime('%Y-%m-%d')
            if f.date.duplicated().any(): raise ValueError('duplicate SSOD date')
            for col,values in inspect_ssod(f).items(): f[col]=values
            f['ssod_count']=pd.to_numeric(f.mean_temperature_Measurement_Code,errors='raise')
            if not f.ssod_count.isin([0]+list(range(4,25))).all(): raise ValueError('unknown SSOD count semantics')
            return f[['date','ssod_count','ssod_raw_line']+FIELDS].set_index('date'),p
    return pd.DataFrame(columns=['ssod_count','ssod_raw_line']+FIELDS).rename_axis('date'),None

def scan(st,year,icao):
    path=find_hourly(st,year)
    name=f'{st}_{year}'
    summary=CACHE/(name+'.json')
    codehash=sha(HERE/'weather.py')
    if path is None: return {'ghcn_id':st,'year':year,'scan_status':'not_scanned_missing_hourly','acquisition_status':acquisition_status(st,year)}
    verify_receipt(path)
    sourcehash=sha(path)
    if summary.exists():
        old=json.loads(summary.read_text())
        if old.get('hourly_sha256')==sourcehash and old.get('code_sha256')==codehash: return old
    raw=pd.read_csv(path,sep='|',usecols=COLS,dtype=str,keep_default_na=False)
    if not raw.STATION.eq(st).all(): raise ValueError('station mismatch')
    if not raw.DATE.str.startswith(str(year)+'-').all(): raise ValueError('year mismatch')
    pd.to_datetime(raw.DATE,errors='raise',utc=True)
    records=[check_observation(r,st,icao) for r in raw.to_dict('records')]
    checked=pd.DataFrame(records)
    # Stable original row number is a source locator, never a selection tiebreaker.
    checked['raw_line']=range(2,len(checked)+2)
    checked['source_path']=str(path.relative_to(BASE))
    checked['source_sha256']=sourcehash
    sel,dup=select_hours(checked)
    anomalies=checked.loc[checked.action.str.startswith(('corrected','unresolved'))].copy()
    csv_out(anomalies,CACHE/(name+'_corrections.csv.gz'))
    csv_out(checked.loc[checked.match_status.eq('precision_compatible_integer_field')],CACHE/(name+'_precision_compatible.csv.gz'))
    csv_out(sel,CACHE/(name+'_selected.csv.gz'))
    counts=checked.groupby(['source_code','quality_code','quality_status','report_type','action','match_status'],dropna=False).size().reset_index(name='rows')
    csv_out(counts,CACHE/(name+'_counts.csv'))
    # Ordinary QC rejects are counted separately; no claim that raw reports exist for every source.
    results={'ghcn_id':st,'year':year,'scan_status':'scanned','hourly_path':str(path.relative_to(BASE)),
      'hourly_sha256':sourcehash,'code_sha256':codehash,'input_rows':len(raw),'acquisition_status':'available',
      'finite_temperature_rows':int(checked.original_temp_c.notna().sum()),
      'matched_message_rows':int(checked.match_status.str.startswith('matched').sum()),
      'precision_compatible_unchanged_rows':int(checked.match_status.eq('precision_compatible_integer_field').sum()),
      'corrected_rows':int(checked.action.str.startswith('corrected').sum()),
      'unresolved_rows':int(checked.action.str.startswith('unresolved').sum()),
      'quality_rejected_rows':int(checked.action.eq('rejected_quality').sum()),
      'unknown_qc_rows':int((checked.quality_status.eq('unknown_source_quality_combination') & checked.original_temp_c.notna()).sum()),
      'selected_hours':len(sel),**dup,
      'requires_project_reconstruction':bool(len(anomalies) or dup['conflicting_identity_rows'] or dup['conflicting_time_rows']),
      'actions':{str(k):int(v) for k,v in checked.action.value_counts().items()},
      'scan_utc':datetime.now(timezone.utc).isoformat()}
    dump(summary,results)
    return results

def mapping_decision(r,year):
    # Existing identifier-based rulings, not model outcomes, establish all ordinary identities.
    if r['identity_decision']!='unresolved_coordinate_conflict':
        return True,r['identity_decision'],'qualification/STATION_IDENTITY_ADJUDICATION.csv'
    p=HERE/'MAPPING_ADJUDICATION.json'
    if p.exists():
        m=json.loads(p.read_text()).get(str(r['gsod_id']))
        if m and m.get('approved') and year in m['years']:
            return True,m['basis'],m['locator']
    return False,'historical physical identity/effective period not yet resolved','raw/metadata/HOMR_*_search.json'

def finalization_allowed(summaries, pending, requested):
    if not requested:
        return False
    keys={(s['ghcn_id'],s['year']) for s in summaries}
    if len(summaries)!=246 or len(keys)!=246 or pending:
        raise ValueError('Cannot finalize an incomplete or duplicate station-year scope')
    terminal_missing={'annual_endpoint_not_found','bounded_transfer_failed','rate_limit_stopped'}
    for s in summaries:
        if s['scan_status']=='scanned':
            continue
        if (s['scan_status']=='not_scanned_missing_hourly' and
                s.get('acquisition_status') in terminal_missing):
            continue
        raise ValueError('Cannot finalize without a scan or individual missing-carrier decision: '+str(s))
    return True

def run(finalize=False):
    for d in (TABLES,CACHE,VALID): d.mkdir(parents=True,exist_ok=True)
    previous=json.loads((HERE/'STATUS.json').read_text()) if (HERE/'STATUS.json').exists() else {}
    dump(HERE/'STATUS.json',{**previous,'stage':'hourly_scan','partial':True,
        'explicit_finalization':False,'utc':datetime.now(timezone.utc).isoformat()})
    scope=pd.read_csv(BASE/'qualification/STATION_IDENTITY_ADJUDICATION.csv',dtype={'gsod_id':str}).fillna('')
    assert len(scope)==123 and scope.ghcn_id.is_unique
    summaries=[]
    for i,r in scope.iterrows():
        for year in (2024,2025):
            try: result=scan(r.ghcn_id,year,r.isd_icao)
            except Exception as exc:
                result={'ghcn_id':r.ghcn_id,'year':year,'scan_status':'scan_failed', 'reason':repr(exc)}
            result['gsod_id']=r.gsod_id
            summaries.append(result)
            dump(HERE/'PROGRESS.json',{'stage':'hourly_scan','completed_station_years':len(summaries),
                'scanned':sum(s['scan_status']=='scanned' for s in summaries),
                'total_station_years':246,'partial':True,'last':result,'utc':datetime.now(timezone.utc).isoformat()})
        print(i+1, r.ghcn_id,[(s['year'],s['scan_status'],s.get('corrected_rows',0)) for s in summaries[-2:]],flush=True)
    all_daily=[]; decisions=[]; all_selected=[]; all_anomalies=[]; qualification=[]
    for _,r in scope.iterrows():
        pair=[s for s in summaries if s['ghcn_id']==r.ghcn_id]
        both_scanned=all(s['scan_status']=='scanned' for s in pair)
        any_project=any(s.get('requires_project_reconstruction',False) for s in pair)
        ssods={}
        ssod_errors={}
        for year in (2024,2025):
            try: ssods[year]=read_ssod(r.ghcn_id,year)
            except Exception as exc:
                ssod_errors[year]=repr(exc)
                ssods[year]=(pd.DataFrame(columns=['ssod_count','ssod_raw_line']+FIELDS).rename_axis('date'),None)
        missing_product=any(p is None for _,p in ssods.values())
        project=any_project or missing_product
        for s in pair:
            year=s['year']; st=r.ghcn_id; name=f'{st}_{year}'
            idx=pd.date_range(f'{year}-01-01',f'{year}-12-31').strftime('%Y-%m-%d')
            ssod,ssodpath=ssods[year]
            d=ssod.reindex(idx).rename_axis('date').copy()
            d['ssod_consistency_status']=d.ssod_consistency_status.fillna('no_daily_product_row')
            d['ssod_consistency_reason']=d.ssod_consistency_reason.fillna('')
            d['ssod_daily_inconsistent']=d.ssod_daily_inconsistent.eq(True)
            d['ssod_source_path']=str(ssodpath.relative_to(BASE)) if ssodpath else ''
            d['ssod_source_sha256']=sha(ssodpath) if ssodpath else ''
            d['ghcn_id']=st; d['gsod_id']=r.gsod_id; d['year']=year
            mappingok,basis,locator=mapping_decision(r,year)
            qc_known=all(s.get('unknown_qc_rows',0)==0 for s in pair)
            # A missing hourly carrier is an evidence gap, not a finding that official SSOD is wrong.
            attempted=all(acquisition_status(st,yr)!='not_yet_downloaded_or_in_transfer' for yr in (2024,2025))
            parser_failed=any(x['scan_status']=='scan_failed' for x in pair)
            daily_product_only=(not project and attempted and not parser_failed and not both_scanned)
            approved=(both_scanned or daily_product_only) and mappingok and qc_known
            reason='full_two_year_hourly_scan; '+basis
            if not both_scanned:
                if daily_product_only: reason='daily_product_checked_hourly_gap'
                elif not attempted: reason='hourly_acquisition_pending'
                elif parser_failed: reason='explicit_hourly_scan_failure'
                else: reason='project_reconstruction_missing_hourly_after_bounded_acquisition'
            if not mappingok: reason+='; mapping_not_approved'
            if not qc_known: reason+='; undocumented_source_QC_semantics'
            if ssod_errors: reason+='; explicit_SSOD_read_error_logged'
            if project and s['scan_status']=='scanned':
                selected=pd.read_csv(CACHE/(name+'_selected.csv.gz'),keep_default_na=False)
                daily=daily_from_hours(selected)
                for col in daily: d[col]=daily[col].reindex(idx)
                d['weather_usable']=d.weather_usable.eq(True)
                d['daily_method']='project_hourly_reconstruction'
                d['source_sha256']=s['hourly_sha256']
                if len(selected): all_selected.append(selected)
                else: reason+='; no_approved_routine_or_synoptic_hours'
            else:
                d['temp_c']=d.ssod_temp_c
                d['weather_usable']=d.ssod_temp_c.between(-60,60) & d.ssod_count.ge(4)
                d['selected_hour_count']=d.ssod_count
                d['utc_segments']=pd.NA # Official product's rule, not a recovered set of rows.
                d['corrected_selected_hours']=0
                d['selected_integer_precision_hours']=0
                d['daily_method']='official_ssod_unchanged' if ssodpath else 'unavailable'
                d['source_sha256']=sha(ssodpath) if ssodpath else ''
            d['unresolved_hours']=0
            d['directly_corrected_record_day']=False
            d['unresolved_record_day']=False
            anomaly_path=CACHE/(name+'_corrections.csv.gz')
            if s['scan_status']=='scanned' and anomaly_path.exists():
                anomalies=pd.read_csv(anomaly_path)
                if len(anomalies): all_anomalies.append(anomalies)
                if len(anomalies):
                    corrected_dates=set(anomalies.loc[anomalies.action.str.startswith('corrected'),'DATE'].str.slice(0,10))
                    unresolved_dates=set(anomalies.loc[anomalies.action.str.startswith('unresolved'),'DATE'].str.slice(0,10))
                    d['directly_corrected_record_day']=d.index.isin(corrected_dates)
                    d['unresolved_record_day']=d.index.isin(unresolved_dates)
                    unresolved=anomalies.loc[anomalies.action.str.startswith('unresolved')].copy()
                    unresolved['date']=unresolved.DATE.str.slice(0,10)
                    d['unresolved_hours']=unresolved.groupby('date').size().reindex(idx,fill_value=0)
            d=apply_daily_consistency(d)
            rejected_count=int(d.ssod_consistency_rejected.sum())
            if rejected_count: reason+=f'; local_SSOD_consistency_rejections={rejected_count}'
            d['weather_usable'] &= approved
            d['daily_status']=d.weather_usable.map({True:'qualified',False:'not_qualified'})
            d.loc[d.ssod_consistency_rejected,'daily_status']='unrecoverable_official_daily_inconsistency'
            d.loc[~d.weather_usable,'temp_c']=float('nan')
            d['rule_version']=RULE
            if not d.weather_usable.any(): approved=False; reason+='; no_qualified_daily_weather'
            d['approved_for_evaluation']=approved
            decision={'gsod_id':r.gsod_id,'ghcn_id':st,'year':year,'status':'approved' if approved else 'not_approved',
                'approved_for_evaluation':approved,'reason':reason,'scan_status':s['scan_status'],
                'both_years_scanned':both_scanned,'mapping_approved':mappingok,'mapping_basis':basis,
                'evidence_level':'hourly_direct_scan' if both_scanned else ('official_daily_product_only' if daily_product_only else 'pending_or_unresolved'),
                'acquisition_status':acquisition_status(st,year),'ssod_product_available':ssodpath is not None,
                'ssod_product_status':'read_error' if year in ssod_errors else ('available' if ssodpath else 'product_not_found'),
                'ssod_error':ssod_errors.get(year,''),
                'metadata_locator':locator,'weather_method':'project_hourly_reconstruction' if project else 'official_ssod_unchanged',
                'usable_weather_days':int(d.weather_usable.sum()),
                'ssod_consistency_rejected_days':rejected_count,
                'source_path':s.get('hourly_path',''),
                'source_sha256':s.get('hourly_sha256',''),'ssod_path':str(ssodpath.relative_to(BASE)) if ssodpath else ''}
            decisions.append(decision); qualification.append({**s,**decision})
            all_daily.append(d.reset_index())
    daily=pd.concat(all_daily,ignore_index=True)
    assert len(daily)==123*(366+365) and not daily.duplicated(['ghcn_id','date']).any()
    dcs=pd.DataFrame(decisions)
    assert len(dcs)==246 and not dcs.duplicated(['gsod_id','year']).any()
    csv_out(daily,TABLES/'WEATHER_DAILY_CORRECTED.csv.gz')
    csv_out(daily.loc[daily.ssod_daily_inconsistent],TABLES/'SSOD_DAILY_INCONSISTENCIES.csv')
    csv_out(dcs,HERE/'station_year_decisions.csv')
    evaluation=daily.loc[daily.approved_for_evaluation & daily.weather_usable,
        ['gsod_id','ghcn_id','date','year','temp_c','selected_hour_count','daily_method']].rename(
        columns={'selected_hour_count':'temperature_observations','daily_method':'weather_method'})
    csv_out(evaluation,HERE/'daily_for_evaluation.csv.gz')
    csv_out(pd.DataFrame(qualification),TABLES/'STATION_YEAR_QUALIFICATION.csv')
    diffs=daily.loc[daily.daily_method.eq('project_hourly_reconstruction')].copy()
    diffs['corrected_minus_ssod_c']=diffs.temp_c-diffs.ssod_temp_c
    csv_out(diffs,TABLES/'DAILY_PRODUCT_DIFFERENCES.csv.gz')
    daily['month']=daily.date.str.slice(0,7)
    monthly=daily.groupby(['gsod_id','ghcn_id','year','month','daily_method']).agg(
        days=('date','size'),usable_days=('weather_usable','sum'),corrected_selected_hours=('corrected_selected_hours','sum'),
        unresolved_hours=('unresolved_hours','sum')).reset_index()
    csv_out(monthly,TABLES/'MONTHLY_COVERAGE.csv')
    csv_out(pd.concat(all_selected,ignore_index=True) if all_selected else pd.DataFrame(columns=['ghcn_id','DATE']),TABLES/'SELECTED_RECONSTRUCTION_HOURS.csv.gz')
    csv_out(pd.concat(all_anomalies,ignore_index=True) if all_anomalies else pd.DataFrame(columns=['ghcn_id','DATE','action']),TABLES/'HOURLY_CORRECTIONS.csv.gz')
    facilities=pd.read_csv(BASE/'qualification/FIXED_FOLLOWUP_135.csv',dtype={'selected_station_id':str})
    fy=facilities[['facility_id','selected_station_id']].rename(columns={'selected_station_id':'gsod_id'}).merge(pd.DataFrame({'year':[2024,2025]}),how='cross')
    mappings=fy.merge(dcs,on=['gsod_id','year'],validate='many_to_one')
    assert len(mappings)==270 and not mappings.duplicated(['facility_id','year']).any()
    mappings['mapping_status']=mappings.mapping_approved.map({True:'approved',False:'not_approved'})
    mappings['weather_carrier_status']=mappings.status
    csv_out(mappings,TABLES/'FACILITY_STATION_MAPPING.csv')
    pending=sum(acquisition_status(s['ghcn_id'],s['year'])=='not_yet_downloaded_or_in_transfer' for s in summaries)
    complete=finalization_allowed(summaries,pending,finalize)
    final={'stage':'outputs_ready_with_explicit_qualification' if complete else 'checkpoint_partial','partial':not complete,
        'pending_hourly_downloads':pending,'utc':datetime.now(timezone.utc).isoformat(),
        'explicit_finalization':complete,
        'rule_version':RULE,'hourly_rule_version':HOURLY_RULE,
        'daily_consistency_amendment':'DAILY_CONSISTENCY_AMENDMENT.md',
        'daily_consistency_after_first_parent_results':True,
        'ssod_consistency_rejected_days':int(daily.ssod_consistency_rejected.sum()),
        'ssod_inconsistent_comparator_project_days':int((daily.ssod_daily_inconsistent & daily.daily_method.eq('project_hourly_reconstruction')).sum()),
        'station_years':246,'hourly_scanned':sum(s['scan_status']=='scanned' for s in summaries),
        'hourly_missing_station_years':sum(s['scan_status']=='not_scanned_missing_hourly' for s in summaries),
        'unknown_qc_rows':sum(s.get('unknown_qc_rows',0) for s in summaries),
        'official_daily_product_only_station_years':int(dcs.evidence_level.eq('official_daily_product_only').sum()),
        'approved_station_years':int(dcs.approved_for_evaluation.sum()),'not_approved_station_years':int((~dcs.approved_for_evaluation).sum()),
        'evaluation_days':len(evaluation),'project_station_years':int(dcs.weather_method.eq('project_hourly_reconstruction').sum()),
        'correction_rows':sum(s.get('corrected_rows',0) for s in summaries),'unresolved_rows':sum(s.get('unresolved_rows',0) for s in summaries),
        'scan_failures':[s for s in summaries if s['scan_status']!='scanned'],
        'models_evaluated':False,'manuscripts_modified':False}
    dump(HERE/'STATUS.json',final);dump(HERE/'PROGRESS.json',final)
    print(json.dumps({k:v for k,v in final.items() if k!='scan_failures'},indent=2))

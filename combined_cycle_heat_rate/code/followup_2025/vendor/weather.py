'Observation-level METAR corroboration and the approved project daily rule.\n\nNo temperatures are divided by ten. Daily reconstruction is not an SSOD replica.\n'
import math
import re
import pandas as pd

RULE = '20260926-v1'
LEGACY = {'313','314','315','322','335','343','344','346'}
OTHER = {'220','221','222','223','347','348'}
FAILED = set('abcdefhik lmnop rstuvwxz'.replace(' ',''))
HEADER = re.compile('(?:METAR\\s+|SPECI\\s+)?(?:COR\\s+)?([A-Z][A-Z0-9]{3})\\s+(?:COR\\s+)?(\\d{2})(\\d{2})(\\d{2})Z\\b')
T_GROUP = re.compile('(?<![A-Z0-9])T([01])(\\d{3})(?:[01]\\d{3}|////)?(?![A-Z0-9])')
BODY = re.compile('(?<!\\S)(M?\\d{2})/(?:M?\\d{2}|/{0,2})(?!\\S)')
ROUTINE = {'FM15','SAO','AUTO'}
SYNOP = {'FM12','FM13','FM14','FM18','FM94_1'}

def clean(v):
    return '' if v is None or (isinstance(v,float) and math.isnan(v)) else str(v).strip()

def quality(source, code):
    source,code=clean(source),clean(code)
    if code == '':
        return (True,'no_flag_reported') if source in LEGACY|OTHER|{'345'} else (False,'unknown_source_quality_combination')
    if any(c in FAILED for c in code): return False, 'ghcnh_failed_test'
    if source in LEGACY:
        if code in {'0','1','4','5','9','A','U','P','I','M','R'}: return True, 'legacy_valid_'+code
        if code in {'2','3','6','7'}: return False, 'legacy_suspect_or_erroneous_'+code
    if source in OTHER:
        if code in {'0','1','4'}: return True, 'alternate_valid_'+code
        if code in {'2','3','5'}: return False, 'alternate_suspect_erroneous_removed_'+code
    return False, 'unknown_source_quality_combination'

def report_temperature(raw, stamp, expected_icao):
    headers = list(HEADER.finditer(raw))
    if len(headers) != 1: return None, None, 'missing_or_ambiguous_header'
    h=headers[0]
    if not expected_icao or h[1] != expected_icao: return None,None,'station_mismatch'
    if stamp[8:10] != h[2] or stamp[11:13] != h[3] or stamp[14:16] != h[4] or stamp[17:19]!='00':
        return None,None,'utc_mismatch'
    # Use only this report, not a neighbouring time or a neighbouring station.
    content=raw[h.end():].strip().rstrip('=')
    vals={(-1 if m[1]=='1' else 1)*int(m[2])/10 for m in T_GROUP.finditer(content)}
    if len(vals)>1: return None,None,'conflicting_t_groups'
    if len(vals)==1: return vals.pop(),.1,'matched_t_group'
    body=content.split('RMK',1)[0]
    vals={-int(m[1][1:]) if m[1].startswith('M') else int(m[1]) for m in BODY.finditer(body)}
    if len(vals)>1: return None,None,'conflicting_integer_groups'
    if len(vals)==1: return float(vals.pop()),1.,'matched_integer_report'
    return None,None,'temperature_not_present'

def check_observation(r, station, icao):
    date=clean(r.get('DATE'))
    try: old=float(r.get('temperature',float('nan')))
    except (ValueError,TypeError): old=float('nan')
    if old <= -999: old=float('nan')
    source=clean(r.get('temperature_Source_Code')); sid=clean(r.get('temperature_Source_Station_ID'))
    report_type=clean(r.get('temperature_Report_Type'))
    qc=clean(r.get('temperature_Quality_Code'))
    good,qreason=quality(source,qc)
    out={'ghcn_id':station,'DATE':date,'source_code':source,'source_station_id':sid,
         'report_type':report_type,'quality_code':qc,'quality_status':qreason,
         'original_temp_c':old,'selected_temp_c':old,'precision_c':.1,
         'action':'unchanged','match_status':'no_message','report_temp_c':float('nan'),
         'valid':bool(good and math.isfinite(old)), 'reason':qreason,
         'report':clean(r.get('REM')),'rule_version':RULE}
    if clean(r.get('STATION')) != station: raise ValueError('hourly physical station identity mismatch')
    if len(date)<19 or date[10]!='T': raise ValueError('unrecognized UTC date syntax')
    if not math.isfinite(old):
        out.update(action='missing_input',selected_temp_c=float('nan'),valid=False)
        return out
    if not good:
        out.update(action='rejected_quality',selected_temp_c=float('nan'),valid=False)
        return out
    raw=out['report']
    if raw:
        if (source != clean(r.get('REM_Source_Code')) or not sid or
                sid != clean(r.get('REM_Source_Station_ID'))):
            match='source_mismatch'; target=precision=None
        elif report_type != clean(r.get('REM_Report_Type')):
            match='report_type_mismatch'; target=precision=None
        else: target,precision,match=report_temperature(raw,date,icao)
        out['match_status']=match
        if target is not None:
            out['report_temp_c']=target
            mismatch=abs(old-target) > (1e-8 if precision==.1 else .50000001)
            # A published whole-degree field can faithfully encode the integer body
            # while the same report also carries tenths. That is not an erratum.
            body_values={-int(m[1][1:]) if m[1].startswith('M') else int(m[1])
                         for m in BODY.finditer(raw.split('RMK',1)[0])}
            if precision==.1 and len(body_values)==1:
                body_value=float(next(iter(body_values)))
                if old==body_value and abs(target-body_value)<=.50000001 and mismatch:
                    mismatch=False
                    out.update(match_status='precision_compatible_integer_field',
                               precision_c=1.,reason='structured_integer_matches_body_and_T_rounding_interval')
            if mismatch:
                # An edited/validator-replaced field can legitimately differ from its old message.
                if qc in {'U','P','M','I'} or not -60<=target<=60:
                    out.update(action='unresolved_conflict',selected_temp_c=float('nan'),valid=False,
                               reason='edited_or_out_of_scope_report_value')
                else:
                    out.update(action='corrected_t_group' if precision==.1 else 'corrected_integer_report',
                               selected_temp_c=target,precision_c=precision,reason=match)
        elif match in {'conflicting_t_groups','conflicting_integer_groups'}:
            out.update(action='unresolved_conflict',selected_temp_c=float('nan'),valid=False,reason=match)
    if out['valid'] and not -60<=out['selected_temp_c']<=60:
        out.update(action='unresolved_out_of_range',selected_temp_c=float('nan'),valid=False,
                   reason='existing_weather_range_and_no_matched_recovery')
    return out

def select_hours(frame):
    'Last whole valid routine record; SYNOP only in an otherwise empty hour.'
    stats={'identical_duplicate_rows':0,'conflicting_identity_rows':0,'conflicting_time_rows':0}
    if frame.empty: return frame.copy(),stats
    f=frame.copy()
    keys=['ghcn_id','DATE','source_code','source_station_id']
    # Conflict includes validity disagreements, not only two different numeric values.
    signature=['selected_temp_c','valid','quality_code','report_type','action']
    exact=f.duplicated(keys+signature)
    stats['identical_duplicate_rows']=int(exact.sum()); f=f.loc[~exact].copy()
    conflict=f.duplicated(keys,keep=False)
    stats['conflicting_identity_rows']=int(conflict.sum())
    f=f.loc[~conflict & f.valid].copy()
    f=f.loc[f.report_type.isin(ROUTINE|SYNOP)].copy()
    if f.empty: return f,stats
    f['priority']=f.report_type.map(lambda t:0 if t in ROUTINE else 1)
    # At one time, independent sources must agree; no unregistered source preference.
    timekeys=['ghcn_id','DATE','priority']
    nvals=f.groupby(timekeys).selected_temp_c.transform('nunique')
    bad=nvals>1
    stats['conflicting_time_rows']=int(bad.sum()); f=f.loc[~bad].copy()
    f=f.sort_values(timekeys+['source_code','source_station_id']).drop_duplicates(timekeys)
    f['hour_key']=f.DATE.str.slice(0,13)
    if f.empty: return f,stats
    minpriority=f.groupby(['ghcn_id','hour_key']).priority.transform('min')
    f=f.loc[f.priority.eq(minpriority)]
    # tail(1) preserves all fields from ONE row; groupby.last() would not.
    f=f.sort_values(['ghcn_id','DATE','source_code']).groupby(['ghcn_id','hour_key'],sort=False).tail(1)
    return f.sort_values(['ghcn_id','DATE']),stats

def daily_from_hours(frame):
    columns=['temp_c','weather_usable','selected_hour_count','utc_segments','corrected_selected_hours','selected_integer_precision_hours']
    if frame.empty: return pd.DataFrame(columns=columns).rename_axis('date')
    f=frame.copy(); f['date']=f.DATE.str.slice(0,10); f['segment']=f.DATE.str.slice(11,13).astype(int)//6
    f['corrected']=f.action.str.startswith('corrected'); f['integer']=f.corrected & f.precision_c.eq(1)
    d=f.groupby('date').agg(temp_c=('selected_temp_c','mean'),selected_hour_count=('DATE','size'),
        utc_segments=('segment','nunique'),corrected_selected_hours=('corrected','sum'),
        selected_integer_precision_hours=('integer','sum'))
    d['weather_usable']=d.utc_segments.eq(4) & d.temp_c.between(-60,60)
    d.loc[~d.weather_usable,'temp_c']=float('nan')
    return d[columns]

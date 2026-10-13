'Portable checks of additional SI tables against packaged frozen results.'
from pathlib import Path
import csv
import gzip
import math
import json
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def table(path):
    with (gzip.open(path,'rt',newline='') if path.suffix=='.gz' else path.open(newline='')) as f:
        return list(csv.DictReader(f))

def keyed(rows, keys):
    out={}
    for r in rows:
        key=tuple(str(r[k]) for k in keys)
        if key in out:raise ValueError('duplicate table identity')
        out[key]=r
    return out

def close(a,b):
    if not math.isclose(float(a),float(b),rel_tol=0,abs_tol=1e-12):
        raise ValueError('table value differs from frozen reduction')


def heat_rate():
    source=ROOT/'followup2025_evidence'
    metrics=keyed([r for r in table(source/'results/FACILITY_MODEL_METRICS.csv') if int(r['year'])==2025],['facility_id','model'])
    predictions=table(source/'results/MATCHED_PREDICTIONS.csv.gz')
    season=table(source/'results/SEASONAL_COVERAGE.csv')
    mapping=keyed([r for r in table(source/'weather/FACILITY_STATION_MAPPING.csv') if int(r['year'])==2025],['facility_id'])
    actual=keyed(table(ROOT/'supporting_tables/S14_2025_LOW_TAIL_DENOMINATORS.csv'),['facility_id','model'])
    if set(actual)!={('56349','M1'),('56349','M2'),('52019','M1'),('52019','M2')}:
        raise ValueError('tail union membership')
    for (fid,model),r in actual.items():
        frozen=metrics[(fid,model)]
        local=[x for x in predictions if x['facility_id']==fid and x['model']==model and int(x['year'])==2025]
        if len({x['date'] for x in local})!=len(local):raise ValueError('duplicate tail daily identity')
        covered=sum(str(x['covered']).lower() in {'1','true','1.0'} for x in local)
        months=[x for x in season if x['facility_id']==fid and int(x['year'])==2025 and float(x['eligible_days'])>0]
        for name,value in [('support_days',len(local)),('covered_days',covered),('week_clusters',len({x['week'] for x in local})),('months',len(months)),('eligible_days',sum(int(x['eligible_days']) for x in months))]:
            close(r[name],value)
        for field,sourcefield in [('coverage','coverage_90'),('existing_ci_low','coverage_ci_low'),('existing_ci_high','coverage_ci_high')]:
            close(r[field],frozen[sourcefield])
        if r['weather_branch']!=mapping[(fid,)]['weather_method'] or r['weather_station']!=mapping[(fid,)]['ghcn_id']:
            raise ValueError('wrong tail weather branch')
    return dict(tail_model_rows=len(actual),new_ci=False)

if __name__=='__main__':
    result=heat_rate()
    print(json.dumps(dict(status='PASS',**result)))

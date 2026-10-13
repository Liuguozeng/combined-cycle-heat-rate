'Independent display-table and paired-figure identities from packaged derivatives.'
import csv
import datetime as dt
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def verify(root=ROOT):
    checks=[]
    table=root/'supporting_tables/S7_TAIL_DENOMINATORS.csv'
    if table.exists():
        days=rows(root/'supporting_tables/S7_MATCHED_DAILY_COVERAGE.csv')
        metrics={(r['year'],r['model'],r['facility_id']):r for r in rows(root/'results/FACILITY_MODEL_METRICS.csv')}
        for r in rows(table):
            key=(r['year'],r['model'],r['facility_id'])
            d=[x for x in days if (x['year'],x['model'],x['facility_id'])==key]
            dates=[dt.date.fromisoformat(x['date']) for x in d]
            weeks={x-dt.timedelta(days=x.weekday()) for x in dates}
            covered=sum(int(x['covered']) for x in d);ref=metrics[key]
            ok=(len(set(dates))==len(d)==int(r['matched_support_days'])==int(ref['matched_support_days'])
                and covered==int(r['covered_days']) and len(weeks)==int(r['calendar_week_clusters'])
                and int(r['daily_endpoint_days'])==int(ref['daily_endpoint_days'])
                and abs(covered/len(d)-float(ref['coverage_90']))<1e-14
                and all(abs(float(r[k])-float(ref[k]))<1e-14 for k in ['coverage_90','coverage_ci_low','coverage_ci_high']))
            checks.append((ok,'tail-denominators:'+':'.join(key)))
        paired=rows(root/'figures/retained_primary/Figure_5/paired_bias_values.csv')
        common={(r['model'],r['facility_id'],r['year']):r for r in rows(root/'results/COMMON_FACILITY_PANEL.csv')}
        checks.append((len(paired)==240 and len({(x['model'],x['facility_id']) for x in paired})==240,'paired-figure-identities'))
        for p in paired:
            checks.append((all(abs(float(p['bias_'+y])-float(common[p['model'],p['facility_id'],y]['signed_bias_mmbtu_per_mwh']))<1e-13
                for y in ['2022','2023','2024']),'paired-bias:'+p['model']+':'+p['facility_id']))
    else:
        data=rows(root/'supporting_tables/S6_SAME_PERIOD_RELATIVE_CONTRASTS.csv')
        checks.append((len(data)==4 and {(r['split'],r['short_days'],r['long_days']) for r in data}==
            {(s,d,'90') for s in ['chronology_holdout','organization_holdout'] for d in ['7','14']},'development-index-identities'))
        actual=rows(root/'figures/Figure_2/plotted_values.csv')
        source=rows(root/'results/CROSS_ECOSYSTEM_CONTRASTS.csv')
        key=lambda r:(r['provider'],r['endpoint_family'],r['comparison'])
        ref={key(r):r for r in source}
        checks.append((len(actual)==18 and {key(r) for r in actual}==set(ref),'endpoint-contrast-identities'))
        for r in actual:
            checks.append((all((not r[k] and not ref[key(r)][k]) or abs(float(r[k])-float(ref[key(r)][k]))<1e-13
                for k in ['estimate','ci_low','ci_high']),'endpoint-contrast:'+':'.join(key(r))))
    return checks

if __name__=='__main__':
    checks=verify();failed=[label for ok,label in checks if not ok]
    print(json.dumps({'checks':len(checks),'failed':failed},indent=2));raise SystemExit(bool(failed))

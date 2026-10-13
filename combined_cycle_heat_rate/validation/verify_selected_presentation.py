'Check current main-figure identities against branch-specific frozen tables.'
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(row, key):
    return int(float(row[key]))


def category(value):
    return 'undercoverage' if float(value)<.85 else 'overcoverage' if float(value)>.95 else 'nominal_band'


def close(a,b):
    assert math.isfinite(float(a)) and math.isfinite(float(b))
    assert math.isclose(float(a),float(b),rel_tol=0,abs_tol=1e-12),(a,b)


def unique(data, keys):
    out={tuple(row[k] for k in keys):row for row in data}
    assert len(out)==len(data),'Duplicate identity'
    return out


def ranks(values):
    ordered=sorted(enumerate(map(float,values)),key=lambda x:x[1])
    result=[0.0]*len(ordered)
    i=0
    while i<len(ordered):
        j=i+1
        while j<len(ordered) and ordered[j][1]==ordered[i][1]:j+=1
        for index,_ in ordered[i:j]:result[index]=(i+1+j)/2
        i=j
    return result


def rank_correlation(a,b):
    a,b=ranks(a),ranks(b)
    ma,mb=sum(a)/len(a),sum(b)/len(b)
    return sum((x-ma)*(y-mb) for x,y in zip(a,b))/math.sqrt(sum((x-ma)**2 for x in a)*sum((y-mb)**2 for y in b))


def verify(root=ROOT):
    selected=root/'selected_presentation'
    cfg=json.loads((selected/'presentation_inputs.json').read_text())
    expected_roles={'primary_fig4':'primary_GSOD_2022_2024','primary_fig5':'primary_GSOD_2022_2024',
                    'followup_metrics':'weather_bridge_2024_and_followup_2025',
                    'followup_pairs':'weather_bridge_2024_to_followup_2025',
                    'followup_persistence':'weather_bridge_2024_to_followup_2025'}
    assert set(cfg['inputs'])==set(expected_roles)
    data={}
    for name,spec in cfg['inputs'].items():
        path=selected/spec['path']
        assert spec['role']==expected_roles[name],'Wrong weather branch'
        assert path.is_file() and sha(path)==spec['sha256'],'Missing or changed presentation input'
        data[name]=rows(path)
    primary=[r for r in data['primary_fig4'] if r['record_type']=='facility_interval_quality_full_annual_cohort']
    follow=[r for r in data['followup_metrics'] if number(r,'year')==2025]
    assert len(primary)==774 and len(follow)==242
    points={('primary_GSOD_2022_2024',number(r,'facility_id'),number(r,'year'),r['model']):r for r in primary}
    points.update({('followup_2025',number(r,'facility_id'),number(r,'year'),r['model']):r for r in follow})
    assert len(points)==1016
    artists=json.loads((root/'figures/Figure_4/ARTIST_DATA.json').read_text())
    actual={(r['data_branch'],r['facility_id'],r['year'],r['model']):r for r in artists}
    assert len(actual)==len(artists)==1016 and set(actual)==set(points),'Missing, added or repeated Figure 4 point'
    for key,r in actual.items():
        ref=points[key]
        close(r['coverage'],ref['coverage_90'])
        close(r['winkler'],ref['mean_winkler_score'])
        assert r['category']==category(ref['coverage_90'])
    plotted=rows(root/'figures/Figure_4/plotted_values.csv')
    table=unique(plotted,['data_branch','facility_id','year','model'])
    assert len(table)==1016
    for r in table.values():
        key=(r['data_branch'],number(r,'facility_id'),number(r,'year'),r['model'])
        assert key in actual
        ref=actual[key]
        close(r['coverage'],ref['coverage'])
        close(r['winkler'],ref['winkler'])
        assert r['category']==ref['category']
    common=[r for r in data['primary_fig5'] if r['record_type']=='common_panel_facility']
    old={number(r,'facility_id') for r in common}
    pairs=unique(data['followup_pairs'],['facility_id','model'])
    new={number(r,'facility_id') for r in pairs.values()}
    assert len(old)==len(new)==120 and len(old&new)==106
    for r in pairs.values():
        assert number(r,'year_2024')==2024 and number(r,'year_2025')==2025
    for figure,branch,wanted in [('5','followup',new),('S1','retrospective',old)]:
        artists=json.loads((root/'figures'/('Figure_'+figure)/'ARTIST_DATA.json').read_text())
        actual={(r['data_branch'],r['model'],r['facility_id']):r for r in artists}
        assert len(actual)==len(artists)==240
        assert set(actual)=={(branch,model,fid) for model in ['M1','M2'] for fid in wanted}
        plotted=rows(root/'figures'/('Figure_'+figure)/'plotted_values.csv')
        table=unique(plotted,['data_branch','model','facility_id'])
        assert len(table)==240
        for r in table.values():
            key=(r['data_branch'],r['model'],number(r,'facility_id'))
            assert key in actual
            for field in ['bias_a','bias_b']:
                close(r[field],actual[key][field])
            assert number(r,'year_a')==int(actual[key]['year_a'])
            assert number(r,'year_b')==int(actual[key]['year_b'])
        for (_,model,fid),r in actual.items():
            if branch=='followup':
                ref=pairs[str(fid),model]
                close(r['bias_a'],ref['signed_bias_mmbtu_per_mwh_2024'])
                close(r['bias_b'],ref['signed_bias_mmbtu_per_mwh_2025'])
            else:
                refs=[x for x in common if x['model']==model and number(x,'facility_id')==fid]
                for year,field in [(2022,'bias_a'),(2024,'bias_b')]:
                    ref=next(x for x in refs if number(x,'year')==year)
                    close(r[field],ref['signed_bias_mmbtu_per_mwh'])
        presentation=json.loads((root/'figures'/('Figure_'+figure)/'PRESENTATION_DATA.json').read_text())
        if figure=='5':
            summaries=unique(presentation['followup_summaries'],['model'])
            frozen=unique(data['followup_persistence'],['model'])
            for model in ['M1','M2']:
                rr=[r for r in pairs.values() if r['model']==model]
                current=summaries[model,]
                assert current['facilities']==len(rr)==120
                for metric in ['bias','coverage']:
                    fields=(['signed_bias_mmbtu_per_mwh_2024','signed_bias_mmbtu_per_mwh_2025'] if metric=='bias' else ['coverage_90_2024','coverage_90_2025'])
                    computed=rank_correlation([r[fields[0]] for r in rr],[r[fields[1]] for r in rr])
                    field='signed_bias_spearman' if metric=='bias' else 'coverage_spearman'
                    close(current[field],computed)
                    close(current[field],frozen[model,][field])
                same=sum(r['coverage_direction_2024']==r['coverage_direction_2025'] for r in rr)
                assert current['same_coverage_category']==same==number(frozen[model,],'same_coverage_category')
        counts=Counter((r['model'],number(r,'year'),category(r['coverage_90'])) for r in common)
        for r in presentation.get('coverage_counts',[]):
            assert r['facilities']==120
            for cat in ['undercoverage','nominal_band','overcoverage']:
                assert r[cat]==counts[r['model'],r['year'],cat]
    return {'status':'PASS','primary_points':774,'followup_points':242,
            'paired_points_per_model':120,'retrospective_points_per_model':120,'panel_intersection':106}


if __name__=='__main__':
    print(json.dumps(verify(),indent=2))

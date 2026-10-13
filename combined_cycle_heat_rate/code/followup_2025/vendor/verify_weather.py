import argparse

import json

from pathlib import Path

import numpy as np

import pandas as pd

def check(root):
    daily=pd.read_csv(root/'tables/WEATHER_DAILY_CORRECTED.csv.gz')
    selected=pd.read_csv(root/'tables/SELECTED_RECONSTRUCTION_HOURS.csv.gz')
    corrections=pd.read_csv(root/'tables/HOURLY_CORRECTIONS.csv.gz')
    if daily.duplicated(['ghcn_id','date']).any():raise ValueError('Duplicate daily identities')
    if selected.empty:raise ValueError('Missing reconstruction hours')
    selected['day']=selected.DATE.str.slice(0,10)
    selected['hour']=selected.DATE.str.slice(11,13).astype(int)
    if selected.duplicated(['ghcn_id','day','hour']).any():raise ValueError('More than one selected observation per clock hour')
    checks=0;maximum=0.
    project=daily.loc[daily.daily_method.eq('project_hourly_reconstruction')].set_index(['ghcn_id','date'])
    for key,g in selected.groupby(['ghcn_id','day']):
        if key not in project.index:raise ValueError('Selected hours missing their project day')
        row=project.loc[key];segments={int(h)//6 for h in g.hour}
        approved=str(row.get('approved_for_evaluation',True)).lower()=='true'
        valid=segments=={0,1,2,3} and approved
        if bool(row.weather_usable)!=valid:raise ValueError('Daily completeness mismatch')
        if int(row.selected_hour_count)!=len(g):raise ValueError('Hourly denominator mismatch')
        if valid:
            direct=sum(map(float,g.selected_temp_c))/len(g)
            delta=abs(direct-float(row.temp_c));maximum=max(maximum,delta)
            if delta>1e-12:raise ValueError('Independent daily average differs')
        checks+=1
    known={'2025-04-18T14:54:00':10.,'2025-05-16T22:53:00':27.,'2025-05-28T20:54:00':26.,'2025-07-07T15:55:00':21.}
    fixtures=[]
    for stamp,value in known.items():
        g=corrections.loc[corrections.ghcn_id.eq('USW00023122')&corrections.DATE.eq(stamp)]
        if len(g)!=1 or float(g.selected_temp_c.iloc[0])!=value:raise ValueError('Known fixture missing or incorrect: '+stamp)
        if float(g.precision_c.iloc[0])!=1.:raise ValueError('Integer precision not retained')
        fixtures.append(dict(stamp=stamp,original=float(g.original_temp_c.iloc[0]),corrected=value))
    return dict(status='PASS',independent_daily_checks=checks,maximum_daily_delta=maximum,known_fixtures=fixtures,
                scope='selected UTC hour aggregation and four original conflicts; not a claim that every upstream record is correct')

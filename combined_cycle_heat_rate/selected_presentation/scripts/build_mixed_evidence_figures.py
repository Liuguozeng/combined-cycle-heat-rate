#!/usr/bin/env python3
'Deterministic presentation of frozen values; no model fitting or CI estimation.'
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

BLUE = '#2879AD'
MID = '#91B9D0'
RED = '#C97E9C'
GRAY = '#7C8792'
INK = '#26333D'
PALE = '#E5EBEF'
CATS = {'undercoverage': RED, 'nominal_band': GRAY, 'overcoverage': BLUE}
PAPERS = {'return_temperature_comparability':'return_temperature_comparability', 'combined_cycle_heat_rate':'combined_cycle_heat_rate'}
SUPS = {'return_temperature_comparability':'Supplementary_Data', 'combined_cycle_heat_rate':'Supplementary_Data'}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def configure_style():
    plt.rcParams.update({'font.family':'serif', 'font.serif':['Times New Roman', 'DejaVu Serif'],
        'font.size':9, 'axes.labelsize':9, 'axes.titlesize':10, 'xtick.labelsize':8.5,
        'ytick.labelsize':8.5, 'legend.fontsize':8.5, 'axes.edgecolor':GRAY,
        'axes.linewidth':.6, 'text.color':INK, 'xtick.color':INK, 'ytick.color':INK,
        'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'none', 'savefig.facecolor':'white'})

def clean(ax, grid='y'):
    ax.spines[['top','right']].set_visible(False)
    if grid:ax.grid(axis=grid, color='#E5EBEF', linewidth=.45, zorder=0)
    ax.set_axisbelow(True);ax.tick_params(length=3, width=.6)

def validate_categories(frame):
    unknown=set(frame.coverage_direction.dropna())-set(CATS)
    if unknown:raise ValueError(f'Unknown coverage categories: {unknown}')
    expected=np.where(frame.coverage_90<.85,'undercoverage',np.where(frame.coverage_90>.95,'overcoverage','nominal_band'))
    if not np.array_equal(expected,frame.coverage_direction):raise ValueError('Coverage category disagrees with frozen thresholds')

def bound(value):
    return float(value) if np.isfinite(value) else None

def uncertainty(provider):
    return 'point_only' if provider=='AAU' else 'descriptive_range' if provider=='Empa/NEST' else 'cluster_CI'

def draw_interval(ax, position, lo, hi, color, provider, horizontal=False, cap=.10):
    'Whiskers distinguish intervals from distribution boxes; AAU has none.'
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return
    style='--' if provider=='Empa/NEST' else '-'
    if horizontal:
        ax.plot([lo,hi],[position,position],color=color,lw=1.15,ls=style,zorder=4)
        for value in [lo,hi]:
            ax.plot([value,value],[position-cap,position+cap],color=color,lw=.75,zorder=4)
    else:
        ax.plot([position,position],[lo,hi],color=color,lw=1.15,ls=style,zorder=4)
        for value in [lo,hi]:
            ax.plot([position-cap,position+cap],[value,value],color=color,lw=.75,zorder=4)

def effect(ax,y,value,lo,hi,color,provider,height=.18):
    'Magnitude bar plus a co-located interval band, not a duplicate estimate.'
    bar=ax.barh(y,value,height=height*.50,color=color,alpha=.90,zorder=3)[0]
    if np.isfinite(lo) and np.isfinite(hi):
        ax.add_patch(Rectangle((lo,y-height/2),hi-lo,height,facecolor=color,alpha=.12,edgecolor='none',zorder=2))
        draw_interval(ax,y,lo,hi,color,provider,horizontal=True,cap=height*.42)
    ax.plot([value,value],[y-height*.63,y+height*.63],color=color,lw=1.25,zorder=5)
    return float(bar.get_width())

def proportion_bands(ax, model, rows, panel):
    'Paired zero-based proportion tracks; only the operator estimate has a CI.'
    ax.clear();ax.axis('off');ax.set(xlim=(-.17,1.39),ylim=(3,0))
    ax.set_title(f'{panel}  {model} | 120 common facilities',loc='left',fontsize=9.2,weight='bold',pad=25)
    for x,label,color in [(0,'Operator equal',BLUE),(.72,'Facility equal',MID)]:
        ax.add_patch(Rectangle((x,-.33),.06,.12,facecolor=color,clip_on=False))
        ax.text(x+.085,-.27,label,va='center',fontsize=7.8)
    for y,r in enumerate(rows.sort_values('year').itertuples()):
        ax.text(-.04,y+.39,str(int(r.year)),ha='right',va='center',fontsize=8.1)
        for offset,value,color,height in [(.16,float(r.operator_equal_supported_and_covered),BLUE,.20),
                                           (.55,float(r.facility_equal_supported_and_covered),MID,.16)]:
            ax.add_patch(Rectangle((0,y+offset-height/2),1,height,facecolor=PALE,edgecolor='none'))
            ax.add_patch(Rectangle((0,y+offset-height/2),value,height,facecolor=color,edgecolor='none'))
            ax.text(1.045,y+offset,f'{value:.1%}',va='center',fontsize=8.5,color=color,weight='bold')
        lo,hi=float(r.operator_cluster_ci95_lower),float(r.operator_cluster_ci95_upper)
        ax.add_patch(Rectangle((lo,y+.025),hi-lo,.27,facecolor='white',alpha=.55,edgecolor=INK,lw=.6))
        ax.text(.02,y+.86,f'95% CI [{lo:.1%}, {hi:.1%}]',va='center',fontsize=7.2,color=INK)
    ax.text(0,3.10,'0%',ha='center',va='top',fontsize=7.2,color=INK)
    ax.text(1,3.10,'100%',ha='center',va='top',fontsize=7.2,color=INK)

def supported_coverage_tracks(ax, model, rows, panel):
    'Two full-scale tracks, with numerical CI beside its operator estimate.'
    ax.clear();ax.axis('off');ax.set(xlim=(-.17,1.55),ylim=(3,0))
    ax.set_title(f'{panel}  {model} | Supported-and-covered',loc='left',
                 fontsize=9.2,weight='bold',pad=24)
    ax.text(0,-.66,'120 common facilities',fontsize=8.1,color=INK,clip_on=False)
    for x,label,color in [(0,'Operator equal',BLUE),(.82,'Facility equal',MID)]:
        ax.add_patch(Rectangle((x,-.35),.055,.12,facecolor=color,clip_on=False))
        ax.text(x+.08,-.29,label,va='center',fontsize=7.8)
    for y,r in enumerate(rows.sort_values('year').itertuples()):
        year=int(r.year)
        ax.text(-.05,y+.43,str(year),ha='right',va='center',fontsize=8.1)
        for offset,value,color,weight in [(.20,float(r.operator_equal_supported_and_covered),BLUE,'operator'),
                                          (.67,float(r.facility_equal_supported_and_covered),MID,'facility')]:
            height=.38
            ax.add_patch(Rectangle((0,y+offset-height/2),1,height,
                                   facecolor=PALE,edgecolor='none'))
            bar=Rectangle((0,y+offset-height/2),value,height,facecolor=color,edgecolor='none')
            bar.set_gid(f'proportion:{model}:{year}:{weight}');ax.add_patch(bar)
            ax.text(.035,y+offset,f'{value:.1%}',va='center',fontsize=8.0,
                    color='white' if weight=='operator' else INK,weight='bold')
        lo,hi=float(r.operator_cluster_ci95_lower),float(r.operator_cluster_ci95_upper)
        band=Rectangle((lo,y+.01),hi-lo,.38,facecolor='white',alpha=.70,
                       edgecolor=INK,lw=.6)
        band.set_gid(f'operator_ci:{model}:{year}');ax.add_patch(band)
        value=float(r.operator_equal_supported_and_covered)
        ax.plot([value,value],[y+.01,y+.39],color=INK,lw=.85)
        ax.text(1.04,y+.20,f'[{lo:.1%}, {hi:.1%}]',va='center',fontsize=7.8,color=INK)
    for x,label in [(0,'0%'),(.5,'50%'),(1,'100%')]:
        ax.text(x,3.08,label,ha='center',va='top',fontsize=7.5,color=INK)


class Builder:
    def __init__(self,candidate,output,selected=None,fig4_variant='A',followup=None,fig5_variant='legacy',emit_retrospective_si=False,package_root=None):
        self.candidate,self.output=candidate,output
        self.selected=set(selected or []);self.records=[];self.fig4_variant=fig4_variant
        self.followup=followup;self.fig5_variant=fig5_variant;self.emit_retrospective_si=emit_retrospective_si
        self.package_root=package_root
        if package_root is not None and (not self.selected or not self.selected.issubset({'return_temperature_comparability:2','return_temperature_comparability:3','return_temperature_comparability:4'})):
            raise ValueError('Package-root rendering requires an explicit return_temperature_comparability:2/3/4 subset')
        if followup is not None:
            if not self.selected or not self.selected.issubset({'combined_cycle_heat_rate:4','combined_cycle_heat_rate:5'}):
                raise ValueError('Followup presentation requires an explicit combined_cycle_heat_rate:4/5 subset')
            if fig4_variant!='A':raise ValueError('Followup preserves the accepted Fig.4 A organization')
            if 'combined_cycle_heat_rate:5' in self.selected and fig5_variant=='legacy':
                raise ValueError('Choose parallel or followup_first; do not silently render the old Fig.5')
        elif fig5_variant!='legacy' or emit_retrospective_si:
            raise ValueError('A hash-bound followup input manifest is required')
        if emit_retrospective_si and fig5_variant!='followup_first':
            raise ValueError('The retrospective SI preview belongs to the followup-first option')

    def data(self,paper,name):
        if self.package_root is not None:
            if paper!='return_temperature_comparability':raise ValueError('return_temperature_comparability package cannot supply another paper')
            return pd.read_csv(self.package_root/'results'/name)
        return pd.read_csv(self.candidate/PAPERS[paper]/SUPS[paper]/'results'/name)

    def wanted(self,paper,number):
        return not self.selected or f'{paper}:{number}' in self.selected

    def export(self,paper,index,fig,artists,supporting=None):
        parent=(self.package_root/'figures'/f'Figure_{index}' if self.package_root is not None
                else next((self.candidate/PAPERS[paper]/'figures').glob(f'FIG{index}_*')))
        out=self.output/paper/parent.name
        if out.resolve()==parent.resolve():raise ValueError('Render to an isolated output root, without overwriting source inputs')
        out.mkdir(parents=True,exist_ok=True)
        for path in parent.glob('*'):
            if path.is_file() and path.name not in {'ARTIST_DATA.json','PRESENTATION_DATA.json'} and path.suffix not in {'.pdf','.svg','.png','.tiff'}:
                shutil.copyfile(path,out/path.name)
        fig.canvas.draw();hidden_ticks=set()
        for ax in fig.axes:
            for axis in (ax.xaxis,ax.yaxis):
                lo,hi=sorted(axis.get_view_interval())
                for tick in axis.get_major_ticks()+axis.get_minor_ticks():
                    if not lo<=tick.get_loc()<=hi:hidden_ticks.update((tick.label1,tick.label2))
        overflow=[]
        for t in fig.findobj(matplotlib.text.Text):
            if t in hidden_ticks or not t.get_visible() or not t.get_text():continue
            b=t.get_window_extent()
            if b.x0 < -2 or b.y0 < -2 or b.x1 > fig.bbox.x1+2 or b.y1 > fig.bbox.y1+2:overflow.append(t.get_text())
        stem=out/parent.name
        for suffix in ['pdf','svg','png','tiff']:
            opts={'dpi':500,'pil_kwargs':{'compression':'tiff_lzw'}} if suffix=='tiff' else {'dpi':300}
            fig.savefig(stem.with_suffix('.'+suffix),**opts)
        plt.close(fig)
        contract=json.loads((out/'FIGURE_NUMERIC_CONTRACT.json').read_text())
        contract.update(release_version='1.0.0')
        (out/'FIGURE_NUMERIC_CONTRACT.json').write_text(json.dumps(contract,indent=2)+'\n')
        (out/'ARTIST_DATA.json').write_text(json.dumps(artists,indent=2,allow_nan=False)+'\n')
        (out/'PRESENTATION_DATA.json').write_text(json.dumps(supporting or {},indent=2,allow_nan=False)+'\n')
        self.records.append(dict(paper=paper,figure=index,folder=parent.name,canvas_overflow=overflow,
            original_plotted_sha256=digest(parent/'plotted_values.csv'),plotted_sha256=digest(out/'plotted_values.csv'),objects=len(artists)))

    def return_temperature_2(self):
        df=self.data('return_temperature_comparability','CROSS_ECOSYSTEM_CONTRASTS.csv')
        assert len(df)==18 and not df.duplicated(['provider','endpoint_family','comparison']).any()
        providers=['PreDist','XAI4HEAT','AAU','Empa/NEST']
        fig,axes=plt.subplots(1,2,figsize=(178/25.4,4.65),sharex=True)
        fig.subplots_adjust(left=.14,right=.97,bottom=.18,top=.83,wspace=.18)
        artists=[]
        for i,(ax,ep) in enumerate(zip(axes,['median','heat_weighted'])):
            ps=providers+(['Aalborg'] if i else [])
            for j,provider in enumerate(ps):
                y=4-j
                for k,(co,color) in enumerate(zip(['7_minus_90_days','14_minus_90_days'],[BLUE,MID])):
                    r=df[df.provider.eq(provider)&df.endpoint_family.eq(ep)&df.comparison.eq(co)].iloc[0]
                    yy=y+(.17 if k==0 else -.17)
                    v=effect(ax,yy,r.estimate,r.ci_low,r.ci_high,color,provider,.24)
                    ax.text(2.23,yy,f'{v:.2f}',ha='right',va='center',fontsize=8.5,color=INK)
                    artists.append(dict(provider=provider,endpoint=ep,comparison=co,estimate=v,
                        lower=bound(r.ci_low),upper=bound(r.ci_high),uncertainty=uncertainty(provider)))
            ticks=[4,3,2,1] if i==0 else [0]
            ax.set(xlim=(0,2.30),ylim=(-.48,4.52),xticks=[0,.5,1,1.5,2],yticks=ticks,
                   yticklabels=providers if i==0 else ['Aalborg'])
            ax.set_title(f'{chr(97+i)}  '+('Median' if i==0 else 'Heat-weighted'),loc='left',weight='bold',pad=12)
            clean(ax,'x');ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
        fig.legend(handles=[Patch(facecolor=BLUE,label='7−90 days'),Patch(facecolor=MID,label='14−90 days')],
                   loc='upper center',bbox_to_anchor=(.57,.995),ncol=2,frameon=False)
        short=df[df.comparison.eq('7_minus_90_days')]
        ranges=[short[short.endpoint_family.eq(ep)].estimate.agg(['min','max']).to_numpy() for ep in ['median','heat_weighted']]
        fig.text(.57,.90,f'7−90-day contrasts: {ranges[0][0]:.2f}–{ranges[0][1]:.2f} °C median; {ranges[1][0]:.2f}–{ranges[1][1]:.2f} °C heat-weighted',ha='center',fontsize=8.5,color=INK)
        fig.text(.57,.085,'Short minus 90-day absolute error (°C)',ha='center',fontsize=9)
        fig.text(.14,.028,'Whiskers: 95% CI; dashed: two-system range. AAU: estimate only.',fontsize=8.2,color=INK)
        self.export('return_temperature_comparability',2,fig,artists)

    def return_temperature_3(self):
        df=self.data('return_temperature_comparability','CROSS_ECOSYSTEM_DURATION_SUMMARY.csv')
        df=df[df.endpoint_family.eq('median')]
        fig,axes=plt.subplots(4,1,figsize=(178/25.4,6.2),sharex=True,sharey=True)
        fig.subplots_adjust(left=.15,right=.975,bottom=.14,top=.94,hspace=.34)
        artists=[]
        upper=max(df.estimate.max(),df.ci_high.max())
        ymax=np.ceil(upper*5)/5+.20
        for i,(ax,provider) in enumerate(zip(axes,['PreDist','XAI4HEAT','AAU','Empa/NEST'])):
            rr=df[df.provider.eq(provider)].sort_values('duration_days')
            for j,r in enumerate(rr.itertuples()):
                color=RED if r.duration_days==90 else BLUE
                bar=ax.bar(j,r.estimate,width=.23,color=color,alpha=.72,zorder=3)[0]
                draw_interval(ax,j,r.ci_low,r.ci_high,color,provider,cap=.13)
                ax.plot([j-.18,j+.18],[r.estimate]*2,color=color,lw=1.25,zorder=5)
                ax.text(j+.26,r.estimate,f'{r.estimate:.2f}',va='center',fontsize=8.8,color=INK)
                artists.append(dict(provider=provider,duration=int(r.duration_days),estimate=float(bar.get_height()),
                    lower=bound(r.ci_low),upper=bound(r.ci_high),uncertainty=uncertainty(provider)))
            ax.set(ylim=(0,ymax),xlim=(-.62,4.72),yticks=[0,1,2],xticks=range(5),xticklabels=['7','14','30','60','90'])
            clean(ax,'y');ax.spines['left'].set_visible(False);ax.spines['bottom'].set_color('#C9D2D9')
            ax.tick_params(axis='y',length=0)
            ax.text(-.17,.65,f'{chr(97+i)}\n{provider}',transform=ax.transAxes,va='center',fontsize=9,weight='bold')
        axes[-1].set_xlabel('Campaign duration (days)')
        fig.text(.15,.975,'Mean absolute error for the median endpoint (°C)',fontsize=10,weight='bold')
        fig.text(.15,.043,'Common scale; whiskers: 95% CI; dashed: two-system range; AAU: estimates.',fontsize=8.1,color=INK)
        fig.text(.15,.016,'Red marks the 90-day comparator. All five protocols are discrete observations.',fontsize=8.1,color=INK)
        self.export('return_temperature_comparability',3,fig,artists,{'axis_scales':'common, zero based','axis_limits':[0,float(ymax)],'durations':[7,14,30,60,90]})

    def return_temperature_4(self):
        equal=self.data('return_temperature_comparability','EQUAL_DAY_CONTRASTS.csv')
        disjoint=self.data('return_temperature_comparability','NONOVERLAP_CONTRASTS.csv')
        fig=plt.figure(figsize=(178/25.4,5.35))
        gs=fig.add_gridspec(2,1,height_ratios=[4,2],left=.18,right=.975,bottom=.19,top=.88,hspace=.52)
        artists=[]
        for i,(df,providers,title) in enumerate([(equal,['PreDist','XAI4HEAT','AAU','Empa/NEST'],'Equal-day aggregation'),
                    (disjoint,['PreDist','XAI4HEAT'],'Date-disjoint reference')]):
            ax=fig.add_subplot(gs[i]);n=len(providers)
            for j,provider in enumerate(providers):
                y=n-1-j
                for offset,short,color in [(.17,7,BLUE),(-.17,14,MID)]:
                    r=df[df.provider.eq(provider)&df.short_days.eq(short)].iloc[0]
                    v=effect(ax,y+offset,r.contrast_c,r.ci95_lower,r.ci95_upper,color,provider,.23)
                    ax.text(1.67,y+offset,f'{v:+.2f}',ha='right',va='center',color=INK,fontsize=8.5)
                    artists.append(dict(analysis='equal_day' if i==0 else 'date_disjoint',provider=provider,
                        short_days=short,estimate=v,lower=bound(r.ci95_lower),upper=bound(r.ci95_upper),uncertainty=uncertainty(provider)))
            ax.axvline(0,color=GRAY,lw=.75,zorder=1)
            ax.set(xlim=(-.58,1.73),ylim=(-.48,n-.48),yticks=range(n),yticklabels=providers[::-1],xticks=[-.5,0,.5,1,1.5])
            clean(ax,'x');ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0)
            ax.set_title(f'{chr(97+i)}  {title}',loc='left',weight='bold',pad=8)
        ax.set_xlabel('Median-endpoint error contrast (°C)')
        fig.legend(handles=[Patch(facecolor=BLUE,label='7−90 days'),Patch(facecolor=MID,label='14−90 days')],loc='upper right',bbox_to_anchor=(.985,.985),ncol=2,frameon=False)
        fig.text(.18,.085,'Whiskers: 95% CI; dashed: two-system range; AAU: estimate only.',fontsize=8.2,color=INK)
        fig.text(.18,.040,'Date-disjoint eligibility: ≥5 systems and ≥30 common-start instances;',fontsize=8.2,color=INK)
        fig.text(.18,.014,'AAU and Empa/NEST do not meet this requirement.',fontsize=8.2,color=INK)
        self.export('return_temperature_comparability',4,fig,artists)

    def heat_rate_2(self):
        df=pd.read_csv(next((self.candidate/PAPERS['combined_cycle_heat_rate']/'figures').glob('FIG2_*/plotted_values.csv')))
        fig=plt.figure(figsize=(178/25.4,4.05))
        gs=fig.add_gridspec(1,2,width_ratios=[3.1,1.25],left=.10,right=.98,bottom=.24,top=.88,wspace=.28)
        ax=fig.add_subplot(gs[0]);summary=fig.add_subplot(gs[1]);summary.axis('off')
        artists=[];quantiles=[]
        for i,(year,color) in enumerate(zip([2022,2023,2024],[BLUE,MID,RED])):
            rr=df[df.year.eq(year)].sort_values(['support_fraction','facility_id'])
            values=rr.support_fraction.to_numpy();p=np.arange(1,len(values)+1)/len(values)
            ax.step(np.r_[values[0],values],np.r_[0,p],where='post',color=color,lw=1.6,label=str(year))
            ax.vlines(values,-.13-i*.05,-.097-i*.05,color=color,lw=.55,alpha=.50,clip_on=False)
            q=np.quantile(values,[.05,.25,.5,.75,.95]);y=.88-i*.31
            summary.text(0,y,f'{year}   n = {len(values)}',fontsize=10,weight='bold',color=INK)
            summary.text(0,y-.09,f'Median  {q[2]:.1%}',fontsize=9)
            summary.text(0,y-.17,f'IQR  {q[1]:.1%}–{q[3]:.1%}',fontsize=8.5,color=INK)
            quantiles.append(dict(year=year,n=len(values),q05=float(q[0]),q25=float(q[1]),median=float(q[2]),q75=float(q[3]),q95=float(q[4])))
            for r,yy in zip(rr.itertuples(),p):artists.append(dict(facility_id=int(r.facility_id),year=year,support=float(r.support_fraction),ecdf=float(yy)))
        xmin=max(0,float(df.support_fraction.min())-.025)
        ax.set(xlim=(xmin,1.005),ylim=(0,1.01),ylabel='Cumulative proportion of facilities')
        ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=0));ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
        ax.set_xlabel('Temperature–load input-support retention',labelpad=40)
        clean(ax,'both');ax.legend(loc='upper left',frameon=False)
        fig.text(.10,.95,'Operating-domain overlap across receiving facilities',weight='bold',fontsize=10)
        self.export('combined_cycle_heat_rate',2,fig,artists,{'quantiles':quantiles,'quantile_method':'numpy linear; descriptive only','empirical_facility_years':len(df)})

    def heat_rate_3(self):
        df=pd.read_csv(next((self.candidate/PAPERS['combined_cycle_heat_rate']/'figures').glob('FIG3_*/plotted_values.csv')))
        fig,axes=plt.subplots(2,2,figsize=(178/25.4,4.45),gridspec_kw={'width_ratios':[1.9,1]})
        fig.subplots_adjust(left=.075,right=.975,bottom=.16,top=.89,hspace=.53,wspace=.24)
        bias_lim=float(df.signed_bias_mmbtu_per_mwh.abs().max())*1.08
        mae_lim=float(df.mae_mmbtu_per_mwh.max())*1.06;artists=[];stats=[]
        for i,model in enumerate(['M1','M2']):
            rr=df[df.model.eq(model)]
            for j,column in enumerate(['signed_bias_mmbtu_per_mwh','mae_mmbtu_per_mwh']):
                ax=axes[i,j];values=np.sort(rr[column].to_numpy())
                grid=np.linspace(-bias_lim,bias_lim,360) if j==0 else np.linspace(0,mae_lim,300)
                density=gaussian_kde(values,bw_method=.18)(grid);density=.35*density/density.max()
                ax.fill_between(grid,0,density,color=MID,alpha=.28,lw=0);ax.plot(grid,density,color=BLUE,lw=1.3)
                ax.vlines(values,-.048,0,color=BLUE,alpha=.5,lw=.55)
                med=float(np.median(values));q25,q75=np.quantile(values,[.25,.75])
                ax.plot([q25,q75],[-.11]*2,color=MID,lw=5,solid_capstyle='butt')
                ax.scatter(med,-.11,s=25,facecolor='white',edgecolor=BLUE,zorder=3)
                if j==0:ax.axvline(0,color=GRAY,lw=.65)
                ax.text(.98,.93,f'{values.min():.3f} to {values.max():.3f}',transform=ax.transAxes,ha='right',va='top',fontsize=8.3,color=INK)
                ax.set(xlim=(grid.min(),grid.max()),ylim=(-.18,.46),yticks=[])
                ax.set_title(f'{chr(97+2*i+j)}  {model} | '+('Signed bias' if j==0 else 'MAE'),loc='left',weight='bold',fontsize=9.5)
                if i==1:ax.set_xlabel(('Signed bias' if j==0 else 'MAE')+' (MMBtu/MWh)')
                clean(ax,'x');ax.spines['left'].set_visible(False)
                stats.append(dict(model=model,metric=column,minimum=float(values.min()),maximum=float(values.max()),median=med,q25=float(q25),q75=float(q75)))
            for r in rr.itertuples():artists.append(dict(facility_id=int(r.facility_id),year=int(r.year),model=model,bias=float(r.signed_bias_mmbtu_per_mwh),mae=float(r.mae_mmbtu_per_mwh)))
        fig.text(.075,.965,'2024 screened cohort | 135 facilities per relation',fontsize=9.5,color=INK)
        fig.text(.075,.037,'Rugs: observed facilities. Open marks: medians. Thick bands: interquartile ranges.',fontsize=8.2,color=INK)
        self.export('combined_cycle_heat_rate',3,fig,artists,{'descriptive_summaries':stats,'density_bandwidth':.18})

    def heat_rate_4(self):
        from matplotlib.patches import ConnectionPatch
        df=pd.read_csv(next((self.candidate/PAPERS['combined_cycle_heat_rate']/'figures').glob('FIG4_*/plotted_values.csv')))
        hero=df[df.record_type.eq('facility_interval_quality_full_annual_cohort')].copy()
        joint=df[df.record_type.eq('supported_and_covered_common_panel')].copy()
        validate_categories(hero)
        assert len(hero)==774 and not hero.duplicated(['facility_id','year','model']).any()
        fig=plt.figure(figsize=(178/25.4,6.65))
        gs=fig.add_gridspec(3,2,height_ratios=[1,1,.68],left=.10,right=.98,bottom=.12,top=.87,hspace=.78,wspace=.20)
        markers={2022:'o',2023:'s',2024:'^'};artists=[];matrix=[]
        ylim=np.ceil(hero.mean_winkler_score.max())
        cmap=LinearSegmentedColormap.from_list('proportion',['#F4F6F7','#C6D5DF'])
        for i,model in enumerate(['M1','M2']):
            rg=gs[i,:].subgridspec(1,2,width_ratios=[1.95,1.2],wspace=.25)
            ax=fig.add_subplot(rg[0]);inset=fig.add_subplot(rg[1]);block=hero[hero.model.eq(model)]
            width=float(block.interval_width_mmbtu_per_mwh.iloc[0]);zoom=block[block.coverage_90.ge(.85)]
            for target,is_zoom in [(ax,False),(inset,True)]:
                target.axvspan(.85,.95,color=PALE,zorder=-3);target.axvline(.90,ls='--',color=GRAY,lw=.7)
                target.axhline(width,ls=':',color=GRAY,lw=.7)
                for year,marker in markers.items():
                    for cat,color in CATS.items():
                        rr=block[block.year.eq(year)&block.coverage_direction.eq(cat)]
                        if is_zoom:rr=rr[rr.coverage_90.ge(.85)]
                        points=target.scatter(rr.coverage_90,rr.mean_winkler_score,s=18,marker=marker,color=color,alpha=.75,edgecolor='white',lw=.25)
                        if not is_zoom:
                            for r,xy in zip(rr.itertuples(),points.get_offsets()):
                                artists.append(dict(facility_id=int(r.facility_id),year=int(r.year),model=model,category=cat,x=float(xy[0]),y=float(xy[1])))
                clean(target,'y');target.xaxis.set_major_formatter(PercentFormatter(1,decimals=0))
            ax.set(xlim=(0,1.01),ylim=(0,ylim),ylabel='Mean Winkler score\n(MMBtu/MWh)')
            ax.set_title(f'{chr(97+i)}  {model} | 387 facility-years',loc='left',weight='bold',fontsize=10,pad=17)
            ax.set_xlabel('Coverage conditional on matched input support')
            inset.set(xlim=(.85,1.005),ylim=(width*.96,float(zoom.mean_winkler_score.max())*1.06),xticks=[.85,.90,.95,1])
            inset.set_title('85–100% coverage detail',fontsize=9,pad=17);inset.tick_params(labelsize=8)
            lo,hi=inset.get_ylim()
            ax.add_patch(Rectangle((.85,lo),.15,hi-lo,fill=False,edgecolor=MID,lw=.6))
            fig.add_artist(ConnectionPatch(xyA=(1,(lo+hi)/2),coordsA=ax.transData,
                xyB=(0,.5),coordsB=inset.transAxes,arrowstyle='->',mutation_scale=9,
                linewidth=.75,color=MID,shrinkA=3,shrinkB=5,clip_on=False))
            ax.text(0,1.025,f'Full width {width:.3f} MMBtu/MWh',transform=ax.transAxes,va='bottom',fontsize=8,color=INK)
            ax.text(.025,.90,f'Minimum {block.coverage_90.min():.1%}',transform=ax.transAxes,fontsize=8.5,color=RED)
            ma=fig.add_subplot(gs[2,i]);rr=joint[joint.model.eq(model)].sort_values('year')
            ma.set(xlim=(-.43,2),ylim=(3,0));ma.axis('off')
            ma.set_title(f'{chr(99+i)}  {model} | 120 common facilities',loc='left',fontsize=9.2,weight='bold',pad=24)
            for x,title in [(0,'Operator equal'),(1,'Facility equal')]:ma.text(x+.5,-.18,title,ha='center',va='bottom',fontsize=8.5,color=INK)
            for y,r in enumerate(rr.itertuples()):
                values=[float(r.operator_equal_supported_and_covered),float(r.facility_equal_supported_and_covered)]
                ma.text(-.1,y+.5,str(int(r.year)),ha='right',va='center',fontsize=8.5)
                for x,v in enumerate(values):
                    ma.add_patch(Rectangle((x,y),1,1,facecolor=cmap(Normalize(0,1)(v)),edgecolor='white',lw=2))
                    ma.text(x+.5,y+.39 if x==0 else y+.5,f'{v:.1%}',ha='center',va='center',fontsize=10,color=BLUE,weight='bold')
                    if x==0:ma.text(x+.5,y+.72,f'[{r.operator_cluster_ci95_lower:.1%}, {r.operator_cluster_ci95_upper:.1%}]',ha='center',va='center',fontsize=7.8,color=INK)
                matrix.append(dict(model=model,year=int(r.year),operator_equal=values[0],facility_equal=values[1],
                    lower=float(r.operator_cluster_ci95_lower),upper=float(r.operator_cluster_ci95_upper),facilities=120))
            supported_coverage_tracks(ma,model,rr,chr(99+i))
        handles=[Patch(facecolor=c,label=s) for c,s in zip(CATS.values(),['<85%','85–95%','>95%'])]+[Line2D([],[],marker=m,ls='',color=GRAY,label=str(y)) for y,m in markers.items()]
        fig.legend(handles=handles,ncol=6,frameon=False,loc='upper center',bbox_to_anchor=(.53,.988),columnspacing=.9)
        fig.text(.10,.055,'c, d: supported-and-covered / eligible days; 95% CI: operator-equal estimates only.',fontsize=8.1,color=INK)
        fig.text(.10,.020,'Detail panels repeat high-coverage observations from the full annual cohorts.',fontsize=8.1,color=INK)
        self.export('combined_cycle_heat_rate',4,fig,artists,{'annual_proportions':matrix,'proportion_track_scale':[0,1],
            'variant':'A','zoom_pointer':'actual selected rectangle to adjacent detailed axes'})

    def heat_rate_4_c(self):
        'Ranked paired-metric tiles: every tile column is a real facility-year.'
        from matplotlib.colors import TwoSlopeNorm
        df=pd.read_csv(next((self.candidate/PAPERS['combined_cycle_heat_rate']/'figures').glob('FIG4_*/plotted_values.csv')))
        hero=df[df.record_type.eq('facility_interval_quality_full_annual_cohort')].copy()
        joint=df[df.record_type.eq('supported_and_covered_common_panel')].copy()
        validate_categories(hero)
        assert len(hero)==774 and not hero.duplicated(['facility_id','year','model']).any()
        fig=plt.figure(figsize=(178/25.4,6.35))
        grid=fig.add_gridspec(2,2,height_ratios=[3.1,1.25],left=.14,right=.975,top=.83,bottom=.13,hspace=.55,wspace=.29)
        coverage_map=LinearSegmentedColormap.from_list('coverage',['#B75955','#F6F7F8','#285B7E'])
        score_map=LinearSegmentedColormap.from_list('winkler',['#F5F0EF','#D6ABA6','#A94D49'])
        score_max=float(np.ceil(hero.mean_winkler_score.max()))
        artists=[];matrix=[];tiles=[]
        for i,model in enumerate(['M1','M2']):
            sub=grid[0,i].subgridspec(3,1,hspace=.55)
            for j,year in enumerate([2022,2023,2024]):
                rows=hero[hero.model.eq(model)&hero.year.eq(year)].sort_values(['coverage_90','mean_winkler_score','facility_id'])
                n=len(rows);ax=fig.add_subplot(sub[j]);edges=np.arange(n+1)
                ax.pcolormesh(edges,[0,1],rows.coverage_90.to_numpy()[None,:],cmap=coverage_map,norm=TwoSlopeNorm(vmin=0,vcenter=.90,vmax=1),rasterized=False)
                ax.pcolormesh(edges,[1.15,2.15],rows.mean_winkler_score.to_numpy()[None,:],cmap=score_map,vmin=0,vmax=score_max,rasterized=False)
                ax.set(xlim=(0,n),ylim=(2.15,0),yticks=[.5,1.65],yticklabels=['Coverage','Winkler'],xticks=[])
                ax.tick_params(axis='y',length=0,labelsize=7.7,pad=4)
                ax.spines[:].set_visible(False)
                ax.set_title(f'{year}  |  {n} facilities',loc='left',fontsize=8.8,pad=5)
                if j==0:
                    ax.text(0,1.72,f'{chr(97+i)}  {model}',transform=ax.transAxes,fontsize=12,weight='bold')
                if j==2:ax.set_xlabel('Increasing facility coverage within each year',fontsize=8,labelpad=7)
                for rank,r in enumerate(rows.itertuples()):
                    artists.append(dict(facility_id=int(r.facility_id),year=year,model=model,category=r.coverage_direction,x=float(r.coverage_90),y=float(r.mean_winkler_score)))
                    tiles.append(dict(facility_id=int(r.facility_id),year=year,model=model,column=rank,
                        coverage=float(r.coverage_90),winkler=float(r.mean_winkler_score)))
            ma=fig.add_subplot(grid[1,i]);ma.axis('off');ma.set(xlim=(-.40,2),ylim=(3,0))
            ma.set_title(f'{chr(99+i)}  {model} | 120 common facilities',loc='left',fontsize=9.0,weight='bold',pad=26)
            for x,title in [(0,'Operator equal'),(1,'Facility equal')]:ma.text(x+.5,-.20,title,ha='center',va='bottom',fontsize=8.1)
            for y,r in enumerate(joint[joint.model.eq(model)].sort_values('year').itertuples()):
                values=[float(r.operator_equal_supported_and_covered),float(r.facility_equal_supported_and_covered)]
                ma.text(-.08,y+.5,str(int(r.year)),ha='right',va='center',fontsize=8.1)
                for x,v in enumerate(values):
                    ma.add_patch(Rectangle((x,y),1,1,facecolor='#E6EDF2',edgecolor='white',lw=2))
                    ma.text(x+.5,y+.35 if x==0 else y+.5,f'{v:.1%}',ha='center',va='center',fontsize=10,color=BLUE,weight='bold')
                    if x==0:ma.text(x+.5,y+.73,f'[{r.operator_cluster_ci95_lower:.1%}, {r.operator_cluster_ci95_upper:.1%}]',ha='center',va='center',fontsize=7.3,color=INK)
                matrix.append(dict(model=model,year=int(r.year),operator_equal=values[0],facility_equal=values[1],lower=float(r.operator_cluster_ci95_lower),upper=float(r.operator_cluster_ci95_upper),facilities=120))
            proportion_bands(ma,model,joint[joint.model.eq(model)],chr(99+i))
        for left,cmap,norm,label,ticks in [(.14,coverage_map,TwoSlopeNorm(vmin=0,vcenter=.90,vmax=1),'Coverage (%)',[0,.90,1]),
            (.62,score_map,Normalize(0,score_max),'Winkler (MMBtu/MWh)',[0,5,10,15])]:
            ca=fig.add_axes([left,.945,.30,.014]);cb=fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm,cmap=cmap),cax=ca,orientation='horizontal',ticks=ticks)
            cb.ax.tick_params(labelsize=7,length=2,pad=1);cb.outline.set_visible(False);ca.set_title(label,fontsize=8,pad=5)
            if label.startswith('Coverage'):cb.ax.set_xticklabels(['0','90','100'])
        fig.text(.14,.065,'Each paired column represents one facility-year; ordering is independent by year.',fontsize=7.7,color=INK)
        fig.text(.14,.030,'c, d: supported-and-covered / all eligible days; brackets: operator-cluster 95% CI.',fontsize=7.7,color=INK)
        self.export('combined_cycle_heat_rate',4,fig,artists,{'variant':'C','annual_proportions':matrix,'facility_tiles':tiles,
            'coverage_normalization':{'minimum':0,'centre':.9,'maximum':1},'winkler_normalization':[0,score_max]})

    def heat_rate_5(self):
        df=pd.read_csv(next((self.candidate/PAPERS['combined_cycle_heat_rate']/'figures').glob('FIG5_*/plotted_values.csv')))
        common=df[df.record_type.eq('common_panel_facility')].copy();persistence=df[df.record_type.eq('bias_rank_persistence')]
        validate_categories(common)
        fig=plt.figure(figsize=(178/25.4,5.1))
        outer=fig.add_gridspec(2,2,height_ratios=[3.35,1.12],left=.10,right=.975,top=.90,bottom=.15,hspace=.35,wspace=.32)
        bias=common[common.year.isin([2022,2024])].signed_bias_mmbtu_per_mwh
        lim=(np.floor(bias.min()*5)/5-.1,np.ceil(bias.max()*5)/5+.1);bins=np.linspace(*lim,15);artists=[];categories=[];marginals=[]
        for i,model in enumerate(['M1','M2']):
            rr=common[common.model.eq(model)];pair=rr.pivot(index='facility_id',columns='year',values='signed_bias_mmbtu_per_mwh')
            assert len(pair)==120 and not pair.isna().any().any()
            inner=outer[0,i].subgridspec(2,2,width_ratios=[1,.17],height_ratios=[.17,1],hspace=.035,wspace=.035)
            ax=fig.add_subplot(inner[1,0]);top=fig.add_subplot(inner[0,0],sharex=ax);side=fig.add_subplot(inner[1,1],sharey=ax)
            points=ax.scatter(pair[2022],pair[2024],s=22,alpha=.75,color=BLUE,edgecolor='white',lw=.4)
            ax.plot(lim,lim,'--',color=GRAY,lw=.8);ax.axhline(0,color='#CCD4DA',lw=.6);ax.axvline(0,color='#CCD4DA',lw=.6)
            ax.set(xlim=lim,ylim=lim,xlabel='2022 bias (MMBtu/MWh)',ylabel='2024 bias (MMBtu/MWh)',aspect='equal')
            for h,values,orientation in [(top,pair[2022],'vertical'),(side,pair[2024],'horizontal')]:
                counts,edges,_=h.hist(values,bins=bins,orientation=orientation,color=MID,edgecolor='white',lw=.4)
                h.tick_params(labelbottom=False,labelleft=False,length=0);h.spines[:].set_visible(False)
                marginals.append(dict(model=model,year=2022 if orientation=='vertical' else 2024,edges=edges.tolist(),counts=counts.astype(int).tolist()))
            top.set_title(f'{chr(97+i)}  {model} | 120 paired facilities',loc='left',fontsize=9.3,weight='bold',pad=7)
            rho=persistence[persistence.model.eq(model)&persistence.year_a.eq(2022)&persistence.year_b.eq(2024)].spearman_rho.item()
            ax.text(.04,.95,f'Rank ρ = {rho:.3f}',transform=ax.transAxes,va='top',fontsize=8.5);clean(ax,None)
            fig.canvas.draw();pos=ax.get_position();tp=top.get_position();sp=side.get_position()
            top.set_position([pos.x0,tp.y0,pos.width,tp.height]);side.set_position([sp.x0,pos.y0,sp.width,pos.height])
            for fid,xy in zip(pair.index,points.get_offsets()):artists.append(dict(model=model,facility_id=int(fid),bias_2022=float(xy[0]),bias_2024=float(xy[1])))
            ca=fig.add_subplot(outer[1,i]);counts=rr.groupby(['year','coverage_direction']).size().unstack(fill_value=0).reindex(index=[2022,2023,2024],columns=list(CATS),fill_value=0)
            assert counts.sum(axis=1).eq(120).all()
            left=np.zeros(3);ys=np.array([2,1,0])
            for cat,color in CATS.items():
                vals=counts[cat].to_numpy();ca.barh(ys,vals,left=left,height=.40,color=color,edgecolor='white',lw=.4);left+=vals
            for y,year in zip(ys,[2022,2023,2024]):
                ca.text(123,y,' / '.join(str(int(v)) for v in counts.loc[year]),fontsize=8,va='center',color=INK)
                categories.append(dict(model=model,year=year,**{k:int(v) for k,v in counts.loc[year].items()}))
            ca.set(xlim=(0,174),ylim=(-.6,2.6),yticks=ys,yticklabels=['2022','2023','2024'],xticks=[])
            ca.set_title(f'{chr(99+i)}  Conditional coverage counts',loc='left',weight='bold',fontsize=9.0,pad=7)
            ca.spines[:].set_visible(False);ca.tick_params(length=0)
        fig.legend(handles=[Patch(facecolor=c,label=s) for c,s in zip(CATS.values(),['<85%','85–95%','>95%'])],
            title='Panels c, d',ncol=3,frameon=False,loc='lower center',bbox_to_anchor=(.57,.025),fontsize=8.3,title_fontsize=8.3,columnspacing=1.3)
        fig.text(.10,.010,'Dashed: equal bias. Marginals: facility counts. Category numbers follow legend order.',fontsize=8.0,color=INK)
        self.export('combined_cycle_heat_rate',5,fig,artists,{'coverage_counts':categories,'marginal_histograms':marginals})

    def heat_rate_4_b(self):
        'Focus-and-context alternative: full cohorts above linked detailed views.'
        from matplotlib.patches import ConnectionPatch
        df=pd.read_csv(next((self.candidate/PAPERS['combined_cycle_heat_rate']/'figures').glob('FIG4_*/plotted_values.csv')))
        hero=df[df.record_type.eq('facility_interval_quality_full_annual_cohort')].copy()
        joint=df[df.record_type.eq('supported_and_covered_common_panel')].copy()
        validate_categories(hero)
        assert len(hero)==774 and not hero.duplicated(['facility_id','year','model']).any()
        fig=plt.figure(figsize=(178/25.4,7.15))
        grid=fig.add_gridspec(3,2,height_ratios=[1.12,1.15,.72],left=.105,right=.975,bottom=.10,top=.89,hspace=.76,wspace=.28)
        markers={2022:'o',2023:'s',2024:'^'};artists=[];matrix=[]
        ymax=float(np.ceil(hero.mean_winkler_score.max()))
        zoommax=float(hero.loc[hero.coverage_90.ge(.85),'mean_winkler_score'].max())*1.06
        zoommin=float(hero.interval_width_mmbtu_per_mwh.min())*.96
        for i,model in enumerate(['M1','M2']):
            block=hero[hero.model.eq(model)];width=float(block.interval_width_mmbtu_per_mwh.iloc[0])
            context=fig.add_subplot(grid[0,i]);focus=fig.add_subplot(grid[1,i])
            for ax,is_zoom in [(context,False),(focus,True)]:
                ax.axvspan(.85,.95,color=PALE,zorder=-4)
                ax.axvline(.9,color=GRAY,lw=.7,ls='--');ax.axhline(width,color=GRAY,lw=.7,ls=':')
                for year,marker in markers.items():
                    for cat,color in CATS.items():
                        rr=block[block.year.eq(year)&block.coverage_direction.eq(cat)]
                        if is_zoom:rr=rr[rr.coverage_90.ge(.85)]
                        points=ax.scatter(rr.coverage_90,rr.mean_winkler_score,s=19 if is_zoom else 17,
                            color=color,marker=marker,alpha=.77,edgecolor='white',lw=.3,zorder=3)
                        if not is_zoom:
                            for r,xy in zip(rr.itertuples(),points.get_offsets()):
                                artists.append(dict(facility_id=int(r.facility_id),year=int(r.year),model=model,category=cat,x=float(xy[0]),y=float(xy[1])))
                clean(ax,'y');ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=0))
                if i==0:ax.set_ylabel('Mean Winkler score\n(MMBtu/MWh)')
            context.set(xlim=(0,1.01),ylim=(0,ymax),xticks=[0,.5,1])
            context.set_title(f'{chr(97+i)}  {model}',loc='left',fontsize=12,weight='bold',pad=18)
            context.text(1,1.12,'387 facility-years',transform=context.transAxes,ha='right',fontsize=8.8,color=INK)
            context.text(0,1.025,f'Fixed full width: {width:.3f} MMBtu/MWh',transform=context.transAxes,va='bottom',fontsize=7.8,color=INK)
            context.text(.02,.95,f'Minimum coverage {block.coverage_90.min():.1%}',transform=context.transAxes,va='top',fontsize=8.2,color=RED)
            focus.set(xlim=(.845,1.005),ylim=(zoommin,zoommax),xticks=[.85,.90,.95,1])
            focus.set_title('High-coverage region',loc='left',fontsize=9,weight='bold',pad=8)
            focus.set_xlabel('Coverage on matched-support days')
            context.add_patch(Rectangle((.85,zoommin),.15,zoommax-zoommin,fill=False,lw=.7,edgecolor=MID))
            # Connect the actual selected rectangle to the corresponding enlarged axes.
            connector=ConnectionPatch(xyA=(1,zoommin),coordsA=context.transData,
                xyB=(1,zoommax),coordsB=focus.transData,linewidth=.65,color=MID,alpha=.7,zorder=0)
            fig.add_artist(connector)
            ma=fig.add_subplot(grid[2,i]);ma.axis('off');ma.set(xlim=(-.43,2),ylim=(3,0))
            ma.set_title(f'{chr(99+i)}  120 common facilities',loc='left',fontsize=9.5,weight='bold',pad=22)
            for x,title in [(0,'Operator equal'),(1,'Facility equal')]:ma.text(x+.5,-.16,title,ha='center',va='bottom',fontsize=8.4)
            for y,r in enumerate(joint[joint.model.eq(model)].sort_values('year').itertuples()):
                values=[float(r.operator_equal_supported_and_covered),float(r.facility_equal_supported_and_covered)]
                ma.text(-.1,y+.5,str(int(r.year)),ha='right',va='center',fontsize=8.5)
                for x,v in enumerate(values):
                    ma.add_patch(Rectangle((x,y),1,1,facecolor='#E6EDF2' if y!=1 else '#DCE7EE',edgecolor='white',lw=2))
                    ma.text(x+.5,y+.36 if x==0 else y+.5,f'{v:.1%}',ha='center',va='center',fontsize=10,color=BLUE,weight='bold')
                    if x==0:ma.text(x+.5,y+.73,f'[{r.operator_cluster_ci95_lower:.1%}, {r.operator_cluster_ci95_upper:.1%}]',ha='center',va='center',fontsize=7.7,color=INK)
                matrix.append(dict(model=model,year=int(r.year),operator_equal=values[0],facility_equal=values[1],
                    lower=float(r.operator_cluster_ci95_lower),upper=float(r.operator_cluster_ci95_upper),facilities=120))
        handles=[Patch(facecolor=c,label=s) for c,s in zip(CATS.values(),['<85%','85–95%','>95%'])]+[Line2D([],[],marker=m,ls='',color=GRAY,label=str(y)) for y,m in markers.items()]
        fig.legend(handles=handles,ncol=6,frameon=False,loc='upper center',bbox_to_anchor=(.54,.992),columnspacing=.9)
        fig.text(.105,.050,'c, d: supported-and-covered / all eligible days; brackets: operator-cluster 95% CI.',fontsize=8.0,color=INK)
        fig.text(.105,.016,'Detailed views repeat the outlined region. Matrix shading separates rows only.',fontsize=8.0,color=INK)
        self.export('combined_cycle_heat_rate',4,fig,artists,{'annual_proportions':matrix,'variant':'B','matrix_shading':'alternating year rows; no quantitative encoding'})

    def run(self):
        for paper,numbers in [('return_temperature_comparability',[2,3,4]),('combined_cycle_heat_rate',[2,3,4,5])]:
            for n in numbers:
                if self.wanted(paper,n):
                    if self.followup is not None and paper=='combined_cycle_heat_rate':
                        from followup_presentation import render_fig4,render_fig5
                        if n==4:render_fig4(self,self.followup)
                        elif n==5:render_fig5(self,self.followup,self.fig5_variant)
                        continue
                    method=('return_temperature' if paper=='return_temperature_comparability' else 'heat_rate')+'_'+str(n)
                    if paper=='combined_cycle_heat_rate' and n==4 and self.fig4_variant=='B':method+='_b'
                    if paper=='combined_cycle_heat_rate' and n==4 and self.fig4_variant=='C':method+='_c'
                    getattr(self,method)()
        if self.emit_retrospective_si:
            from followup_presentation import render_retrospective_si
            render_retrospective_si(self,self.followup)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--candidate',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--package-root',type=Path,help='Package directory containing the declared source values and figure metadata')
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--figures',nargs='*',help='Optional subset of figure keys defined by the package figure metadata')
    p.add_argument('--heat_rate-fig4-variant',choices=['A','B','C'],default='A')
    p.add_argument('--followup-input-manifest',type=Path,help='Manifest identifying the primary and subsequent-year result inputs')
    p.add_argument('--heat_rate-fig5-variant',choices=['legacy','parallel','followup_first'],default='legacy')
    p.add_argument('--emit-retrospective-si',action='store_true')
    args=p.parse_args();followup=None
    if args.followup_input_manifest:
        from followup_presentation import FollowupInputs
        followup=FollowupInputs(args.followup_input_manifest)
    configure_style();b=Builder(args.candidate,args.output_root,args.figures,args.heat_rate_fig4_variant,followup,args.heat_rate_fig5_variant,args.emit_retrospective_si,args.package_root);b.run()
    args.output_root.mkdir(parents=True,exist_ok=True)
    (args.output_root/'BUILD_RECORD.json').write_text(json.dumps(b.records,indent=2)+'\n')
    print(json.dumps(b.records,indent=2))

if __name__=='__main__':main()

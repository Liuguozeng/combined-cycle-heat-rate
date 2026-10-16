"""Historical-level errors and plant-specific response coefficients."""
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch

BLUE, PINK, INK, GRAY = '#2879AD', '#C97E9C', '#26333D', '#7C8792'
NAMES = ['historical_mean', 'historical_median', 'original_centered',
         'within_source_centered', 'local_response']
LABELS = ['Mean', 'Median', 'Original\ncentered', 'Within-\nsource', 'Local\ncalibration']


def response_figure(summary, coefficients):
    data=summary[summary.scope.eq('main')].copy()
    assert len(data)==10 and not data.duplicated(['model','candidate']).any()
    assert set(data.candidate)==set(NAMES) and set(data.model)=={'M1','M2'}
    assert coefficients.groupby('model').size().to_dict()=={'M1':120,'M2':120}
    assert (coefficients.history_year==2024).all()
    assert not coefficients.duplicated(['model','facility_id']).any()
    try:
        font_manager.findfont('Times New Roman',fallback_to_default=False)
        font_family='Times New Roman'
    except ValueError:
        font_family='DejaVu Serif'
    plt.rcParams.update({'font.family':font_family,'font.size':8.5,'axes.labelsize':9,
        'xtick.labelsize':8.5,'ytick.labelsize':8.5,'axes.linewidth':0.65,
        'text.color':INK,'axes.labelcolor':INK,'axes.edgecolor':INK,'xtick.color':INK,'ytick.color':INK,
        'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})
    fig=plt.figure(figsize=(178/25.4,106/25.4))
    a=fig.add_axes([0.105,0.235,0.55,0.575])
    b=fig.add_axes([0.755,0.235,0.225,0.575])
    colors={'M1':BLUE,'M2':PINK}
    markers={'M1':'o','M2':'D'}
    positions=[0,0.9,2.1,3.15,4.2]
    for k,candidate in enumerate(NAMES):
        for model in ['M1','M2']:
            if k<2 and model=='M2':
                continue
            r=data[(data.candidate==candidate)&(data.model==model)].iloc[0]
            assert r.mae_ci95_lower<=r.mae<=r.mae_ci95_upper
            if k<2:
                control=data[data.candidate.eq(candidate)]
                assert all(control[c].nunique()==1 for c in ['mae','mae_ci95_lower','mae_ci95_upper'])
            x=positions[k]+(0 if k<2 else (-0.15 if model=='M1' else 0.15))
            color=GRAY if k<2 else colors[model]
            a.bar(x,r.mae,width=0.36 if k<2 else 0.26,color=color,edgecolor=color,linewidth=0.5,
                  yerr=[[r.mae-r.mae_ci95_lower],[r.mae_ci95_upper-r.mae]],zorder=3,
                  error_kw=dict(ecolor=INK,elinewidth=0.75,capsize=2,capthick=0.7))
    a.set(xlim=(-0.5,4.7),ylim=(0,0.32),xticks=positions,xticklabels=LABELS,
          yticks=[0,0.1,0.2,0.3],ylabel='Facility-equal MAE (MMBtu/MWh)')
    a.grid(axis='y',lw=0.45,color='#E3E7EA',zorder=0)
    a.axvline(1.5,color='#D5DCE1',lw=0.6,zorder=0)
    a.tick_params(axis='x',length=0,pad=8)
    a.legend(handles=[Patch(facecolor=colors[m],edgecolor='none',label=m) for m in ['M1','M2']],
             loc='upper center',bbox_to_anchor=(0.5,-0.25),ncol=2,frameon=False,fontsize=8.5,
             handlelength=1.2,columnspacing=1.8,borderaxespad=0)
    for center,label in [(0.45,'Historical level'),(3.15,'Condition response')]:
        fx=0.105+0.55*(center+0.5)/5.2
        fig.text(fx,0.865,label,ha='center',fontsize=9,fontweight='bold')
    fig.text(0.8675,0.865,'Historical calibration',ha='center',fontsize=9,fontweight='bold')
    b.set(xlim=(-0.75,1.12),ylim=(-0.48,1.65),xticks=[-0.5,0,0.5,1],
          xlabel='Response coefficient',yticks=[0,1],yticklabels=['M2','M1'])
    b.axvline(0,lw=0.65,color=GRAY,zorder=0)
    b.axvline(1,lw=0.65,color=GRAY,ls=(0,(2,2)),zorder=0)
    b.spines['left'].set_visible(False)
    for model in ['M1','M2']:
        selected=coefficients[coefficients.model.eq(model)].sort_values(['response_coefficient','facility_id'])
        x=selected.response_coefficient.to_numpy()
        assert np.isfinite(x).all() and x.min()>-0.75 and x.max()<1.12
        base=1 if model=='M1' else 0
        offsets=np.array([int(hashlib.sha256(str(r.facility_id).encode()).hexdigest()[:8],16)/0xffffffff
                          for r in selected.itertuples()])
        b.scatter(x,base+(offsets-0.5)*0.48,s=8.0,color=colors[model],alpha=0.76,
                  marker=markers[model],edgecolors='white',linewidths=0.22,zorder=2)
        q1,median,q3=np.quantile(x,[0.25,0.5,0.75],method='linear')
        sy=base+0.33
        b.plot([x.min(),x.max()],[sy,sy],lw=0.55,color=colors[model],alpha=0.8,zorder=3)
        b.plot([q1,q3],[sy,sy],lw=3,color=colors[model],solid_capstyle='butt',zorder=4)
        b.plot([median,median],[sy-0.075,sy+0.075],lw=1.1,color=INK,zorder=5)
        b.text(-0.68,base+0.52,f'Median {median:.3f}',color=colors[model],fontsize=8.5,
               va='center',bbox=dict(facecolor='white',edgecolor='none',pad=1.2),zorder=6)
    for ax in [a,b]:
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(width=0.6,length=2.8)
    a.tick_params(axis='x',length=0)
    b.tick_params(axis='y',length=0,pad=5)
    for x,letter,title in [(0.035,'a','2025 prediction error'),(0.725,'b','2024 response scale')]:
        fig.text(x,0.935,letter,fontweight='bold',fontsize=11,va='baseline')
        fig.text(x+0.033,0.935,title,fontsize=10,va='baseline')
    fig.text(0.755,0.085,'0  historical level only',fontsize=8.5)
    fig.text(0.755,0.044,'1  full shared response',fontsize=8.5)
    fig.canvas.draw()
    return fig

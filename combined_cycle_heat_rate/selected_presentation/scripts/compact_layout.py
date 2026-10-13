'Selected compact A: all evidence retained in the main figures.'
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from panel_geometry import panel_label, above_frame, write_panel_geometry
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Patch, Rectangle
import numpy as np
import pandas as pd
import followup_presentation as h
def save(fig, stem, records, metadata):
    write_panel_geometry(fig, stem)
    fig.canvas.draw()
    texts = [t for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text()]
    fonts = [t.get_fontsize() for t in texts]
    outside = []
    hidden = set()
    for ax in fig.axes:
        for axis in [ax.xaxis, ax.yaxis]:
            low, high = sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if not low <= tick.get_loc() <= high:
                    hidden.update([tick.label1, tick.label2])
    for t in texts:
        if t in hidden:
            continue
        b = t.get_window_extent()
        if b.x0 < -2 or b.y0 < -2 or b.x1 > fig.bbox.x1 + 2 or b.y1 > fig.bbox.y1 + 2:
            outside.append(t.get_text())
    if outside:
        raise ValueError(f'Canvas overflow: {outside}')
    if min(fonts) < 8:
        raise ValueError('Font smaller than 8 pt')
    for suffix in ['pdf', 'png']:
        fig.savefig(stem.with_suffix('.' + suffix), dpi=250, facecolor='white')
    pd.DataFrame(records).to_csv(stem.with_name(stem.name + '_values.csv'), index=False)
    stem.with_name(stem.name + '_contract.json').write_text(json.dumps(
        dict(**metadata, minimum_text_pt=min(fonts), canvas_mm=(fig.get_size_inches()*25.4).tolist(),
             canvas_overflow=outside, palette=[h.BLUE, h.RED], values_recomputed=False,
             formal_choice_replaced=False), indent=2) + '\n')
    plt.close(fig)

def proportions(ax, rows, model, letter):
    ax.axis('off')
    ax.set(xlim=(-.12, 2.05), ylim=(7.6, 0))
    panel_label(ax, letter, model + ' | Supported-and-covered', title_size=8.5, content_bounds=(0, 0, 1, 1))
    ax.text(0, 0.97, 'Original common panel: 120 facilities', transform=ax.transAxes, fontsize=8)
    ax.text(0, 0.87, 'Dark: operator equal; pale: facility equal', transform=ax.transAxes, fontsize=8, color=h.INK)
    ax.text(1.24, 1.90, 'Mean', fontsize=8, ha='center', va='center')
    ax.text(1.72, 1.90, '95% CI', fontsize=8, ha='center', va='center')
    out = []
    for j, r in enumerate(rows.sort_values('year').itertuples()):
        y = 2.8 + 1.5*j
        ax.text(-.06, y+.38, str(int(r.year)), ha='right', va='center', fontsize=8)
        op, fac = float(r.operator_equal_supported_and_covered), float(r.facility_equal_supported_and_covered)
        lo, hi = float(r.operator_cluster_ci95_lower), float(r.operator_cluster_ci95_upper)
        for yy, val, col in [(y, op, h.BLUE), (y+.76, fac, h.MID)]:
            ax.add_patch(Rectangle((0, yy-.19), 1, .38, facecolor=h.PALE))
            ax.add_patch(Rectangle((0, yy-.19), val, .38, facecolor=col))
            ax.text(1.24, yy, f'{val:.1%}', va='center', ha='center', fontsize=8, color=h.INK)
        ax.add_patch(Rectangle((lo, y-.19), hi-lo, .38, facecolor='white', edgecolor=h.INK, linewidth=.6))
        ax.plot([op, op], [y-.19, y+.19], color=h.INK, lw=.8)
        ax.text(1.72, y, f'[{lo:.1%}, {hi:.1%}]', fontsize=8, ha='center', va='center')
        ax.text(1.72, y+.76, '--', fontsize=8, ha='center', va='center', color=h.GRAY)
        out.append(dict(model=model, year=int(r.year), operator_equal=op, facility_equal=fac, lower=lo, upper=hi))
    for x in [0, .5, 1]:
        ax.text(x, 7.3, f'{x:.0%}', ha='center', va='center', fontsize=8)
    return out

def coverage(inputs, variant, out, ymax):
    full = variant == 'A'
    fig = plt.figure(figsize=(178/25.4, (210 if full else 190)/25.4))
    g = fig.add_gridspec(4 if full else 3, 2, height_ratios=[2.1, 2.1, 2.45, 1.85] if full else [2.1, 2.1, 1.85],
                        left=.105, right=.975, bottom=.09, top=.885, hspace=.70 if full else .72, wspace=.40)
    artists, matrix = [], []
    for i, model in enumerate(['M1', 'M2']):
        sg = g[i, :].subgridspec(1, 2, width_ratios=[1.45, 1], wspace=.38)
        ax, detail = fig.add_subplot(sg[0]), fig.add_subplot(sg[1])
        b = inputs.hero[inputs.hero.model == model]
        width = h.scatter_quality(ax, b, model, 'primary_GSOD_2022_2024', (0, ymax), artists)
        h.scatter_quality(detail, b, model, 'primary_GSOD_2022_2024', (0, ymax), artists, zoom=True)
        tail = b[b.coverage_90 >= .85]
        detail.set(xlim=(.85, 1.005), ylim=(width*.96, tail.mean_winkler_score.max()*1.06), xticks=[.85, .90, .95, 1])
        panel_label(ax, chr(97+i), model + ' | 2022-2024', title_size=9.3)
        above_frame(ax, f'387 facility-years; width {width:.3f}')
        detail.set_title('85-100% coverage detail', fontsize=8.8, pad=12)
        ax.text(.02, .90, f'Minimum {b.coverage_90.min():.1%}', transform=ax.transAxes, fontsize=8.3)
        ax.set_ylabel('Mean Winkler score\n(MMBtu/MWh)', fontsize=8.5)
        ax.set_xlabel('In-support interval coverage', fontsize=8.2)
        detail.set_xlabel('In-support interval coverage', fontsize=8.2)
        lo, hi = detail.get_ylim()
        ax.add_patch(Rectangle((.85, lo), .15, hi-lo, fill=False, edgecolor=h.MID, lw=.65))
        fig.add_artist(ConnectionPatch(xyA=(1, (lo+hi)/2), coordsA=ax.transData, xyB=(0, .5),
                                      coordsB=detail.transAxes, arrowstyle='->', mutation_scale=8, color=h.MID, lw=.7))
    if full:
        for i, model in enumerate(['M1', 'M2']):
            matrix += proportions(fig.add_subplot(g[2, i]), inputs.joint[inputs.joint.model == model], model, chr(99+i))
    for i, model in enumerate(['M1', 'M2']):
        ax = fig.add_subplot(g[-1, i])
        b = inputs.follow[inputs.follow.model == model]
        h.scatter_quality(ax, b, model, 'followup_2025', (0, ymax), artists)
        panel_label(ax, chr((101 if full else 99)+i), model + ' | 2025 follow-up')
        lowest = b.loc[b.coverage_90.idxmin()]
        ax.text(.02, .90, f'121 facilities; {lowest.facility_name}: {lowest.coverage_90:.1%}',
                transform=ax.transAxes, fontsize=8)
        ax.set_ylabel('Mean Winkler score\n(MMBtu/MWh)', fontsize=8.3)
        ax.set_xlabel('In-support interval coverage', fontsize=8.2)
    fig.text(.105,.974,'Coverage class',fontsize=8,color=h.INK)
    fig.legend(handles=[Patch(facecolor=c,label=s) for c,s in zip(h.CATS.values(),
               ['<85% undercoverage','85-95% band','>95% overcoverage'])],
               ncol=3,loc='upper center',bbox_to_anchor=(.64,.992),frameon=False,fontsize=8,columnspacing=1.0)
    fig.text(.105,.948,'Year',fontsize=8,color=h.INK)
    fig.legend(handles=[Line2D([],[],marker=m,ls='',color=h.GRAY,label=str(y))
               for y,m in [(2022,'o'),(2023,'s'),(2024,'^'),(2025,'D')]],
               ncol=4,loc='upper center',bbox_to_anchor=(.56,.963),frameon=False,fontsize=8,columnspacing=1.0)
    fig.text(.105,.014,'Shading: 85-95%; dashed: nominal 90%; dotted: full interval width.', fontsize=8, color=h.INK)
    assert len(artists) == 1016
    assert len({(r['model'],r['year'],r['facility_id']) for r in artists}) == 1016
    save(fig, out / f'Figure_5_preview_{variant}', artists, dict(variant=variant, original_objects=774, followup_objects=242,
         common_full_score_limits=[0,ymax], annual_proportions=matrix, input_identities=inputs.identities))

def followup(inputs, variant, out, limits, bins):
    categories = variant == 'A'
    fig = plt.figure(figsize=(178/25.4, (153 if categories else 122)/25.4))
    g = fig.add_gridspec(3 if categories else 2, 2,
                        height_ratios=[3.3,.56,1] if categories else [3.3,.56],
                        left=.105,right=.975,top=.89,bottom=.15 if categories else .13,hspace=.51,wspace=.40)
    artists, histograms, counts = [], [], []
    for i,model in enumerate(['M1','M2']):
        h.paired_panel(fig,g[0,i],inputs,model,'followup',chr(97+i),limits,bins,artists,histograms,
                       separate_interval_summary=True)
        summary = h.followup_summary(inputs,model)
        ax = fig.add_subplot(g[1,i])
        ax.axis('off')
        ax.text(0,1,'Interval containment | same follow-up cohort',fontsize=8.2,weight='bold',va='top')
        ax.text(0,.51,f"Coverage rank ρ = {summary['coverage_spearman']:.3f}",fontsize=8.2,va='top')
        ax.text(0,.01,f"Same category: {summary['same_coverage_category']}/120 facilities",fontsize=8.2,va='top')
    if categories:
        for i,model in enumerate(['M1','M2']):
            counts += h.category_panel(fig.add_subplot(g[-1,i]),inputs,model,chr(99+i))
        fig.legend(handles=[Patch(facecolor=c,label=s) for c,s in zip(h.CATS.values(),['<85%','85-95%','>95%'])],
                   title='c, d: annual composition of the original 120-facility panel',ncol=3,loc='lower center',
                   bbox_to_anchor=(.54,.02),frameon=False,fontsize=8,title_fontsize=8)
    else:
        fig.text(.105,.035,'120 paired facilities per model; original three-year panel is separate.',fontsize=8)
    fig.text(.105,.010,'Separate follow-up and original three-year cohorts; intersection: 106 facilities.',fontsize=8,color=h.INK)
    assert len(artists) == 240
    assert all(sum(x['counts']) == 120 for x in histograms)
    save(fig,out/f'Figure_6_preview_{variant}',artists,dict(variant=variant,pair_points=240,bias_limits=list(limits),
         bins=bins.tolist(),histograms=histograms,original_categories=counts,panel_overlap=106,input_identities=inputs.identities))

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    plt.rcdefaults()
    plt.rcParams.update({'font.family':'Times New Roman','font.size':8.5,'text.color':h.INK,
        'axes.labelcolor':h.INK,'axes.edgecolor':h.INK,'axes.linewidth':.65,'pdf.fonttype':42,'ps.fonttype':42})
    inputs = h.FollowupInputs(a.input_manifest)
    ymax = float(np.ceil(max(inputs.hero.mean_winkler_score.max(), inputs.follow.mean_winkler_score.max())))+.25
    vals = np.concatenate([h.paired_data(inputs,m,b)[0].to_numpy().ravel()
                           for m in ['M1','M2'] for b in ['retrospective','followup']])
    limits = (float(np.floor(vals.min()*5)/5-.1),float(np.ceil(vals.max()*5)/5+.1))
    bins = np.linspace(*limits,15)
    coverage(inputs,'A',a.output,ymax)
    followup(inputs,'A',a.output,limits,bins)
    for number in (5,6):
        for suffix in ('pdf','png'):
            old = a.output / f'Figure_{number}_preview_A.{suffix}'
            old.rename(a.output / f'Figure_{number}.{suffix}')

if __name__ == '__main__':
    main()

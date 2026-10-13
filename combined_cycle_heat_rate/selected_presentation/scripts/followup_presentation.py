'Hash-bound presentation of existing primary and subsequent-year results.'
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from panel_geometry import panel_label, write_panel_geometry

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Patch, Rectangle
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

BLUE, MID, RED, GRAY, INK = '#2879AD', '#91B9D0', '#C97E9C', '#7C8792', '#26333D'
PALE = '#E5EBEF'
CATS = {'undercoverage': RED, 'nominal_band': GRAY, 'overcoverage': BLUE}
REQUIRED = {'primary_fig4', 'primary_fig5', 'followup_metrics', 'followup_pairs', 'followup_persistence'}
ROLES = {'primary_fig4': 'primary_GSOD_2022_2024', 'primary_fig5': 'primary_GSOD_2022_2024',
         'followup_metrics': 'weather_bridge_2024_and_followup_2025',
         'followup_pairs': 'weather_bridge_2024_to_followup_2025',
         'followup_persistence': 'weather_bridge_2024_to_followup_2025'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_categories(frame):
    if frame[['coverage_90', 'mean_winkler_score']].isna().any().any():
        raise ValueError('Missing plotted metric')
    if not frame.coverage_90.between(0, 1).all():
        raise ValueError('Coverage outside [0, 1]')
    actual = np.where(frame.coverage_90 < .85, 'undercoverage',
                      np.where(frame.coverage_90 > .95, 'overcoverage', 'nominal_band'))
    if not np.array_equal(actual, frame.coverage_direction):
        raise ValueError('Unknown or inconsistent coverage category')


class FollowupInputs:
    def __init__(self, manifest):
        self.manifest = Path(manifest).resolve()
        cfg = json.loads(self.manifest.read_text())
        if cfg.get('schema_version') != 1 or set(cfg.get('inputs', {})) != REQUIRED:
            raise ValueError('Exactly the declared five inputs are required')
        self.frames, self.identities = {}, {}
        for name, item in cfg['inputs'].items():
            if item.get('role') != ROLES[name]:
                raise ValueError(f"Wrong analysis branch: {name}")
            path = Path(item['path'])
            if not path.is_absolute():
                path = self.manifest.parent / path
            if not path.is_file():
                raise FileNotFoundError(f"Required input missing: {name}")
            if sha(path) != item['sha256']:
                raise ValueError(f"Input hash mismatch: {name}")
            self.frames[name] = pd.read_csv(path)
            self.identities[name] = dict(sha256=item['sha256'], role=item['role'], rows=len(self.frames[name]))
        self.hero = self.frames['primary_fig4'].query("record_type == 'facility_interval_quality_full_annual_cohort'").copy()
        self.joint = self.frames['primary_fig4'].query("record_type == 'supported_and_covered_common_panel'").copy()
        self.common = self.frames['primary_fig5'].query("record_type == 'common_panel_facility'").copy()
        self.primary_rho = self.frames['primary_fig5'].query("record_type == 'bias_rank_persistence'").copy()
        self.metrics = self.frames['followup_metrics']
        self.pairs = self.frames['followup_pairs']
        self.rho = self.frames['followup_persistence']
        self.follow = self.metrics.query('year == 2025').copy()
        self.validate()

    def validate(self):
        for name, frame in [('primary', self.hero), ('common', self.common), ('followup', self.metrics)]:
            if frame.duplicated(['facility_id', 'year', 'model']).any():
                raise ValueError(f"Duplicate identity: {name}")
            require_categories(frame)
            if set(frame.model) != {'M1', 'M2'}:
                raise ValueError(f"Wrong models: {name}")
            if not np.isfinite(frame[['signed_bias_mmbtu_per_mwh', 'mean_winkler_score']]).all().all():
                raise ValueError(f"Nonfinite plotted value: {name}")
        if set(self.hero.year) != {2022, 2023, 2024} or set(self.common.year) != {2022, 2023, 2024}:
            raise ValueError('Wrong primary years')
        if set(self.metrics.year) != {2024, 2025}:
            raise ValueError('Wrong followup years')
        expected = {2022: 124, 2023: 128, 2024: 135}
        for (model, year), frame in self.hero.groupby(['model', 'year']):
            if len(frame) != expected[year]:
                raise ValueError('Primary object omitted or added')
        if len(self.follow) != 242 or not self.follow.groupby('model').size().eq(121).all():
            raise ValueError('Followup object omitted or added')
        if self.pairs.duplicated(['facility_id', 'model']).any() or not self.pairs.groupby('model').size().eq(120).all():
            raise ValueError('Invalid paired identities')
        if set(self.pairs.year_2024) != {2024} or set(self.pairs.year_2025) != {2025}:
            raise ValueError('Invalid paired years')
        old_ids = None
        new_ids = None
        for model in ['M1', 'M2']:
            blocks = [set(g.facility_id) for _, g in self.common[self.common.model.eq(model)].groupby('year')]
            if len(blocks) != 3 or any(len(s) != 120 or s != blocks[0] for s in blocks):
                raise ValueError('Three-year common-panel membership changed')
            current = set(self.pairs.loc[self.pairs.model.eq(model), 'facility_id'])
            if old_ids is not None and (old_ids != blocks[0] or new_ids != current):
                raise ValueError('Models do not share the declared facility panels')
            old_ids, new_ids = blocks[0], current
            rows = self.pairs[self.pairs.model.eq(model)]
            for year in [2024, 2025]:
                source = self.metrics[self.metrics.model.eq(model) & self.metrics.year.eq(year)].set_index('facility_id')
                for metric in ['signed_bias_mmbtu_per_mwh', 'coverage_90', 'mean_winkler_score']:
                    if not np.array_equal(rows[f"{metric}_{year}"].to_numpy(), source.loc[rows.facility_id, metric].to_numpy()):
                        raise ValueError('Pairs disagree with the branch-specific metrics')
            actual = rows.signed_bias_mmbtu_per_mwh_2024.corr(rows.signed_bias_mmbtu_per_mwh_2025, method='spearman')
            reported = self.rho.loc[self.rho.model.eq(model), 'signed_bias_spearman'].item()
            if abs(actual - reported) > 1e-12:
                raise ValueError('Pairwise rank summary disagrees')
        if len(old_ids & new_ids) != 106:
            raise ValueError('Panel overlap differs from the declared result')
        self.panel_overlap = len(old_ids & new_ids)
        if self.joint.duplicated(['model', 'year']).any():
            raise ValueError('Duplicate aggregate rows')
        if len(self.joint) != 6:
            raise ValueError('Missing three-year aggregate')


def clean(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#E5EBEF', linewidth=.45)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=.6, labelsize=8.2)

def followup_summary(inputs, model):
    rows=inputs.pairs[inputs.pairs.model.eq(model)]
    ref=inputs.rho[inputs.rho.model.eq(model)].iloc[0]
    bias=float(rows.signed_bias_mmbtu_per_mwh_2024.corr(rows.signed_bias_mmbtu_per_mwh_2025,method='spearman'))
    coverage=float(rows.coverage_90_2024.corr(rows.coverage_90_2025,method='spearman'))
    same=int(rows.coverage_direction_2024.eq(rows.coverage_direction_2025).sum())
    if abs(coverage-float(ref.coverage_spearman))>1e-12:
        raise ValueError('Coverage rank summary disagrees with paired records')
    if same!=int(ref.same_coverage_category):
        raise ValueError('Same-category count disagrees with paired records')
    if abs(bias-float(ref.signed_bias_spearman))>1e-12 or len(rows)!=int(ref.common_facilities):
        raise ValueError('Signed-bias summary or denominator disagrees')
    return dict(model=model,facilities=len(rows),signed_bias_spearman=bias,
                coverage_spearman=coverage,same_coverage_category=same)


def export(builder, number, variant, fig, artists, supporting, inputs):
    folder = builder.output / 'combined_cycle_heat_rate' / f"FIG{number}_2025_CORE_{variant}"
    folder.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    overflow, hidden_ticks = [], set()
    for ax in fig.axes:
        for axis in [ax.xaxis, ax.yaxis]:
            low, high = sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if not low <= tick.get_loc() <= high:
                    hidden_ticks.update([tick.label1, tick.label2])
    for t in fig.findobj(matplotlib.text.Text):
        if t in hidden_ticks or not t.get_visible() or not t.get_text():
            continue
        box = t.get_window_extent()
        if box.x0 < -2 or box.y0 < -2 or box.x1 > fig.bbox.x1 + 2 or box.y1 > fig.bbox.y1 + 2:
            overflow.append(t.get_text())
    fonts = sorted({float(t.get_fontsize()) for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text()})
    base = folder / f"Figure_{number}_{variant}"
    write_panel_geometry(fig, base)
    for suffix in ['pdf', 'svg', 'png']:
        fig.savefig(base.with_suffix('.' + suffix), dpi=300, facecolor='white')
    presentation = dict(variant=variant, figure=number, results_recomputed=False,
                        input_identities=inputs.identities, minimum_text_points=min(fonts),
                        canvas_mm=[float(x*25.4) for x in fig.get_size_inches()], **supporting)
    (folder / 'ARTIST_DATA.json').write_text(json.dumps(artists, indent=2, allow_nan=False) + '\n')
    (folder / 'PRESENTATION_DATA.json').write_text(json.dumps(presentation, indent=2, allow_nan=False) + '\n')
    pd.DataFrame(artists).to_csv(folder / 'plotted_values.csv', index=False)
    record = dict(paper='combined_cycle_heat_rate', figure=number, variant=variant, folder=str(folder),
                  canvas_overflow=overflow, objects=len(artists), minimum_text_points=min(fonts))
    (folder / 'BUILD_RECORD.json').write_text(json.dumps(record, indent=2) + '\n')
    builder.records.append(record)
    plt.close(fig)


def scatter_quality(ax, frame, model, branch, ylim, artists, zoom=False):
    ax.axvspan(.85, .95, color=PALE, zorder=-3)
    ax.axvline(.90, ls='--', color=GRAY, lw=.7)
    width = frame.interval_width_mmbtu_per_mwh.iloc[0]
    ax.axhline(width, ls=':', color=GRAY, lw=.7)
    for year, marker in {2022: 'o', 2023: 's', 2024: '^', 2025: 'D'}.items():
        for cat, color in CATS.items():
            rows = frame[frame.year.eq(year) & frame.coverage_direction.eq(cat)]
            if zoom:
                rows = rows[rows.coverage_90.ge(.85)]
            dots = ax.scatter(rows.coverage_90, rows.mean_winkler_score, s=17 if branch.startswith('primary') else 21,
                              marker=marker, color=color, alpha=.78, edgecolor='white', lw=.25)
            if not zoom:
                for r, xy in zip(rows.itertuples(), dots.get_offsets()):
                    artists.append(dict(data_branch=branch, facility_id=int(r.facility_id), year=int(r.year),
                                        model=model, category=cat, coverage=float(xy[0]), winkler=float(xy[1])))
    ax.set(xlim=(0, 1.01), ylim=ylim, xticks=[0, .2, .4, .6, .8, 1])
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    clean(ax)
    return float(width)


def proportion_panel(ax, rows, model, letter):
    ax.axis('off')
    ax.set(xlim=(-.17, 1.55), ylim=(4.05, 0))
    ax.set_title(f"{letter}  {model} | Supported-and-covered", loc='left', fontsize=9.2, weight='bold', pad=24)
    ax.text(0, -.68, '2022-2024 common panel: 120 facilities', fontsize=8.2, color=INK, clip_on=False)
    for x, label, color in [(0, 'Operator equal', BLUE), (.84, 'Facility equal', MID)]:
        ax.add_patch(Rectangle((x, -.35), .055, .12, facecolor=color, clip_on=False))
        ax.text(x + .08, -.29, label, va='center', fontsize=8.0)
    out = []
    for row_index, r in enumerate(rows.sort_values('year').itertuples()):
        y = row_index * 1.34
        ax.text(-.05, y+.58, str(int(r.year)), ha='right', va='center', fontsize=8.2)
        values = [float(r.operator_equal_supported_and_covered), float(r.facility_equal_supported_and_covered)]
        for offset, value, color, fg in zip([.26, .90], values, [BLUE, MID], [INK, INK]):
            ax.add_patch(Rectangle((0, y+offset-.23), 1, .46, facecolor=PALE, edgecolor='none'))
            ax.add_patch(Rectangle((0, y+offset-.23), value, .46, facecolor=color, edgecolor='none'))
            ax.text(.035, y+offset, f"{value:.1%}", va='center', fontsize=8.2, color=fg, weight='bold')
        lo, hi = float(r.operator_cluster_ci95_lower), float(r.operator_cluster_ci95_upper)
        ax.add_patch(Rectangle((lo, y+.09), hi-lo, .34, facecolor='white', alpha=.84, edgecolor=INK, lw=.6))
        ax.plot([values[0], values[0]], [y+.09, y+.43], color=INK, lw=.85)
        ax.text(1.04, y+.26, f"[{lo:.1%}, {hi:.1%}]", va='center', fontsize=8.2, color=INK)
        out.append(dict(model=model, year=int(r.year), operator_equal=values[0], facility_equal=values[1], lower=lo, upper=hi))
    for x, label in [(0, '0%'), (.5, '50%'), (1, '100%')]:
        ax.text(x, 3.94, label, ha='center', va='top', fontsize=8.0, color=INK)
    return out


def render_fig4(builder, inputs):
    fig = plt.figure(figsize=(178/25.4, 9.4))
    grid = fig.add_gridspec(4, 2, height_ratios=[2.1, 2.1, 1.62, 2.0],
                            left=.10, right=.97, top=.885, bottom=.115, hspace=.78, wspace=.38)
    ymax = float(np.ceil(max(inputs.hero.mean_winkler_score.max(), inputs.follow.mean_winkler_score.max()))) + .25
    artists, matrix = [], []
    for i, model in enumerate(['M1', 'M2']):
        sub = grid[i, :].subgridspec(1, 2, width_ratios=[1.55, 1], wspace=.35)
        ax, detail = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
        block = inputs.hero[inputs.hero.model.eq(model)]
        width = scatter_quality(ax, block, model, 'primary_GSOD_2022_2024', (0, ymax), artists)
        scatter_quality(detail, block, model, 'primary_GSOD_2022_2024', (0, ymax), artists, zoom=True)
        high = block[block.coverage_90.ge(.85)]
        detail.set(xlim=(.85, 1.005), ylim=(width*.96, high.mean_winkler_score.max()*1.06), xticks=[.85, .9, .95, 1])
        detail.set_title('85-100% coverage detail', fontsize=9.0, pad=12)
        ax.set_title(f"{chr(97+i)}  {model} | 2022-2024: 387 facility-years", loc='left', weight='bold', fontsize=9.5, pad=26)
        ax.text(0, 1.05, f"Full width {width:.3f} MMBtu/MWh", transform=ax.transAxes, fontsize=8.1, color=INK)
        ax.text(.025, .91, f"Minimum {block.coverage_90.min():.1%}", transform=ax.transAxes, fontsize=8.5, color=INK)
        ax.set_ylabel('Mean Winkler score\n(MMBtu/MWh)', fontsize=8.8)
        ax.set_xlabel('Coverage on matched-support days', fontsize=8.7)
        lo, hi = detail.get_ylim()
        ax.add_patch(Rectangle((.85, lo), .15, hi-lo, fill=False, edgecolor=MID, lw=.6))
        fig.add_artist(ConnectionPatch(xyA=(1, (lo+hi)/2), coordsA=ax.transData, xyB=(0, .5),
                                      coordsB=detail.transAxes, arrowstyle='->', mutation_scale=9, color=MID, lw=.75))
        matrix += proportion_panel(fig.add_subplot(grid[2, i]), inputs.joint[inputs.joint.model.eq(model)], model, chr(99+i))
        follow = fig.add_subplot(grid[3, i])
        block = inputs.follow[inputs.follow.model.eq(model)]
        scatter_quality(follow, block, model, 'followup_2025', (0, ymax), artists)
        follow.set_title(f"{chr(101+i)}  {model} | 2025 follow-up: 121 facilities", loc='left', weight='bold', fontsize=9.2, pad=12)
        follow.text(.025, .91, f"Minimum {block.coverage_90.min():.1%}", transform=follow.transAxes, color=INK, fontsize=8.5)
        follow.set_xlabel('Coverage on matched-support days', fontsize=8.7)
        follow.set_ylabel('Mean Winkler score\n(MMBtu/MWh)', fontsize=8.8)
    handles = [Patch(facecolor=c, label=s) for c, s in zip(CATS.values(), ['<85%', '85-95%', '>95%'])]
    handles += [Line2D([], [], marker=m, ls='', color=GRAY, label=str(y)) for y, m in [(2022, 'o'), (2023, 's'), (2024, '^'), (2025, 'D')]]
    fig.legend(handles=handles, ncol=7, frameon=False, loc='upper center', bbox_to_anchor=(.54, .985), columnspacing=.65, fontsize=8.1)
    fig.text(.10, .040, 'c, d: supported-and-covered / eligible days; brackets: operator-equal 95% CI.', fontsize=8.0, color=INK)
    fig.text(.10, .020, 'a, b: 2022-2024 cohorts; e, f: 2025 follow-up. Detail panels repeat points.', fontsize=8.0, color=INK)
    export(builder, 4, 'COMMON', fig, artists, dict(annual_proportions=matrix,
           primary_objects=774, followup_objects=242, equal_full_panel_ylim=[0, ymax]), inputs)


def paired_data(inputs, model, branch):
    if branch == 'retrospective':
        frame = inputs.common[inputs.common.model.eq(model)]
        pair = frame.pivot(index='facility_id', columns='year', values='signed_bias_mmbtu_per_mwh')
        return pair[[2022, 2024]], float(inputs.primary_rho.loc[inputs.primary_rho.model.eq(model) & inputs.primary_rho.year_a.eq(2022) & inputs.primary_rho.year_b.eq(2024), 'spearman_rho'].item())
    frame = inputs.pairs[inputs.pairs.model.eq(model)].set_index('facility_id')
    pair = frame[['signed_bias_mmbtu_per_mwh_2024', 'signed_bias_mmbtu_per_mwh_2025']].copy()
    pair.columns = [2024, 2025]
    return pair, float(inputs.rho.loc[inputs.rho.model.eq(model), 'signed_bias_spearman'].item())


def paired_panel(fig, cell, inputs, model, branch, letter, limits, bins, artists, histograms,
                 separate_interval_summary=False):
    inner = cell.subgridspec(2, 2, width_ratios=[1, .17], height_ratios=[.17, 1], hspace=.025, wspace=.025)
    ax = fig.add_subplot(inner[1, 0])
    top = fig.add_subplot(inner[0, 0], sharex=ax)
    side = fig.add_subplot(inner[1, 1], sharey=ax)
    pair, rho = paired_data(inputs, model, branch)
    first, last = pair.columns
    dots = ax.scatter(pair[first], pair[last], s=20, alpha=.78, color=BLUE, edgecolor='white', lw=.35)
    ax.plot(limits, limits, '--', color=GRAY, lw=.8)
    ax.axhline(0, color='#E5EBEF', lw=.6)
    ax.axvline(0, color='#E5EBEF', lw=.6)
    xlab = '2022 bias' if branch == 'retrospective' else '2024 weather-bridge bias'
    ylab = '2024 bias' if branch == 'retrospective' else '2025 follow-up bias'
    ax.set(xlim=limits, ylim=limits, aspect='equal')
    ax.set_anchor('W')
    ax.set_xlabel(xlab+' (MMBtu/MWh)', fontsize=8.4)
    ax.set_ylabel(ylab+' (MMBtu/MWh)', fontsize=8.4)
    ax.text(.04, .95, f"Bias rank ρ = {rho:.3f}", transform=ax.transAxes, va='top', fontsize=8.5, color=INK)
    if branch == 'followup':
        summary=followup_summary(inputs,model)
        if not separate_interval_summary:
            ax.text(.04,.885,f"Coverage rank ρ = {summary['coverage_spearman']:.3f}",
                    transform=ax.transAxes,va='top',fontsize=8.5,color=INK)
    clean(ax)
    ax.grid(False)
    for h, values, orientation, year in [(top, pair[first], 'vertical', first), (side, pair[last], 'horizontal', last)]:
        counts, edges, _ = h.hist(values, bins=bins, orientation=orientation, color=MID, edgecolor='white', lw=.4)
        h.tick_params(labelbottom=False, labelleft=False, length=0)
        h.spines[:].set_visible(False)
        histograms.append(dict(model=model, branch=branch, year=int(year), edges=edges.tolist(), counts=counts.astype(int).tolist()))
    name = 'Original 2022-2024 cohort' if branch == 'retrospective' else '2024 weather bridge to 2025'
    title = (f"{letter}  {model} | {name}\nFollow-up paired cohort: 120 facilities\n"
             f"Same coverage class: {summary['same_coverage_category']}/{summary['facilities']}"
             if branch == 'followup' else f"{letter}  {model} | 2022 to 2024\nOriginal GSOD | 120 paired facilities")
    if branch == 'followup' and separate_interval_summary:
        title = f"{letter}  {model} | 2024 weather bridge → 2025\nFollow-up paired cohort: 120 facilities"
    title_lines = title.removeprefix(letter + '  ').split('\n')
    fig.canvas.draw()
    pos, tp, sp = ax.get_position(), top.get_position(), side.get_position()
    # The title now has its own header; retain a positive-height marginal below it.
    bottom = max(tp.y0, pos.y1 + 1 / 25.4 / fig.get_figheight())
    header = fig.add_axes([pos.x0, tp.y1, pos.width, .001])
    header.axis('off')
    marginal_height = tp.y1 - bottom - 4.2 / 25.4 / fig.get_figheight()
    if marginal_height <= 0:
        raise ValueError('Insufficient height for marginal histogram')
    top.set_position([pos.x0, bottom, pos.width, marginal_height])
    side.set_position([sp.x0, pos.y0, sp.width, pos.height])
    panel_label(header, letter, title_lines[0], title_size=8.7)
    from matplotlib.transforms import ScaledTranslation
    for line_number, line in enumerate(title_lines[1:], 1):
        transform = header.transAxes + ScaledTranslation(4 / 25.4,
            (2 - 3.8 * line_number) / 25.4, fig.dpi_scale_trans)
        header.text(0, 1, line, transform=transform, ha='left', va='baseline',
                 fontsize=8.2, fontweight='bold', clip_on=False)
    for fid, xy in zip(pair.index, dots.get_offsets()):
        artists.append(dict(data_branch=branch, model=model, facility_id=int(fid), year_a=int(first), year_b=int(last), bias_a=float(xy[0]), bias_b=float(xy[1])))


def category_panel(ax, inputs, model, letter):
    rr = inputs.common[inputs.common.model.eq(model)]
    counts = rr.groupby(['year', 'coverage_direction']).size().unstack(fill_value=0).reindex(index=[2022, 2023, 2024], columns=list(CATS), fill_value=0)
    left, ys = np.zeros(3), np.array([2, 1, 0])
    for cat, color in CATS.items():
        vals = counts[cat].to_numpy()
        ax.barh(ys, vals, left=left, height=.40, color=color, edgecolor='white', lw=.4)
        left += vals
    for y, year in zip(ys, [2022, 2023, 2024]):
        ax.text(123, y, ' / '.join(str(int(v)) for v in counts.loc[year]), fontsize=8.0, va='center', color=INK)
    ax.set(xlim=(0, 174), ylim=(-.6, 2.6), yticks=ys, yticklabels=['2022', '2023', '2024'], xticks=[])
    panel_label(ax, letter, model + ' | Original common panel', title_size=8.8, content_bounds=(0, 0, 1, 1))
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0, labelsize=8.1)
    return [dict(model=model, year=year, facilities=120, **{k:int(v) for k,v in counts.loc[year].items()}) for year in [2022, 2023, 2024]]


def render_fig5(builder, inputs, variant):
    branches = ['retrospective', 'followup'] if variant == 'parallel' else ['followup']
    all_values = np.concatenate([paired_data(inputs, model, branch)[0].to_numpy().ravel() for model in ['M1', 'M2'] for branch in ['retrospective', 'followup']])
    limits = (float(np.floor(all_values.min()*5)/5-.1), float(np.ceil(all_values.max()*5)/5+.1))
    bins = np.linspace(*limits, 15)
    fig = plt.figure(figsize=(178/25.4, 8.6 if variant == 'parallel' else 5.4))
    grid = fig.add_gridspec(len(branches)+1, 2, height_ratios=[3.35]*len(branches)+[1.0],
                            left=.10, right=.975, top=.92, bottom=.14, hspace=.50, wspace=.39)
    artists, histograms, categories = [], [], []
    for row, branch in enumerate(branches):
        for col, model in enumerate(['M1', 'M2']):
            paired_panel(fig, grid[row, col], inputs, model, branch, chr(97+2*row+col), limits, bins, artists, histograms)
    for col, model in enumerate(['M1', 'M2']):
        categories += category_panel(fig.add_subplot(grid[-1, col]), inputs, model, chr(97+2*len(branches)+col))
    panels = 'e, f' if variant == 'parallel' else 'c, d'
    fig.legend(handles=[Patch(facecolor=c, label=s) for c, s in zip(CATS.values(), ['<85%', '85-95%', '>95%'])] ,
               title=f"Panels {panels}: original three-year panel (120 facilities)", ncol=3, frameon=False,
               loc='lower center', bbox_to_anchor=(.54, .025), fontsize=8.2, title_fontsize=8.2)
    fig.text(.10, .010, 'Dashed: equal bias. Three-year/follow-up cohort overlap: 106 facilities.', fontsize=8.0, color=INK)
    export(builder, 5, 'A' if variant == 'parallel' else 'B', fig, artists,
           dict(bias_limits=list(limits), histogram_edges=bins.tolist(), coverage_counts=categories,
                marginal_histograms=histograms, panel_overlap=inputs.panel_overlap,
                followup_summaries=[followup_summary(inputs,m) for m in ['M1','M2']]), inputs)


def render_retrospective_si(builder, inputs):
    values = np.concatenate([paired_data(inputs, model, branch)[0].to_numpy().ravel() for model in ['M1', 'M2'] for branch in ['retrospective', 'followup']])
    limits = (float(np.floor(values.min()*5)/5-.1), float(np.ceil(values.max()*5)/5+.1))
    bins = np.linspace(*limits, 15)
    fig = plt.figure(figsize=(178/25.4, 3.7))
    grid = fig.add_gridspec(1, 2, left=.10, right=.975, top=.88, bottom=.20, wspace=.39)
    artists, histograms = [], []
    for i, model in enumerate(['M1', 'M2']):
        paired_panel(fig, grid[0, i], inputs, model, 'retrospective', chr(97+i), limits, bins, artists, histograms)
    export(builder, 'S1', 'RETROSPECTIVE', fig, artists, dict(bias_limits=list(limits), histogram_edges=bins.tolist(), marginal_histograms=histograms), inputs)

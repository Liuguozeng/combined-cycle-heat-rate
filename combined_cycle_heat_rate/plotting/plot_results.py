'Render frozen primary combined_cycle_heat_rate Figures 2 and 3; no fitting or CI estimation.\n\nFigures 4 and 5 and retrospective S1 use the existing hash-bound\nselected_presentation/scripts/build_mixed_evidence_figures.py entrypoint with\n--followup-input-manifest, --figures combined_cycle_heat_rate:4 combined_cycle_heat_rate:5, --heat_rate-fig4-variant A,\n--heat_rate-fig5-variant followup_first and --emit-retrospective-si.\n'
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from panel_geometry import panel_label, write_panel_geometry
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
PAPERS = {'combined_cycle_heat_rate': '.'}
SUPS = {'combined_cycle_heat_rate': '.'}

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def configure_style():
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 10, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'legend.fontsize': 8.5, 'axes.edgecolor': GRAY, 'axes.linewidth': 0.6, 'text.color': INK, 'xtick.color': INK, 'ytick.color': INK, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none', 'savefig.facecolor': 'white'})

def clean(ax, grid='y'):
    ax.spines[['top', 'right']].set_visible(False)
    if grid:
        ax.grid(axis=grid, color='#E5EBEF', linewidth=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.6)

def validate_categories(frame):
    unknown = set(frame.coverage_direction.dropna()) - set(CATS)
    if unknown:
        raise ValueError(f'Unknown coverage categories: {unknown}')
    expected = np.where(frame.coverage_90 < 0.85, 'undercoverage', np.where(frame.coverage_90 > 0.95, 'overcoverage', 'nominal_band'))
    if not np.array_equal(expected, frame.coverage_direction):
        raise ValueError('Coverage category disagrees with frozen thresholds')

def bound(value):
    return float(value) if np.isfinite(value) else None

def uncertainty(provider):
    return 'point_only' if provider == 'AAU' else 'descriptive_range' if provider == 'Empa/NEST' else 'cluster_CI'

def effect(ax, y, value, lo, hi, color, provider, height=0.18):
    'Magnitude bar plus a co-located interval band, not a duplicate estimate.'
    bar = ax.barh(y, value, height=height * 0.5, color=color, alpha=0.9, zorder=3)[0]
    if np.isfinite(lo) and np.isfinite(hi):
        ax.add_patch(Rectangle((lo, y - height / 2), hi - lo, height, facecolor=color, alpha=0.12, edgecolor='none', zorder=2))
        ax.add_patch(Rectangle((lo, y - height / 2), hi - lo, height, fill=False, edgecolor=color, lw=0.5, linestyle='--' if provider == 'Empa/NEST' else '-', zorder=4))
    ax.plot([value, value], [y - height * 0.63, y + height * 0.63], color=color, lw=1.25, zorder=5)
    return float(bar.get_width())

def proportion_bands(ax, model, rows, panel):
    'Paired zero-based proportion tracks; only the operator estimate has a CI.'
    ax.clear()
    ax.axis('off')
    ax.set(xlim=(-0.17, 1.39), ylim=(3, 0))
    ax.set_title(f'{panel}  {model} | 120 common facilities', loc='left', fontsize=9.2, weight='bold', pad=25)
    for x, label, color in [(0, 'Operator equal', BLUE), (0.72, 'Facility equal', MID)]:
        ax.add_patch(Rectangle((x, -0.33), 0.06, 0.12, facecolor=color, clip_on=False))
        ax.text(x + 0.085, -0.27, label, va='center', fontsize=7.8)
    for y, r in enumerate(rows.sort_values('year').itertuples()):
        ax.text(-0.04, y + 0.39, str(int(r.year)), ha='right', va='center', fontsize=8.1)
        for offset, value, color, height in [(0.16, float(r.operator_equal_supported_and_covered), BLUE, 0.2), (0.55, float(r.facility_equal_supported_and_covered), MID, 0.16)]:
            ax.add_patch(Rectangle((0, y + offset - height / 2), 1, height, facecolor=PALE, edgecolor='none'))
            ax.add_patch(Rectangle((0, y + offset - height / 2), value, height, facecolor=color, edgecolor='none'))
            ax.text(1.045, y + offset, f'{value:.1%}', va='center', fontsize=8.5, color=INK, weight='bold')
        lo, hi = (float(r.operator_cluster_ci95_lower), float(r.operator_cluster_ci95_upper))
        ax.add_patch(Rectangle((lo, y + 0.025), hi - lo, 0.27, facecolor='white', alpha=0.55, edgecolor=INK, lw=0.6))
        ax.text(0.02, y + 0.86, f'95% CI [{lo:.1%}, {hi:.1%}]', va='center', fontsize=7.2, color=INK)
    ax.text(0, 3.1, '0%', ha='center', va='top', fontsize=7.2, color=INK)
    ax.text(1, 3.1, '100%', ha='center', va='top', fontsize=7.2, color=INK)

def supported_coverage_tracks(ax, model, rows, panel):
    'Two full-scale tracks, with numerical CI beside its operator estimate.'
    ax.clear()
    ax.axis('off')
    ax.set(xlim=(-0.17, 1.55), ylim=(3, 0))
    ax.set_title(f'{panel}  {model} | Supported-and-covered', loc='left', fontsize=9.2, weight='bold', pad=24)
    ax.text(0, -0.66, '120 common facilities', fontsize=8.1, color=INK, clip_on=False)
    for x, label, color in [(0, 'Operator equal', BLUE), (0.82, 'Facility equal', MID)]:
        ax.add_patch(Rectangle((x, -0.35), 0.055, 0.12, facecolor=color, clip_on=False))
        ax.text(x + 0.08, -0.29, label, va='center', fontsize=7.8)
    for y, r in enumerate(rows.sort_values('year').itertuples()):
        year = int(r.year)
        ax.text(-0.05, y + 0.43, str(year), ha='right', va='center', fontsize=8.1)
        for offset, value, color, weight in [(0.2, float(r.operator_equal_supported_and_covered), BLUE, 'operator'), (0.67, float(r.facility_equal_supported_and_covered), MID, 'facility')]:
            height = 0.38
            ax.add_patch(Rectangle((0, y + offset - height / 2), 1, height, facecolor=PALE, edgecolor='none'))
            bar = Rectangle((0, y + offset - height / 2), value, height, facecolor=color, edgecolor='none')
            bar.set_gid(f'proportion:{model}:{year}:{weight}')
            ax.add_patch(bar)
            ax.text(0.035, y + offset, f'{value:.1%}', va='center', fontsize=8.0, color=INK, weight='bold')
        lo, hi = (float(r.operator_cluster_ci95_lower), float(r.operator_cluster_ci95_upper))
        band = Rectangle((lo, y + 0.01), hi - lo, 0.38, facecolor='white', alpha=0.7, edgecolor=INK, lw=0.6)
        band.set_gid(f'operator_ci:{model}:{year}')
        ax.add_patch(band)
        value = float(r.operator_equal_supported_and_covered)
        ax.plot([value, value], [y + 0.01, y + 0.39], color=INK, lw=0.85)
        ax.text(1.04, y + 0.2, f'[{lo:.1%}, {hi:.1%}]', va='center', fontsize=7.8, color=INK)
    for x, label in [(0, '0%'), (0.5, '50%'), (1, '100%')]:
        ax.text(x, 3.08, label, ha='center', va='top', fontsize=7.5, color=INK)

class Builder:

    def __init__(self, candidate, output, selected=None, fig4_variant='A'):
        self.candidate, self.output = (candidate, output)
        self.selected = set(selected or [])
        self.records = []
        self.fig4_variant = fig4_variant

    def data(self, paper, name):
        return pd.read_csv(self.candidate / PAPERS[paper] / SUPS[paper] / 'results' / name)

    def wanted(self, paper, number):
        return not self.selected or f'{paper}:{number}' in self.selected

    def export(self, paper, index, fig, artists, supporting=None):
        import json
        parent = self.candidate / 'figures' / ('Figure_' + str(index))
        out = self.output / ('Figure_' + str(index))
        if out.resolve().is_relative_to(self.candidate.resolve()):
            raise ValueError('Output must be outside the read-only release')
        out.mkdir(parents=True, exist_ok=True)
        write_panel_geometry(fig, out / ('Figure_' + str(index)))
        for suffix in ['pdf', 'svg', 'png']:
            fig.savefig(out / ('Figure_' + str(index) + '.' + suffix), dpi=300)
        plt.close(fig)
        (out / 'ARTIST_DATA.json').write_text(json.dumps(artists, indent=2) + '\n')
        (out / 'PRESENTATION_DATA.json').write_text(json.dumps(supporting or {}, indent=2) + '\n')

    def heat_rate_2(self):
        df = pd.read_csv(next((self.candidate / PAPERS['combined_cycle_heat_rate'] / 'figures').glob('Figure_2/plotted_values.csv')))
        fig = plt.figure(figsize=(178 / 25.4, 4.05))
        gs = fig.add_gridspec(1, 2, width_ratios=[3.1, 1.25], left=0.1, right=0.98, bottom=0.24, top=0.88, wspace=0.28)
        ax = fig.add_subplot(gs[0])
        summary = fig.add_subplot(gs[1])
        summary.axis('off')
        artists = []
        quantiles = []
        for i, (year, color) in enumerate(zip([2022, 2023, 2024], [BLUE, MID, RED])):
            rr = df[df.year.eq(year)].sort_values(['support_fraction', 'facility_id'])
            values = rr.support_fraction.to_numpy()
            p = np.arange(1, len(values) + 1) / len(values)
            ax.step(np.r_[values[0], values], np.r_[0, p], where='post', color=color,
                    linestyle=['-', '--', ':'][i], lw=1.6, label=str(year))
            ax.vlines(values, -0.13 - i * 0.05, -0.097 - i * 0.05, color=color, lw=0.55, alpha=0.5, clip_on=False)
            ax.text(-0.025, -0.1135 - i * 0.05, str(year),
                    transform=ax.get_yaxis_transform(), ha='right', va='center',
                    color=INK, fontsize=8.5, clip_on=False)
            q = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
            y = 0.88 - i * 0.31
            summary.text(0, y, f'{year}   n = {len(values)}', fontsize=10, weight='bold', color=INK)
            summary.text(0, y - 0.09, f'Median  {q[2]:.1%}', fontsize=9)
            summary.text(0, y - 0.17, f'IQR  {q[1]:.1%}–{q[3]:.1%}', fontsize=8.5, color=INK)
            quantiles.append(dict(year=year, n=len(values), q05=float(q[0]), q25=float(q[1]), median=float(q[2]), q75=float(q[3]), q95=float(q[4])))
            for r, yy in zip(rr.itertuples(), p):
                artists.append(dict(facility_id=int(r.facility_id), year=year, support=float(r.support_fraction), ecdf=float(yy)))
        xmin = max(0, float(df.support_fraction.min()) - 0.025)
        ax.set(xlim=(xmin, 1.005), ylim=(0, 1.01), ylabel='Cumulative proportion of facilities')
        ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        ax.set_xlabel('In-support fraction of eligible weather-joined days', labelpad=40)
        clean(ax, 'both')
        ax.legend(loc='upper left', frameon=False)
        fig.text(0.1, 0.95, 'Operating-domain overlap across receiving facilities', weight='bold', fontsize=10)
        self.export('combined_cycle_heat_rate', 2, fig, artists, {'quantiles': quantiles, 'quantile_method': 'numpy linear; descriptive only', 'empirical_facility_years': len(df)})

    def heat_rate_3(self):
        df = pd.read_csv(next((self.candidate / PAPERS['combined_cycle_heat_rate'] / 'figures').glob('Figure_3/plotted_values.csv')))
        fig, axes = plt.subplots(2, 2, figsize=(178 / 25.4, 4.45), gridspec_kw={'width_ratios': [1.9, 1]})
        fig.subplots_adjust(left=0.075, right=0.975, bottom=0.16, top=0.89, hspace=0.53, wspace=0.24)
        bias_lim = float(df.signed_bias_mmbtu_per_mwh.abs().max()) * 1.08
        mae_lim = float(df.mae_mmbtu_per_mwh.max()) * 1.06
        artists = []
        stats = []
        for i, model in enumerate(['M1', 'M2']):
            rr = df[df.model.eq(model)]
            for j, column in enumerate(['signed_bias_mmbtu_per_mwh', 'mae_mmbtu_per_mwh']):
                ax = axes[i, j]
                values = np.sort(rr[column].to_numpy())
                grid = np.linspace(-bias_lim, bias_lim, 360) if j == 0 else np.linspace(0, mae_lim, 300)
                density = gaussian_kde(values, bw_method=0.18)(grid)
                density = 0.35 * density / density.max()
                ax.fill_between(grid, 0, density, color=MID, alpha=0.28, lw=0)
                ax.plot(grid, density, color=BLUE, lw=1.3)
                ax.vlines(values, -0.048, 0, color=BLUE, alpha=0.5, lw=0.55)
                med = float(np.median(values))
                q25, q75 = np.quantile(values, [0.25, 0.75])
                ax.plot([q25, q75], [-0.11] * 2, color=MID, lw=5, solid_capstyle='butt')
                ax.scatter(med, -0.11, s=25, facecolor='white', edgecolor=BLUE, zorder=3)
                if j == 0:
                    ax.axvline(0, color=GRAY, lw=0.65)
                ax.text(0.98, 0.93, f'{values.min():.3f} to {values.max():.3f}', transform=ax.transAxes, ha='right', va='top', fontsize=8.3, color=INK)
                ax.set(xlim=(grid.min(), grid.max()), ylim=(-0.18, 0.46), yticks=[])
                panel_label(ax, chr(97 + 2 * i + j), model + ' | ' + ('Signed bias' if j == 0 else 'MAE'), title_size=9.5)
                if i == 1:
                    ax.set_xlabel(('Signed bias' if j == 0 else 'MAE') + ' (MMBtu/MWh)')
                clean(ax, 'x')
                ax.spines['left'].set_visible(False)
                stats.append(dict(model=model, metric=column, minimum=float(values.min()), maximum=float(values.max()), median=med, q25=float(q25), q75=float(q75)))
            for r in rr.itertuples():
                artists.append(dict(facility_id=int(r.facility_id), year=int(r.year), model=model, bias=float(r.signed_bias_mmbtu_per_mwh), mae=float(r.mae_mmbtu_per_mwh)))
        fig.text(0.075, 0.965, '2024 screened cohort | 135 facilities per relation', fontsize=9.5, color=INK)
        fig.text(0.075, 0.037, 'Rugs: observed facilities. Open marks: medians. Thick bands: interquartile ranges.', fontsize=8.2, color=INK)
        self.export('combined_cycle_heat_rate', 3, fig, artists, {'descriptive_summaries': stats, 'density_bandwidth': 0.18})

    def heat_rate_4(self):
        from matplotlib.patches import ConnectionPatch
        df = pd.read_csv(next((self.candidate / PAPERS['combined_cycle_heat_rate'] / 'figures').glob('Figure_4/plotted_values.csv')))
        hero = df[df.record_type.eq('facility_interval_quality_full_annual_cohort')].copy()
        joint = df[df.record_type.eq('supported_and_covered_common_panel')].copy()
        validate_categories(hero)
        assert len(hero) == 774 and (not hero.duplicated(['facility_id', 'year', 'model']).any())
        fig = plt.figure(figsize=(178 / 25.4, 6.65))
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.68], left=0.1, right=0.98, bottom=0.12, top=0.87, hspace=0.78, wspace=0.2)
        markers = {2022: 'o', 2023: 's', 2024: '^'}
        artists = []
        matrix = []
        ylim = np.ceil(hero.mean_winkler_score.max())
        cmap = LinearSegmentedColormap.from_list('proportion', ['#E5EBEF', '#91B9D0'])
        for i, model in enumerate(['M1', 'M2']):
            rg = gs[i, :].subgridspec(1, 2, width_ratios=[1.95, 1.2], wspace=0.25)
            ax = fig.add_subplot(rg[0])
            inset = fig.add_subplot(rg[1])
            block = hero[hero.model.eq(model)]
            width = float(block.interval_width_mmbtu_per_mwh.iloc[0])
            zoom = block[block.coverage_90.ge(0.85)]
            for target, is_zoom in [(ax, False), (inset, True)]:
                target.axvspan(0.85, 0.95, color=PALE, zorder=-3)
                target.axvline(0.9, ls='--', color=GRAY, lw=0.7)
                target.axhline(width, ls=':', color=GRAY, lw=0.7)
                for year, marker in markers.items():
                    for cat, color in CATS.items():
                        rr = block[block.year.eq(year) & block.coverage_direction.eq(cat)]
                        if is_zoom:
                            rr = rr[rr.coverage_90.ge(0.85)]
                        points = target.scatter(rr.coverage_90, rr.mean_winkler_score, s=18, marker=marker, color=color, alpha=0.75, edgecolor='white', lw=0.25)
                        if not is_zoom:
                            for r, xy in zip(rr.itertuples(), points.get_offsets()):
                                artists.append(dict(facility_id=int(r.facility_id), year=int(r.year), model=model, category=cat, x=float(xy[0]), y=float(xy[1])))
                clean(target, 'y')
                target.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            ax.set(xlim=(0, 1.01), ylim=(0, ylim), ylabel='Mean Winkler score\n(MMBtu/MWh)')
            ax.set_title(f'{chr(97 + i)}  {model} | 387 facility-years', loc='left', weight='bold', fontsize=10, pad=17)
            ax.set_xlabel('Coverage conditional on matched input support')
            inset.set(xlim=(0.85, 1.005), ylim=(width * 0.96, float(zoom.mean_winkler_score.max()) * 1.06), xticks=[0.85, 0.9, 0.95, 1])
            inset.set_title('85–100% coverage detail', fontsize=9, pad=17)
            inset.tick_params(labelsize=8)
            lo, hi = inset.get_ylim()
            ax.add_patch(Rectangle((0.85, lo), 0.15, hi - lo, fill=False, edgecolor=MID, lw=0.6))
            fig.add_artist(ConnectionPatch(xyA=(1, (lo + hi) / 2), coordsA=ax.transData, xyB=(0, 0.5), coordsB=inset.transAxes, arrowstyle='->', mutation_scale=9, linewidth=0.75, color=MID, shrinkA=3, shrinkB=5, clip_on=False))
            ax.text(0, 1.025, f'Full width {width:.3f} MMBtu/MWh', transform=ax.transAxes, va='bottom', fontsize=8, color=INK)
            ax.text(0.025, 0.9, f'Minimum {block.coverage_90.min():.1%}', transform=ax.transAxes, fontsize=8.5, color=INK)
            ma = fig.add_subplot(gs[2, i])
            rr = joint[joint.model.eq(model)].sort_values('year')
            ma.set(xlim=(-0.43, 2), ylim=(3, 0))
            ma.axis('off')
            ma.set_title(f'{chr(99 + i)}  {model} | 120 common facilities', loc='left', fontsize=9.2, weight='bold', pad=24)
            for x, title in [(0, 'Operator equal'), (1, 'Facility equal')]:
                ma.text(x + 0.5, -0.18, title, ha='center', va='bottom', fontsize=8.5, color=INK)
            for y, r in enumerate(rr.itertuples()):
                values = [float(r.operator_equal_supported_and_covered), float(r.facility_equal_supported_and_covered)]
                ma.text(-0.1, y + 0.5, str(int(r.year)), ha='right', va='center', fontsize=8.5)
                for x, v in enumerate(values):
                    ma.add_patch(Rectangle((x, y), 1, 1, facecolor=cmap(Normalize(0, 1)(v)), edgecolor='white', lw=2))
                    ma.text(x + 0.5, y + 0.39 if x == 0 else y + 0.5, f'{v:.1%}', ha='center', va='center', fontsize=10, color=INK, weight='bold')
                    if x == 0:
                        ma.text(x + 0.5, y + 0.72, f'[{r.operator_cluster_ci95_lower:.1%}, {r.operator_cluster_ci95_upper:.1%}]', ha='center', va='center', fontsize=7.8, color=INK)
                matrix.append(dict(model=model, year=int(r.year), operator_equal=values[0], facility_equal=values[1], lower=float(r.operator_cluster_ci95_lower), upper=float(r.operator_cluster_ci95_upper), facilities=120))
            supported_coverage_tracks(ma, model, rr, chr(99 + i))
        handles = [Patch(facecolor=c, label=s) for c, s in zip(CATS.values(), ['<85%', '85–95%', '>95%'])] + [Line2D([], [], marker=m, ls='', color=GRAY, label=str(y)) for y, m in markers.items()]
        fig.legend(handles=handles, ncol=6, frameon=False, loc='upper center', bbox_to_anchor=(0.53, 0.988), columnspacing=0.9)
        fig.text(0.1, 0.055, 'c, d: supported-and-covered / eligible days; 95% CI: operator-equal estimates only.', fontsize=8.1, color=INK)
        fig.text(0.1, 0.02, 'Detail panels repeat high-coverage observations from the full annual cohorts.', fontsize=8.1, color=INK)
        self.export('combined_cycle_heat_rate', 4, fig, artists, {'annual_proportions': matrix, 'proportion_track_scale': [0, 1], 'variant': 'A', 'zoom_pointer': 'actual selected rectangle to adjacent detailed axes'})

    def heat_rate_5(self):
        df = pd.read_csv(next((self.candidate / PAPERS['combined_cycle_heat_rate'] / 'figures').glob('Figure_5/plotted_values.csv')))
        common = df[df.record_type.eq('common_panel_facility')].copy()
        persistence = df[df.record_type.eq('bias_rank_persistence')]
        validate_categories(common)
        fig = plt.figure(figsize=(178 / 25.4, 5.1))
        outer = fig.add_gridspec(2, 2, height_ratios=[3.35, 1.12], left=0.1, right=0.975, top=0.9, bottom=0.15, hspace=0.35, wspace=0.32)
        bias = common[common.year.isin([2022, 2024])].signed_bias_mmbtu_per_mwh
        lim = (np.floor(bias.min() * 5) / 5 - 0.1, np.ceil(bias.max() * 5) / 5 + 0.1)
        bins = np.linspace(*lim, 15)
        artists = []
        categories = []
        marginals = []
        for i, model in enumerate(['M1', 'M2']):
            rr = common[common.model.eq(model)]
            pair = rr.pivot(index='facility_id', columns='year', values='signed_bias_mmbtu_per_mwh')
            assert len(pair) == 120 and (not pair.isna().any().any())
            inner = outer[0, i].subgridspec(2, 2, width_ratios=[1, 0.17], height_ratios=[0.17, 1], hspace=0.035, wspace=0.035)
            ax = fig.add_subplot(inner[1, 0])
            top = fig.add_subplot(inner[0, 0], sharex=ax)
            side = fig.add_subplot(inner[1, 1], sharey=ax)
            points = ax.scatter(pair[2022], pair[2024], s=22, alpha=0.75, color=BLUE, edgecolor='white', lw=0.4)
            ax.plot(lim, lim, '--', color=GRAY, lw=0.8)
            ax.axhline(0, color='#E5EBEF', lw=0.6)
            ax.axvline(0, color='#E5EBEF', lw=0.6)
            ax.set(xlim=lim, ylim=lim, xlabel='2022 bias (MMBtu/MWh)', ylabel='2024 bias (MMBtu/MWh)', aspect='equal')
            for h, values, orientation in [(top, pair[2022], 'vertical'), (side, pair[2024], 'horizontal')]:
                counts, edges, _ = h.hist(values, bins=bins, orientation=orientation, color=MID, edgecolor='white', lw=0.4)
                h.tick_params(labelbottom=False, labelleft=False, length=0)
                h.spines[:].set_visible(False)
                marginals.append(dict(model=model, year=2022 if orientation == 'vertical' else 2024, edges=edges.tolist(), counts=counts.astype(int).tolist()))
            top.set_title(f'{chr(97 + i)}  {model} | 120 paired facilities', loc='left', fontsize=9.3, weight='bold', pad=7)
            rho = persistence[persistence.model.eq(model) & persistence.year_a.eq(2022) & persistence.year_b.eq(2024)].spearman_rho.item()
            ax.text(0.04, 0.95, f'Rank ρ = {rho:.3f}', transform=ax.transAxes, va='top', fontsize=8.5)
            clean(ax, None)
            fig.canvas.draw()
            pos = ax.get_position()
            tp = top.get_position()
            sp = side.get_position()
            top.set_position([pos.x0, tp.y0, pos.width, tp.height])
            side.set_position([sp.x0, pos.y0, sp.width, pos.height])
            for fid, xy in zip(pair.index, points.get_offsets()):
                artists.append(dict(model=model, facility_id=int(fid), bias_2022=float(xy[0]), bias_2024=float(xy[1])))
            ca = fig.add_subplot(outer[1, i])
            counts = rr.groupby(['year', 'coverage_direction']).size().unstack(fill_value=0).reindex(index=[2022, 2023, 2024], columns=list(CATS), fill_value=0)
            assert counts.sum(axis=1).eq(120).all()
            left = np.zeros(3)
            ys = np.array([2, 1, 0])
            for cat, color in CATS.items():
                vals = counts[cat].to_numpy()
                ca.barh(ys, vals, left=left, height=0.4, color=color, edgecolor='white', lw=0.4)
                left += vals
            for y, year in zip(ys, [2022, 2023, 2024]):
                ca.text(123, y, ' / '.join((str(int(v)) for v in counts.loc[year])), fontsize=8, va='center', color=INK)
                categories.append(dict(model=model, year=year, **{k: int(v) for k, v in counts.loc[year].items()}))
            ca.set(xlim=(0, 174), ylim=(-0.6, 2.6), yticks=ys, yticklabels=['2022', '2023', '2024'], xticks=[])
            ca.set_title(f'{chr(99 + i)}  Conditional coverage counts', loc='left', weight='bold', fontsize=9.0, pad=7)
            ca.spines[:].set_visible(False)
            ca.tick_params(length=0)
        fig.legend(handles=[Patch(facecolor=c, label=s) for c, s in zip(CATS.values(), ['<85%', '85–95%', '>95%'])], title='Panels c, d', ncol=3, frameon=False, loc='lower center', bbox_to_anchor=(0.57, 0.025), fontsize=8.3, title_fontsize=8.3, columnspacing=1.3)
        fig.text(0.1, 0.01, 'Dashed: equal bias. Marginals: facility counts. Category numbers follow legend order.', fontsize=8.0, color=INK)
        self.export('combined_cycle_heat_rate', 5, fig, artists, {'coverage_counts': categories, 'marginal_histograms': marginals})

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True,
                        help='Destination outside the read-only release for primary Figures 2 and 3')
    args=parser.parse_args()
    configure_style()
    builder=Builder(Path(__file__).resolve().parents[1], args.output_root)
    for number in [2, 3]:
        getattr(builder, 'heat_rate_'+str(number))()

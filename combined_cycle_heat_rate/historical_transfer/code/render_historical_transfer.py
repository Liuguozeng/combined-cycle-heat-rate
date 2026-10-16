'Portable visual-only rendering of frozen historical-facility evidence.\n\nRequired inputs are explicit. No fit, matching, resampling or new inference runs.\nRun tests with --self-test; all test fixtures stay under the output directory.\n'

import argparse
import hashlib
import json
import tempfile
import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from panel_geometry import panel_label, above_frame, write_panel_geometry

import numpy as np
import pandas as pd

PALETTE = {'original': '#7C8792', 'pooled': '#C97E9C', 'facility': '#2879AD'}
EFFECT_PALETTE = {'improvement': '#2879AD', 'worsening': '#C97E9C', 'zero': '#7C8792'}
MAPPING_LABELS = {'wrong_identity': 'Permuted facility assignments',
                  'correct_identity': 'Actual facility assignments'}
INK = '#26333D'
CASES = ['FULL', 'LOO_7153', 'LOO_55042', 'LOO_55107', 'LOO_55294', 'LOO_55322', 'LOO_56047']
MODELS = ['M1', 'M2']
SCOPES = {'main': 120, 'conditional': 101}
MEAN_SHA256 = '0832c29ed06247969a909b8eb5b11e4f3d494f69d8dd1c88051ef911cc7a6199'
REQUIRED = {
    'FACILITY_ALL_METRICS.csv': (['case', 'scope', 'model', 'facility_id', 'alternative', 'mae', 'days', 'width'], ['case', 'scope', 'model', 'facility_id', 'alternative']),
    'FACILITY_ALL_CHANGES.csv': (['case', 'scope', 'model', 'facility_id', 'mae_facility_minus_original', 'mae_facility_minus_pooled'], ['case', 'scope', 'model', 'facility_id']),
    'FACILITY_EQUAL_SUMMARIES.csv': (['case', 'scope', 'model', 'alternative', 'mae', 'facilities'], ['case', 'scope', 'model', 'alternative']),
    'BOOTSTRAP_CI_ALL_CASES.csv': (['case', 'scope', 'model', 'quantity', 'estimate', 'ci_low', 'ci_high', 'facilities', 'valid_replicates'], ['case', 'scope', 'model', 'quantity']),
    'PERMUTATION_1000_SUMMARIES.csv': (['scope', 'model', 'replicate', 'facilities', 'mae'], ['scope', 'model', 'replicate']),
    'SOURCE_OFFSET_TRANSFER_SUMMARY.csv': (['case', 'model', 'facilities', 'same_sign_facilities'], ['case', 'model']),
    'SOURCE_CASE_SUMMARIES_DIRECTIONS_RANKS.csv': (['case', 'scope', 'model', 'facilities', 'mae_facility_minus_original', 'mae_facility_minus_pooled'], ['case', 'scope', 'model']),
    'FACILITY_IDENTITY_AND_OPERATOR_CHANGES.csv': (['facility_id', 'operator_id_2024', 'operator_id_2025', 'facility_name_2024'], ['facility_id']),
}
FROZEN_SHA256 = {
    'FACILITY_ALL_METRICS.csv': '1ac22fd9e6b7d907c41c1ba0ab1bd563e07bfb1737be5ebd958dbb25b49b14c2',
    'FACILITY_ALL_CHANGES.csv': '1d6bf91b74af2b8ac750114ea97b7809527d4869d3f49a4a181e526b89003277',
    'FACILITY_EQUAL_SUMMARIES.csv': 'cdda9fa78113e2a548b5cd7d9955354ff111698928be7a40e7175bc4caaa650a',
    'BOOTSTRAP_CI_ALL_CASES.csv': '5f54a647a855288d60ab6350ab5dd3953818e76cb0e0105a4038453f99267744',
    'PERMUTATION_1000_SUMMARIES.csv': 'd8f54b32406e7fcd57252d37d2d4ae7ab6450b182bebd44ba8d4f4209204f0ff',
    'SOURCE_OFFSET_TRANSFER_SUMMARY.csv': 'e4f0f09de0a38e0c16bab33158d291cd1d9b033e53f2bfb73ae4fd5a170ef422',
    'SOURCE_CASE_SUMMARIES_DIRECTIONS_RANKS.csv': '9e1bc9eaf1389c7981d8bef2b8acd807191fdc92f5ce2c9d491c219347911db6',
    'FACILITY_IDENTITY_AND_OPERATOR_CHANGES.csv': 'aa8d224d2ff06f335d4a10e45f034ed9409db92b2d78f076751c86a4dbbede90',
}


def choose(frame, **values):
    mask = pd.Series(True, index=frame.index)
    for key, value in values.items():
        mask &= frame[key].eq(value)
    return frame.loc[mask]


def load_results(results):
    data = {}
    for name, (columns, keys) in REQUIRED.items():
        path = results / name
        if not path.is_file():
            raise FileNotFoundError('Required input missing: ' + name)
        frame = pd.read_csv(path)
        if not set(columns).issubset(frame.columns):
            raise ValueError('Required column missing: ' + name)
        if frame.duplicated(keys).any():
            raise ValueError('duplicate registered key: ' + name)
        if frame[columns].isna().any().any():
            raise ValueError('Missing registered value: ' + name)
        for column in frame.select_dtypes(include='number'):
            if not np.isfinite(frame[column]).all():
                raise ValueError('Nonfinite number: ' + name)
        if 'case' in frame and set(frame.case) != set(CASES):
            raise ValueError('Wrong source-case identity: ' + name)
        if 'model' in frame and set(frame.model) != set(MODELS):
            raise ValueError('Wrong model identity: ' + name)
        if 'scope' in frame and set(frame.scope) != set(SCOPES):
            raise ValueError('Wrong scope identity: ' + name)
        if 'alternative' in frame and set(frame.alternative) != set(PALETTE):
            raise ValueError('Wrong alternative identity: ' + name)
        data[name] = frame
    identities = data['FACILITY_IDENTITY_AND_OPERATOR_CHANGES.csv']
    if len(identities) != 120 or identities.operator_id_2024.nunique() != 91:
        raise ValueError('Wrong paired/operator identity')
    ids = set(identities.facility_id)
    metrics = data['FACILITY_ALL_METRICS.csv']
    changes = data['FACILITY_ALL_CHANGES.csv']
    for case in CASES:
        for scope, expected in SCOPES.items():
            registered_ids = None
            for model in MODELS:
                delta = choose(changes, case=case, scope=scope, model=model)
                actual_ids = set(delta.facility_id)
                if len(delta) != expected or not actual_ids.issubset(ids) or (scope == 'main' and actual_ids != ids):
                    raise ValueError('Wrong facility identity or denominator')
                if registered_ids is not None and actual_ids != registered_ids:
                    raise ValueError('Models use different facility identity')
                registered_ids = actual_ids
                for alt in PALETTE:
                    values = choose(metrics, case=case, scope=scope, model=model, alternative=alt)
                    if len(values) != expected or set(values.facility_id) != actual_ids:
                        raise ValueError('Wrong alternative facility identity')
                    summary = choose(data['FACILITY_EQUAL_SUMMARIES.csv'], case=case, scope=scope, model=model, alternative=alt)
                    if len(summary) != 1 or int(summary.iloc[0].facilities) != expected:
                        raise ValueError('Wrong summary identity')
                    if not np.isclose(values.mae.mean(), summary.iloc[0].mae, atol=1e-12):
                        raise ValueError('MAE number disagrees with existing summary')
                pivot = choose(metrics, case=case, scope=scope, model=model).pivot(index='facility_id', columns='alternative', values='mae')
                for baseline in ['original', 'pooled']:
                    column = 'mae_facility_minus_' + baseline
                    if not np.allclose(delta.set_index('facility_id')[column].sort_index(),
                                       (pivot.facility - pivot[baseline]).sort_index(), atol=1e-12):
                        raise ValueError('Paired MAE number disagrees')
                    ci = choose(data['BOOTSTRAP_CI_ALL_CASES.csv'], case=case, scope=scope, model=model, quantity=column)
                    source = choose(data['SOURCE_CASE_SUMMARIES_DIRECTIONS_RANKS.csv'], case=case, scope=scope, model=model)
                    if len(ci) != 1 or len(source) != 1 or int(ci.iloc[0].facilities) != expected:
                        raise ValueError('Wrong CI/source identity')
                    if not np.isclose(ci.iloc[0].estimate, delta[column].mean(), atol=1e-12):
                        raise ValueError('CI estimate number disagrees')
                    if not np.isclose(source.iloc[0][column], ci.iloc[0].estimate, atol=1e-12):
                        raise ValueError('Source number disagrees')
                    if not (ci.iloc[0].ci_low <= ci.iloc[0].estimate <= ci.iloc[0].ci_high):
                        raise ValueError('Invalid registered CI endpoints')
    for scope, count in SCOPES.items():
        for model in MODELS:
            p = choose(data['PERMUTATION_1000_SUMMARIES.csv'], scope=scope, model=model)
            if len(p) != 1000 or set(p.replicate) != set(range(1, 1001)) or not p.facilities.eq(count).all():
                raise ValueError('Wrong permutation identity')
    for model, expected in [('M1', 24), ('M2', 21)]:
        d = choose(changes, case='FULL', scope='main', model=model)
        if int(d.mae_facility_minus_pooled.gt(0).sum()) != expected:
            raise ValueError('Wrong worsening number versus pooled')
    # Exact bytes bind this rendering to the registered immutable inputs.
    for name in REQUIRED:
        if hashlib.sha256((results / name).read_bytes()).hexdigest() != FROZEN_SHA256[name]:
            raise ValueError('Frozen input digest differs: ' + name)
    return data


def save_figure(fig, output, stem):
    import matplotlib
    write_panel_geometry(fig, output / stem)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ignored = set()
    for ax in fig.axes:
        for ticks, labels, limits in [(ax.get_xticks(), ax.get_xticklabels(), ax.get_xlim()),
                                      (ax.get_yticks(), ax.get_yticklabels(), ax.get_ylim())]:
            lo, hi = sorted(limits)
            ignored.update(id(label) for tick, label in zip(ticks, labels) if not lo <= tick <= hi)
    for label in fig.findobj(matplotlib.text.Text):
        if label.get_visible() and label.get_text() and id(label) not in ignored:
            if label.get_fontsize() < 8:
                raise ValueError('Font smaller than 8 pt')
            box = label.get_window_extent(renderer)
            if box.x0 < -0.5 or box.y0 < -0.5 or box.x1 > fig.bbox.width + 0.5 or box.y1 > fig.bbox.height + 0.5:
                raise ValueError('Figure text outside canvas: ' + label.get_text())
    for suffix in ['pdf', 'svg', 'png']:
        fig.savefig(output / (stem + '.' + suffix), dpi=300, facecolor='white')


def load_mean_summary(path):
    frame = pd.read_csv(path)
    if frame.duplicated(['scope', 'model']).any() or set(zip(frame.scope, frame.model)) != {(s, m) for s in SCOPES for m in MODELS}:
        raise ValueError('Historical-mean comparison identity mismatch')
    if hashlib.sha256(path.read_bytes()).hexdigest() != MEAN_SHA256:
        raise ValueError('Historical-mean input digest differs')
    for row in frame.itertuples():
        if row.facilities != SCOPES[row.scope] or row.valid_replicates != 5000:
            raise ValueError('Historical-mean denominator mismatch')
        if not np.isclose(row.mae_facility_centered - row.mae_historical_mean, row.paired_mae_delta, atol=1e-12):
            raise ValueError('Historical-mean paired difference mismatch')
        if not row.ci_low <= row.paired_mae_delta <= row.ci_high:
            raise ValueError('Historical-mean CI mismatch')
        if row.improved + row.worsened + row.ties != row.facilities:
            raise ValueError('Historical-mean count denominator mismatch')
    return frame


def render(results, output, mean_summary=None):
    data = load_results(results)
    mean_path = mean_summary or results.parent.parent / 'historical_mean/results/SUMMARY_MEAN_CONTROL.csv'
    mean_rows = load_mean_summary(mean_path)
    output.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    font_manager.findfont('Times New Roman', fallback_to_default=False)
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 8,
                        'axes.titlesize': 9, 'axes.labelsize': 8, 'xtick.labelsize': 8,
                        'ytick.labelsize': 8, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
                        'axes.spines.top': False, 'axes.spines.right': False,
                        'legend.frameon': False, 'axes.linewidth': 0.6,
                        'text.color': INK, 'axes.labelcolor': INK,
                        'xtick.color': INK, 'ytick.color': INK, 'axes.edgecolor': INK})
    metrics = choose(data['FACILITY_ALL_METRICS.csv'], case='FULL', scope='main')
    changes = data['FACILITY_ALL_CHANGES.csv']
    ci = data['BOOTSTRAP_CI_ALL_CASES.csv']
    summary = data['FACILITY_EQUAL_SUMMARIES.csv']
    perms = data['PERMUTATION_1000_SUMMARIES.csv']
    source = data['SOURCE_CASE_SUMMARIES_DIRECTIONS_RANKS.csv']
    source_rows, bin_rows = [], []
    bins = np.linspace(0, np.ceil(metrics.mae.max() * 10) / 10, 19)
    max_count = max(np.histogram(choose(metrics, model=m, alternative=a).mae, bins)[0].max()
                    for m in MODELS for a in PALETTE)
    fig = plt.figure(figsize=(178 / 25.4, 190 / 25.4))
    fig.suptitle('Historical facility information and later-year prediction', fontsize=10, y=0.99)
    handles = [Line2D([0], [0], color=PALETTE[a], lw=1.3,
                     linestyle='--' if a == 'pooled' else '-', label=label)
               for a, label in [('original', 'Original'), ('pooled', 'Pooled offset'), ('facility', 'Facility offset')]]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.53, 0.963), ncol=3,
               handlelength=1.6, columnspacing=1.8, title='Panels a, b only', title_fontsize=8)
    for model, letter, left in [('M1', 'a', 0.10), ('M2', 'b', 0.585)]:
        ax = fig.add_axes([left, 0.708, 0.385, 0.171])
        for alternative in PALETTE:
            g = choose(metrics, model=model, alternative=alternative)
            counts, _ = np.histogram(g.mae, bins)
            ax.stairs(counts, bins, color=PALETTE[alternative], linewidth=1.2,
                      linestyle='--' if alternative == 'pooled' else '-',
                      fill=alternative == 'facility', alpha=0.24 if alternative == 'facility' else 1)
            if alternative == 'facility':
                ax.stairs(counts, bins, color=PALETTE[alternative], linewidth=1.2)
            source_rows.extend({'panel': letter, 'model': model, 'facility_id': int(r.facility_id),
                                'alternative': alternative, 'mae': float(r.mae)} for r in g.itertuples())
            bin_rows.extend({'panel': letter, 'model': model, 'alternative': alternative,
                             'bin_left': float(x), 'bin_right': float(y), 'count': int(n)}
                            for x, y, n in zip(bins[:-1], bins[1:], counts))
        ax.set(xlim=(bins[0], bins[-1]), ylim=(0, max_count * 1.13),
               xlabel='Facility MAE (MMBtu/MWh)', ylabel='Facilities')
        panel_label(ax, letter, model + ': all 120 paired facilities')
        ax.set_xticks([0.0, 0.4, 0.8, 1.2, 1.6])
        for baseline, y in [('original', 0.638), ('pooled', 0.616)]:
            row = choose(ci, case='FULL', scope='main', model=model,
                         quantity='mae_facility_minus_' + baseline).iloc[0]
            fig.text(left, y, f"vs {baseline}: {row.estimate:+.3f} [{row.ci_low:+.3f}, {row.ci_high:+.3f}]", fontsize=8)
            source_rows.append({'panel': letter, 'model': model, 'comparison': baseline,
                                'estimate': float(row.estimate), 'ci_low': float(row.ci_low), 'ci_high': float(row.ci_high)})
        fig.text(left, 0.594, 'Facility-offset minus comparator | paired 95% CI', fontsize=8)
        mean = choose(mean_rows, scope='main', model=model).iloc[0]
        source_rows.append({'panel': letter, 'model': model, 'comparison': 'historical_mean',
                            'mae_historical_mean': float(mean.mae_historical_mean),
                            'mae_facility_centered': float(mean.mae_facility_centered),
                            'estimate': float(mean.paired_mae_delta), 'ci_low': float(mean.ci_low),
                            'ci_high': float(mean.ci_high), 'facilities': int(mean.facilities),
                            'historical_mean_better': int(mean.worsened)})
    # A comparison band uses the retained summary estimates, never synthetic samples.
    fig.text(0.10, 0.571, 'Historical mean versus centered relation | 2024 → 2025', fontsize=9, weight='bold')
    band = fig.add_axes([0.10, 0.480, 0.87, 0.081])
    band.set(xlim=(0, 1), ylim=(0, 1))
    band.axis('off')
    for y in [0.99, 0.57, 0.01]:
        band.plot([0, 1], [y, y], color=PALETTE['original'], lw=0.5)
    for x, label in [(0.02, 'Model'), (0.22, 'Historical-mean\npredictor MAE'),
                     (0.43, 'Centered\nMAE'), (0.68, 'Centered - historical mean\nMAE difference [95% CI]'),
                     (0.95, 'History lower\n(facilities)')]:
        band.text(x, 0.79, label, ha='left' if x == 0.02 else 'center', va='center', fontsize=8)
    for model, y in [('M1', 0.40), ('M2', 0.13)]:
        mean = choose(mean_rows, scope='main', model=model).iloc[0]
        band.text(0.02, y, model, va='center', fontsize=8, weight='bold')
        band.text(0.22, y, f"{mean.mae_historical_mean:.3f}", ha='center', va='center', fontsize=8)
        band.text(0.43, y, f"{mean.mae_facility_centered:.3f}", ha='center', va='center', fontsize=8)
        band.text(0.68, y, f"{mean.paired_mae_delta:+.3f} [{mean.ci_low:+.3f}, {mean.ci_high:+.3f}]",
                  ha='center', va='center', fontsize=8)
        band.text(0.95, y, f"{int(mean.worsened)} / 120", ha='center', va='center', fontsize=8)
    fig.text(0.10, 0.469, 'Positive MAE difference favors the historical-mean predictor.', fontsize=8)
    fig.text(0.10, 0.451, '1,000 permuted assignments per model', fontsize=9)
    fig.legend(handles=[Patch(facecolor=PALETTE['pooled'], alpha=0.55,
                              label=MAPPING_LABELS['wrong_identity']),
                        Line2D([], [], color=PALETTE['facility'], marker='|',
                               markersize=9, markeredgewidth=1.5, linestyle='',
                               label=MAPPING_LABELS['correct_identity'])],
               loc='upper left', bbox_to_anchor=(0.10, 0.438), ncol=1,
               handlelength=1.1, handletextpad=0.55, borderaxespad=0, labelspacing=0.5)
    wrong_bins = np.linspace(0.20, np.ceil(perms.mae.max() * 100) / 100, 31)
    wrong_peak = max(np.histogram(choose(perms, scope='main', model=m).mae, wrong_bins)[0].max() for m in MODELS)
    mapping_axes = []
    for model, bottom in [('M1', 0.255), ('M2', 0.112)]:
        ax = fig.add_axes([0.10, bottom, 0.345, 0.105],
                          sharex=mapping_axes[0] if mapping_axes else None)
        mapping_axes.append(ax)
        p = choose(perms, scope='main', model=model)
        counts, _ = np.histogram(p.mae, wrong_bins)
        ax.stairs(counts, wrong_bins, fill=True, color=PALETTE['pooled'], alpha=0.55, linewidth=0.7)
        correct = choose(summary, case='FULL', scope='main', model=model, alternative='facility').iloc[0].mae
        ax.vlines(correct, 0, wrong_peak * 0.20, color=PALETTE['facility'], lw=1.5)
        ax.text(correct + 0.012, wrong_peak * 0.22, f"{correct:.3f}",
                va='bottom', color=PALETTE['facility'], fontsize=8)
        if model == 'M1':
            panel_label(ax, 'c', model, title_size=8)
        else:
            above_frame(ax, model, offset_mm=2)
        ax.set(xlim=(wrong_bins[0], wrong_bins[-1]), ylim=(0, wrong_peak * 1.22), ylabel='Mappings')
        ax.set_xticks([0.2, 0.3, 0.4, 0.5, 0.6])
        if model == 'M2':
            ax.set_xlabel('Facility-equal mean MAE (MMBtu/MWh)')
        else:
            ax.tick_params(axis='x', labelbottom=False, bottom=False)
            ax.spines['bottom'].set_visible(False)
        source_rows.extend({'panel': 'c', 'model': model, 'replicate': int(r.replicate),
                            'alternative': 'wrong_identity', 'mae': float(r.mae)} for r in p.itertuples())
        source_rows.append({'panel': 'c', 'model': model, 'alternative': 'correct_identity', 'mae': float(correct)})
        bin_rows.extend({'panel': 'c', 'model': model, 'alternative': 'wrong_identity',
                         'bin_left': float(x), 'bin_right': float(y), 'count': int(n)}
                        for x, y, n in zip(wrong_bins[:-1], wrong_bins[1:], counts))
    fig.text(0.585, 0.445, 'Source-setting effects vs pooled', fontsize=8.5)
    ax = fig.add_axes([0.585, 0.116, 0.30, 0.244])
    cols = [('main', 'M1'), ('main', 'M2'), ('conditional', 'M1'), ('conditional', 'M2')]
    matrix = np.array([[choose(source, case=case, scope=s, model=m).iloc[0].mae_facility_minus_pooled
                        for s, m in cols] for case in CASES])
    colors = np.empty((*matrix.shape, 3))
    scale = np.abs(matrix).max()
    for i, case in enumerate(CASES):
        for j, (scope, model) in enumerate(cols):
            value = matrix[i, j]
            role = 'improvement' if value < 0 else 'worsening' if value > 0 else 'zero'
            shade = 0.10 + 0.15 * abs(value) / scale
            colors[i, j] = 1 - shade * (1 - np.array(to_rgb(EFFECT_PALETTE[role])))
            ax.text(j, i, f"{value:+.3f}", va='center', ha='center', fontsize=8)
            source_rows.append({'panel': 'd', 'case': case, 'scope': scope, 'model': model,
                                'comparison': 'pooled', 'estimate': float(value), 'facilities': SCOPES[scope]})
    ax.imshow(colors, aspect='auto', interpolation='nearest')
    panel_label(ax, 'd', content_bounds=(0, 0, 1, 1))
    ax.set(xticks=range(4), xticklabels=['Main\nM1', 'Main\nM2', 'Matched\nM1', 'Matched\nM2'],
           yticks=range(7), yticklabels=['Full'] + ['Omit ' + c[4:] for c in CASES[1:]])
    ax.xaxis.tick_top()
    ax.tick_params(length=0, pad=4)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for x in np.arange(-0.5, 4, 1):
        ax.axvline(x, color='white', lw=1)
    for y in np.arange(-0.5, 7, 1):
        ax.axhline(y, color='white', lw=1)
    fig.text(0.585, 0.090, 'Main n=120; condition-matched n=101', fontsize=8)
    fig.text(0.585, 0.065, 'MAE change (MMBtu/MWh); negative = improvement', fontsize=8)
    fig.text(0.585, 0.040, 'All source-case CIs: Supplementary Table S15b', fontsize=8)
    save_figure(fig, output, 'Figure_2')
    plt.close(fig)
    pd.DataFrame(source_rows).to_csv(output / 'Figure_2_source.csv', index=False)
    pd.DataFrame(bin_rows).to_csv(output / 'Figure_2_histogram_bins.csv', index=False)
    mapping = []
    identities = data['FACILITY_IDENTITY_AND_OPERATOR_CHANGES.csv'].set_index('facility_id')
    full = choose(changes, case='FULL')
    low = min(0, full[['mae_facility_minus_original', 'mae_facility_minus_pooled']].to_numpy().min())
    high = max(0, full[['mae_facility_minus_original', 'mae_facility_minus_pooled']].to_numpy().max())
    padding = (high - low) * 0.1
    for scope, count in SCOPES.items():
        fig, axes = plt.subplots(2, 2, figsize=(178 / 25.4, 180 / 25.4))
        fig.subplots_adjust(left=0.11, right=0.98, bottom=0.15, top=0.85, hspace=0.54, wspace=0.28)
        first = 0 if scope == 'main' else 4
        fig.suptitle(f"Complete facility changes: {scope} panel (n={count})", y=0.99, fontsize=10)
        fig.legend(handles=[Patch(color=EFFECT_PALETTE['improvement'], label='Improvement'),
                            Patch(color=EFFECT_PALETTE['worsening'], label='Worsening')],
                   loc='upper center', bbox_to_anchor=(0.53, 0.95), ncol=2)
        for index, (ax, model, baseline) in enumerate(zip(axes.ravel(), ['M1', 'M2', 'M1', 'M2'],
                                                       ['original', 'original', 'pooled', 'pooled'])):
            column = 'mae_facility_minus_' + baseline
            g = choose(full, scope=scope, model=model).sort_values([column, 'facility_id'])
            ranks = np.arange(1, count + 1)
            colors = np.where(g[column].lt(0), EFFECT_PALETTE['improvement'],
                              np.where(g[column].gt(0), EFFECT_PALETTE['worsening'], EFFECT_PALETTE['zero']))
            letter = chr(ord('a') + first + index)
            ax.bar(ranks, g[column], color=colors, width=1.0, linewidth=0, rasterized=False)
            ax.axhline(0, color=INK, lw=1.1, zorder=4)
            improved = int(g[column].lt(0).sum())
            worsened = int(g[column].gt(0).sum())
            tied = int(g[column].eq(0).sum())
            ax.axvline(improved + tied + 0.5, color=INK, lw=0.65,
                       linestyle=(0, (2, 2)), zorder=4)
            above_frame(ax, f"{improved} improved | {worsened} worsened | {tied} tied")
            ax.set(xlim=(0.5, count + 0.5), ylim=(low - padding, high + padding),
                   xlabel='Facility rank by displayed MAE change', ylabel='MAE change (MMBtu/MWh)')
            panel_label(ax, letter, model + ' vs ' + baseline)
            ax.set_xticks([1, 30, 60, 90, count] if count == 120 else [1, 25, 50, 75, 101])
            for rank, row in zip(ranks, g.itertuples()):
                identity = identities.loc[row.facility_id]
                mapping.append({'figure': 'S2', 'panel': letter, 'scope': scope, 'model': model,
                                'comparison': baseline, 'display_rank': int(rank), 'facility_id': int(row.facility_id),
                                'facility_name_2024': identity.facility_name_2024,
                                'operator_id_2024': int(identity.operator_id_2024), 'operator_id_2025': int(identity.operator_id_2025),
                                'mae_change': float(getattr(row, column)), 'worsened': bool(getattr(row, column) > 0)})
        save_figure(fig, output, 'Figure_S2_' + scope)
        plt.close(fig)
    pd.DataFrame(mapping).to_csv(output / 'Figure_S2_identity_mapping.csv', index=False)
    ci[ci.quantity.str.startswith('mae_')].to_csv(output / 'Source_case_MAE_CIs.csv', index=False)
    metadata = {'width_mm': 178, 'height_mm': 190, 'supplementary_height_mm': 180, 'font': 'Times New Roman', 'minimum_label_pt': 8,
                'palette_by_alternative': PALETTE, 'palette_by_effect': EFFECT_PALETTE,
                'source_files': [{'path': name, 'sha256': FROZEN_SHA256[name]} for name in REQUIRED],
                'histograms': {'a_b': '120 actual facility MAEs per alternative and model, shared 18 equal-width bins and scales, complete tails',
                               'c': '1000 actual main-panel permuted-assignment aggregates per model, shared bins and scales'},
                'matrix': 'all 7 source settings, main n=120 and conditional n=101, facility minus pooled MAE',
                'supplementary_individual_marks': len(mapping), 'supplementary_panels': 8,
                'stable_scientific_id': 'historical_facility_information',
                'manuscript_figure_number': 2,
                'wrong_identity_color': PALETTE['pooled'],
                'mapping_labels': MAPPING_LABELS,
                'mapping_marker': 'one short blue mark per model at the exact correct-identity MAE; no CI',
                'mapping_axis': 'one shared x-axis; identical count scales for M1 and M2',
                'top_legend_scope': 'a,b only',
                'historical_mean_summary': {'name': mean_path.name, 'sha256': MEAN_SHA256,
                    'comparison': 'centered minus historical mean; positive favors historical mean',
                    'history_year': 2024, 'evaluation_year': 2025,
                    'historical_mean_lower_facility_mae': {'M1': 108, 'M2': 110},
                    'ci_role': 'paired MAE difference, never facility count'},
                'source_data_files': ['Figure_2_source.csv', 'Figure_2_histogram_bins.csv', 'Figure_S2_identity_mapping.csv', 'Source_case_MAE_CIs.csv'],
                'new_fitting_matching_resampling_or_inference': False, 'smoothing': False,
                'all_negative_boundary_results_retained': True, 'permutation_p_value': False,
                'technical_render_checks': 'canvas bounds and minimum font passed'}
    (output / 'FIGURE_METADATA.json').write_text(json.dumps(metadata, indent=2) + '\n')


def run_tests(results, output):
    output.mkdir(parents=True, exist_ok=True)

    class ContractTests(unittest.TestCase):
        def test_mean_summary(self):
            rows = load_mean_summary(results.parent.parent / 'historical_mean/results/SUMMARY_MEAN_CONTROL.csv')
            self.assertEqual(len(rows), 4)
            for row in rows.itertuples():
                self.assertEqual(row.improved + row.worsened + row.ties, row.facilities)

        def test_missing_mean_input(self):
            with self.assertRaises(FileNotFoundError):
                load_mean_summary(output / 'absent_mean_summary.csv')

        def test_tampered_mean_input(self):
            with tempfile.TemporaryDirectory(dir=output) as temp:
                path = Path(temp) / 'summary.csv'
                frame = pd.read_csv(results.parent.parent / 'historical_mean/results/SUMMARY_MEAN_CONTROL.csv')
                frame.loc[0, 'mae_historical_mean'] += 0.02
                frame.to_csv(path, index=False)
                with self.assertRaisesRegex(ValueError, 'digest'):
                    load_mean_summary(path)

        def test_valid_frozen_inputs(self):
            data = load_results(results)
            self.assertEqual(len(data['FACILITY_IDENTITY_AND_OPERATOR_CHANGES.csv']), 120)

        def test_missing_input(self):
            with tempfile.TemporaryDirectory(dir=output) as temp:
                with self.assertRaises(FileNotFoundError):
                    load_results(Path(temp))

        def mutate(self, filename, edit, message):
            with tempfile.TemporaryDirectory(dir=output) as temp:
                target = Path(temp)
                for source in results.glob('*.csv'):
                    if source.name in REQUIRED:
                        (target / source.name).write_bytes(source.read_bytes())
                frame = pd.read_csv(target / filename)
                edit(frame).to_csv(target / filename, index=False)
                with self.assertRaisesRegex(ValueError, message):
                    load_results(target)

        def test_wrong_identity(self):
            def edit(frame):
                frame.loc[0, 'facility_id'] = 999999
                return frame
            self.mutate('FACILITY_ALL_METRICS.csv', edit, 'identity')

        def test_wrong_number(self):
            def edit(frame):
                frame.loc[0, 'mae'] += 0.05
                return frame
            self.mutate('FACILITY_ALL_METRICS.csv', edit, 'number|digest|MAE')

        def test_duplicate(self):
            self.mutate('FACILITY_ALL_METRICS.csv', lambda x: pd.concat([x, x.iloc[[0]]]), 'duplicate')

        def test_missing_column(self):
            self.mutate('FACILITY_ALL_METRICS.csv', lambda x: x.drop(columns='mae'), 'column')

        def test_palette_roles(self):
            self.assertEqual(PALETTE, {'original': '#7C8792', 'pooled': '#C97E9C', 'facility': '#2879AD'})
            self.assertEqual(EFFECT_PALETTE['worsening'], PALETTE['pooled'])
            self.assertEqual(EFFECT_PALETTE['improvement'], PALETTE['facility'])

        def test_mapping_labels(self):
            self.assertEqual(MAPPING_LABELS['wrong_identity'], 'Permuted facility assignments')
            self.assertEqual(MAPPING_LABELS['correct_identity'], 'Actual facility mapping')
            self.assertNotIn('pooled', ' '.join(MAPPING_LABELS.values()).lower())

        def test_complete_sign_counts(self):
            data = load_results(results)
            full = choose(data['FACILITY_ALL_CHANGES.csv'], case='FULL')
            for scope, count in SCOPES.items():
                for model in MODELS:
                    for baseline in ['original', 'pooled']:
                        values = choose(full, scope=scope, model=model)['mae_facility_minus_' + baseline]
                        self.assertEqual(int(values.lt(0).sum() + values.gt(0).sum() + values.eq(0).sum()), count)

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ContractTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    (output / 'TEST_RECEIPT.json').write_text(json.dumps({
        'tests_run': result.testsRun, 'failures': len(result.failures),
        'errors': len(result.errors), 'successful': result.wasSuccessful(),
        'scope': 'portable visual-only input and palette contracts'
    }, indent=2) + '\n')
    return result.wasSuccessful()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mean-summary', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.results.resolve() == args.output.resolve():
        parser.error('Output must not be the frozen input directory')
    if args.self_test:
        raise SystemExit(0 if run_tests(args.results, args.output) else 1)
    render(args.results, args.output, args.mean_summary)


if __name__ == '__main__':
    main()

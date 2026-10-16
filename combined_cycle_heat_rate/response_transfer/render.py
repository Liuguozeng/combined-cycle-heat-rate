"""Render the declared response comparisons from complete supplied tables."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap,TwoSlopeNorm
BLUE="#2879AD";PINK="#C97E9C";INK="#26333D";GRAY="#7C8792"
NAMES=["historical_mean","historical_median","original_centered","within_source_centered","local_response"]
LABELS=["Historical mean","Historical median","Original response","Within-source response","Local response"]
def style():
    font_manager.findfont('Times New Roman', fallback_to_default=False)
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 8.5, 'axes.labelsize': 9, 'axes.titlesize': 10, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'axes.linewidth': 0.65, 'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.facecolor': 'white'})

def clean(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(width=0.6, length=3)

def label(ax, letter, title):
    box = ax.get_position()
    fig = ax.figure
    fig.text(box.x0, box.y1 + 0.035, letter, fontweight='bold', fontsize=11, va='baseline')
    fig.text(box.x0 + 0.035, box.y1 + 0.035, title, fontsize=10, va='baseline')

def save(fig, out, name):
    fig.savefig(out / (name + '.pdf'), metadata={'Title': name.replace('_', ' '), 'Author': 'Di Zhao; Guo-Zeng Liu', 'Creator': '', 'Producer': ''})
    fig.savefig(out / (name + '.png'), dpi=420)
    plt.close(fig)

def response(source, out):
    summary = pd.read_csv(source / 'candidate_summaries.csv')
    cal = pd.read_csv(source / 'historical_calibration.csv')
    strata = pd.read_csv(source / 'stratum_summaries.csv')
    fig = plt.figure(figsize=(178 / 25.4, 151 / 25.4))
    a = fig.add_axes([0.245, 0.59, 0.335, 0.3])
    b = fig.add_axes([0.7, 0.59, 0.26, 0.3])
    colors = [GRAY, GRAY, PINK, '#65A7C4', BLUE]
    markers = ['o', 'D', 's', '^', 'o']
    for model, dy in [('M1', 0.12), ('M2', -0.12)]:
        for k, c in enumerate(NAMES):
            r = summary.query('scope=="main" and model==@model and candidate==@c').iloc[0]
            a.errorbar(r.mae, 4 - k + dy, xerr=[[r.mae - r.mae_ci95_lower], [r.mae_ci95_upper - r.mae]], fmt=markers[k], mfc='white' if model == 'M2' else colors[k], mec=colors[k], ecolor=colors[k], markersize=4.2, capsize=2, elinewidth=0.8)
    a.set(yticks=range(5), yticklabels=LABELS[::-1], xlabel='Facility-equal MAE (MMBtu/MWh)', xlim=(0.1, 0.31))
    a.set_xticks([0.1, 0.15, 0.2, 0.25, 0.3])
    a.grid(axis='x', color='#E6E8EA', lw=0.5)
    clean(a)
    label(a, 'a', 'Later-year point error')
    fig.text(0.045, 0.925, 'a', fontweight='bold', fontsize=11)
    fig.text(0.08, 0.925, 'Later-year point error', fontsize=10)
    for text in fig.texts[-4:-2]:
        text.set_visible(False)
    fig.text(0.245, 0.505, 'Filled: M1   Open: M2; 95% operator-cluster CIs', fontsize=8)
    for model, color, ls in [('M1', BLUE, '-'), ('M2', PINK, '--')]:
        x = np.sort(cal.query('model==@model').response_coefficient.to_numpy())
        b.step(x, np.arange(1, len(x) + 1) / len(x), where='post', color=color, ls=ls, lw=1.4, label=model)
    b.axvline(0, color=GRAY, lw=0.7)
    b.axvline(1, color=GRAY, lw=0.7, ls=':')
    b.set(xlim=(-0.75, 1.12), ylim=(0, 1.03), xlabel='2024 response coefficient', ylabel='Facility proportion')
    b.set_xticks([-0.5, 0, 0.5, 1])
    b.legend(frameon=False, fontsize=8, loc='upper left')
    clean(b)
    label(b, 'b', 'Historical response scale')
    cmap = LinearSegmentedColormap.from_list('direction', [BLUE, '#FFFFFF', PINK])
    norm = TwoSlopeNorm(vmin=-0.03, vcenter=0, vmax=0.03)
    for model, left, letter in [('M1', 0.13, 'c'), ('M2', 0.615, 'd')]:
        ax = fig.add_axes([left, 0.135, 0.285, 0.285])
        d = strata.query('kind=="conditions" and model==@model and candidate=="local_response"').copy()
        z = np.full((3, 3), np.nan)
        for r in d.itertuples():
            t, l = map(int, r.stratum.split('|'))
            z[t - 1, l - 1] = r.minus_historical_mean
        im = ax.imshow(z, cmap=cmap, norm=norm, origin='lower', aspect='equal')
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f'{z[i, j]:+.3f}', ha='center', va='center', fontsize=9, color='white' if z[i, j] < -0.02 else INK)
        ax.set(xticks=range(3), yticks=range(3), xticklabels=['Low', 'Middle', 'High'], yticklabels=['Low', 'Middle', 'High'], xlabel='Historical load tertile', ylabel='Historical temperature tertile')
        ax.tick_params(length=0, pad=5)
        label(ax, letter, model + ' local response minus mean')
        for spine in ax.spines.values():
            spine.set_color('#D1D6DA')
            spine.set_linewidth(0.6)
    cax = fig.add_axes([0.29, 0.055, 0.42, 0.017])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=[-0.03, 0, 0.03])
    cb.ax.tick_params(labelsize=8, length=2)
    fig.text(0.045, 0.049, 'MAE difference\n(MMBtu/MWh)', fontsize=8, va='center')
    save(fig, out, 'Figure_2')
    summary.to_csv(out / 'Figure_2_source_summary.csv', index=False, float_format='%.17g')
    cal.to_csv(out / 'Figure_2_source_coefficients.csv', index=False, float_format='%.17g')
    strata.to_csv(out / 'Figure_2_source_strata.csv', index=False, float_format='%.17g')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);style();response(a.results,a.output)

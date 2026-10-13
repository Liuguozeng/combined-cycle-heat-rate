'Regression tests for interval identity and paired-summary presentation.'
import argparse
import sys
import unittest
from pathlib import Path

import matplotlib.pyplot as plt

from build_mixed_evidence_figures import draw_interval
from followup_presentation import FollowupInputs, followup_summary, paired_panel

MANIFEST = None


class ContributionPresentation(unittest.TestCase):
    def test_ci_is_not_a_box(self):
        fig, ax = plt.subplots()
        draw_interval(ax, 0, .2, .9, 'blue', 'PreDist')
        self.assertEqual(len(ax.patches), 0)
        self.assertEqual(len(ax.lines), 3)
        plt.close(fig)

    def test_range_is_dashed(self):
        fig, ax = plt.subplots()
        draw_interval(ax, 0, .2, .9, 'blue', 'Empa/NEST')
        self.assertEqual(ax.lines[0].get_linestyle(), '--')
        plt.close(fig)

    def test_point_only_no_interval(self):
        fig, ax = plt.subplots()
        draw_interval(ax, 0, float('nan'), float('nan'), 'blue', 'AAU')
        self.assertEqual(len(ax.lines), 0)
        plt.close(fig)

    def test_pair_summary_uses_both_dimensions(self):
        inputs = FollowupInputs(MANIFEST)
        expected = {'M1': (.943, .593, 108), 'M2': (.947, .500, 114)}
        for model, (bias, coverage, same) in expected.items():
            row = followup_summary(inputs, model)
            self.assertEqual(round(row['signed_bias_spearman'], 3), bias)
            self.assertEqual(round(row['coverage_spearman'], 3), coverage)
            self.assertEqual(row['same_coverage_category'], same)
            self.assertEqual(row['facilities'], 120)

    def test_changed_coverage_summary_fails(self):
        inputs = FollowupInputs(MANIFEST)
        inputs.rho.loc[inputs.rho.model.eq('M1'), 'coverage_spearman'] = .8
        with self.assertRaisesRegex(ValueError, 'Coverage rank'):
            followup_summary(inputs, 'M1')

    def test_changed_category_summary_fails(self):
        inputs = FollowupInputs(MANIFEST)
        inputs.rho.loc[inputs.rho.model.eq('M2'), 'same_coverage_category'] = 119
        with self.assertRaisesRegex(ValueError, 'category count'):
            followup_summary(inputs, 'M2')

    def test_followup_rank_annotations_have_equal_emphasis(self):
        inputs = FollowupInputs(MANIFEST)
        fig = plt.figure(figsize=(7, 4))
        grid = fig.add_gridspec(1, 1)
        artists, histograms = [], []
        paired_panel(fig, grid[0, 0], inputs, 'M1', 'followup', 'a',
                     (-2, 2), [-2, -1, 0, 1, 2], artists, histograms)
        labels = {t.get_text().split(' rank')[0]: t for ax in fig.axes
                  for t in ax.texts if ' rank' in t.get_text()}
        self.assertEqual(set(labels), {'Bias', 'Coverage'})
        self.assertGreaterEqual(labels['Coverage'].get_fontsize(), labels['Bias'].get_fontsize())
        self.assertEqual(labels['Coverage'].get_color(), labels['Bias'].get_color())
        self.assertEqual(len(artists), len(inputs.pairs.query("model == 'M1'")))
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    MANIFEST = parser.parse_args().manifest.resolve()
    unittest.main(argv=[sys.argv[0]], verbosity=2)

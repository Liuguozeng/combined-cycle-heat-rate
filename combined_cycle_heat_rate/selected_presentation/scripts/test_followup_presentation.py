#!/usr/bin/env python3
'Presentation regression and negative tests against staged, frozen inputs.'
import argparse
import copy
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from build_mixed_evidence_figures import Builder
from followup_presentation import FollowupInputs, paired_data, require_categories, sha

MANIFEST = None


class PresentationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        cfg = json.loads(MANIFEST.read_text())
        for name, item in cfg['inputs'].items():
            source = MANIFEST.parent / item['path']
            target = self.root / (name + '.csv')
            shutil.copyfile(source, target)
            item['path'] = target.name
        self.cfg = cfg
        self.path = self.root / 'inputs.json'
        self.write()

    def tearDown(self):
        # Absolute paths avoid dir_fd-relative cleanup in the existing audit guard.
        for path in sorted(self.root.rglob('*'), key=lambda p: len(p.parts), reverse=True):
            path.rmdir() if path.is_dir() else path.unlink()
        self.tmp.cleanup()

    def write(self):
        self.path.write_text(json.dumps(self.cfg))

    def mutate(self, name, fn):
        target = self.root / self.cfg['inputs'][name]['path']
        frame = pd.read_csv(target)
        fn(frame).to_csv(target, index=False)
        self.cfg['inputs'][name]['sha256'] = sha(target)
        self.write()

    def test_full_identity(self):
        data = FollowupInputs(self.path)
        self.assertEqual(len(data.hero), 774)
        self.assertEqual(len(data.follow), 242)
        self.assertEqual(data.panel_overlap, 106)
        for model in ['M1', 'M2']:
            for branch in ['retrospective', 'followup']:
                pair, _ = paired_data(data, model, branch)
                self.assertEqual(pair.shape, (120, 2))

    def test_missing_input(self):
        (self.root / self.cfg['inputs']['followup_metrics']['path']).unlink()
        with self.assertRaises(FileNotFoundError): FollowupInputs(self.path)

    def test_missing_manifest_entry(self):
        del self.cfg['inputs']['followup_pairs']; self.write()
        with self.assertRaises(ValueError): FollowupInputs(self.path)

    def test_hash_mismatch(self):
        self.cfg['inputs']['followup_metrics']['sha256'] = '0' * 64; self.write()
        with self.assertRaisesRegex(ValueError, 'hash mismatch'): FollowupInputs(self.path)

    def test_wrong_weather_branch(self):
        self.cfg['inputs']['followup_metrics']['role'] = 'primary_GSOD_2024'; self.write()
        with self.assertRaisesRegex(ValueError, 'Wrong analysis branch'): FollowupInputs(self.path)

    def test_cross_paper_input(self):
        self.cfg['inputs']['primary_fig4']['role'] = 'return_temperature_comparability_heat_weighted'; self.write()
        with self.assertRaises(ValueError): FollowupInputs(self.path)

    def test_duplicate_primary_identity(self):
        self.mutate('primary_fig4', lambda d: pd.concat([d, d.query("record_type == 'facility_interval_quality_full_annual_cohort'").iloc[[0]]]))
        with self.assertRaisesRegex(ValueError, 'Duplicate identity'): FollowupInputs(self.path)

    def test_missing_primary_point(self):
        self.mutate('primary_fig4', lambda d: d.drop(d.query("record_type == 'facility_interval_quality_full_annual_cohort'").index[0]))
        with self.assertRaisesRegex(ValueError, 'omitted or added'): FollowupInputs(self.path)

    def test_duplicate_followup_identity(self):
        self.mutate('followup_metrics', lambda d: pd.concat([d, d.iloc[[0]]]))
        with self.assertRaisesRegex(ValueError, 'Duplicate identity'): FollowupInputs(self.path)

    def test_missing_2025_point(self):
        self.mutate('followup_metrics', lambda d: d.drop(d.query('year == 2025').index[0]))
        with self.assertRaisesRegex(ValueError, 'omitted or added'): FollowupInputs(self.path)

    def test_duplicate_paired_identity(self):
        self.mutate('followup_pairs', lambda d: pd.concat([d, d.iloc[[0]]]))
        with self.assertRaisesRegex(ValueError, 'paired identities'): FollowupInputs(self.path)

    def test_missing_paired_point(self):
        self.mutate('followup_pairs', lambda d: d.iloc[1:])
        with self.assertRaisesRegex(ValueError, 'paired identities'): FollowupInputs(self.path)

    def test_changed_pair_value(self):
        def change(d):
            d.loc[0, 'signed_bias_mmbtu_per_mwh_2025'] += .1
            return d
        self.mutate('followup_pairs', change)
        with self.assertRaisesRegex(ValueError, 'disagree'): FollowupInputs(self.path)

    def test_wrong_paired_year(self):
        self.mutate('followup_pairs', lambda d: d.assign(year_2024=2023))
        with self.assertRaisesRegex(ValueError, 'paired years'): FollowupInputs(self.path)

    def test_wrong_classification(self):
        self.mutate('followup_metrics', lambda d: d.assign(coverage_direction='nominal_band'))
        with self.assertRaisesRegex(ValueError, 'category'): FollowupInputs(self.path)

    def test_missing_common_panel_member(self):
        self.mutate('primary_fig5', lambda d: d.drop(d.query("record_type == 'common_panel_facility'").index[0]))
        with self.assertRaisesRegex(ValueError, 'membership changed'): FollowupInputs(self.path)

    def test_classification_boundaries_exact(self):
        frame = pd.DataFrame(dict(coverage_90=[.85, .95, .849999999999, .950000000001],
                                  mean_winkler_score=[3.] * 4,
                                  coverage_direction=['nominal_band', 'nominal_band', 'undercoverage', 'overcoverage']))
        require_categories(frame)

    def test_no_silent_legacy_fallback(self):
        data = FollowupInputs(self.path)
        with self.assertRaisesRegex(ValueError, 'Choose parallel'):
            Builder(self.root, self.root/'out', ['combined_cycle_heat_rate:5'], followup=data)

    def test_new_layout_requires_inputs(self):
        with self.assertRaisesRegex(ValueError, 'manifest is required'):
            Builder(self.root, self.root/'out', ['combined_cycle_heat_rate:5'], fig5_variant='parallel')

    def test_requires_explicit_subset(self):
        data = FollowupInputs(self.path)
        with self.assertRaisesRegex(ValueError, 'explicit combined_cycle_heat_rate'):
            Builder(self.root, self.root/'out', followup=data)

    def test_legacy_interface_unchanged(self):
        builder = Builder(self.root, self.root/'out', ['combined_cycle_heat_rate:5'])
        self.assertIsNone(builder.followup)
        self.assertEqual(builder.fig5_variant, 'legacy')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    MANIFEST = args.manifest.resolve()
    unittest.main(argv=[sys.argv[0]], verbosity=2)

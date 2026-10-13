import itertools
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

import historical_transfer_analysis as c


class ScientificContractTests(unittest.TestCase):
    def frame(self, future=1.):
        return pd.DataFrame({
            'facility_id': [1, 1, 2, 2, 1, 2],
            'model': ['M1'] * 6, 'year': [2024] * 4 + [2025] * 2,
            'prediction': [3., 3., 3., 3., future, future],
            'observed': [2., 2., 2., 2., 0., 0.],
        })

    def test_pooled_only_has_no_identity_benefit(self):
        offsets, pooled = c.historical_offsets(self.frame(), [1, 2])
        np.testing.assert_allclose(offsets.offset_2024, 1.)
        self.assertEqual(pooled['M1'], 1.)
        x = c.alternatives(self.frame().query('year == 2025'), offsets, pooled, {'M1': 2.})
        np.testing.assert_allclose(x.query("alternative == 'facility'").residual, 0.)
        np.testing.assert_allclose(x.query("alternative == 'pooled'").residual, 0.)
        np.testing.assert_allclose(x.residual_prediction_minus_observed, x.prediction-x.observed)

    def test_sign_and_both_interval_endpoints(self):
        offsets, pooled = c.historical_offsets(self.frame(), [1, 2])
        x = c.alternatives(self.frame().query('year == 2025'), offsets, pooled, {'M1': 2.})
        x = x.query("alternative == 'facility'")
        np.testing.assert_allclose(x.prediction, 0.)
        np.testing.assert_allclose(x.lower, -2.)
        np.testing.assert_allclose(x.upper, 2.)
        np.testing.assert_allclose(x.radius, 2.)

    def test_future_outcomes_cannot_change_offsets(self):
        a = self.frame(); b = self.frame(1000.)
        b.loc[b.year.eq(2025), 'observed'] = -900.
        oa, pa = c.historical_offsets(a, [1, 2])
        ob, pb = c.historical_offsets(b, [1, 2])
        pd.testing.assert_frame_equal(oa, ob)
        self.assertEqual(pa, pb)
        with self.assertRaises(ValueError):
            c.historical_offsets(b.query('year == 2025'), [1, 2])

    def test_offset_can_fail_and_is_retained(self):
        offsets, pooled = c.historical_offsets(self.frame(), [1, 2])
        x = c.alternatives(self.frame(0.).query('year == 2025'), offsets, pooled, {'M1': 2.})
        m = c.facility_metrics(x)
        self.assertTrue((m.query("alternative == 'facility'").mae == 1.).all())
        self.assertTrue((m.query("alternative == 'original'").mae == 0.).all())

    def test_cardinality_priority_adversarial(self):
        # One zero-distance pair blocks two valid .09-distance pairs.
        # Euclidean geometry on a line: A=[0,-.09], B=[0,.09].
        d = np.abs(np.array([0., -.09])[:, None] - np.array([0., .09])[None, :])
        pairs = c.cardinality_assignment(d, .095)
        self.assertEqual(set(map(tuple, pairs)), {(0, 1), (1, 0)})

    def test_assignment_minimum_distance_given_cardinality(self):
        d = np.array([[.08, .01], [.02, .07]])
        pairs = c.cardinality_assignment(d, .095)
        self.assertEqual(set(map(tuple, pairs)), {(0, 1), (1, 0)})

    def test_assignment_exhaustive_oracle(self):
        rng = np.random.default_rng(321)
        for n, m in [(2, 3), (3, 2), (3, 3)]:
            for _ in range(20):
                d = rng.uniform(0, .2, (n, m))
                actual = c.cardinality_assignment(d, .095)
                options = []
                for choices in itertools.product(range(-1, m), repeat=n):
                    used = [j for j in choices if j >= 0]
                    if len(used) != len(set(used)):
                        continue
                    edges = [(i, j) for i, j in enumerate(choices) if j >= 0]
                    if any(d[i, j] > .095 for i, j in edges):
                        continue
                    options.append((-len(edges), sum(d[i, j] for i, j in edges)))
                best = min(options)
                got = (-len(actual), sum(d[i, j] for i, j in actual))
                self.assertEqual(got[0], best[0])
                self.assertAlmostEqual(got[1], best[1], places=12)

    def test_matching_empty_and_deterministic_ties(self):
        self.assertEqual(c.cardinality_assignment(np.empty((0, 3)), .095).shape, (0, 2))
        self.assertEqual(len(c.cardinality_assignment(np.ones((2, 3)), .095)), 0)
        np.testing.assert_array_equal(c.cardinality_assignment(np.zeros((3, 3)), .095),
                                      c.cardinality_assignment(np.zeros((3, 3)), .095))

    def test_duplicates_and_unknown_year_rejected(self):
        d = pd.DataFrame({'facility_id': [1, 1], 'date': ['2024-01-01'] * 2, 'year': [2024] * 2})
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            c.validate_dates(d, [2024, 2025], ['facility_id', 'date'])
        d = d.iloc[:1].copy(); d['year'] = 2026; d['date'] = '2026-01-01'
        with self.assertRaisesRegex(ValueError, 'year'):
            c.validate_dates(d, [2024, 2025], ['facility_id', 'date'])

    def test_bad_support_rejected(self):
        s = {'features_in_order': ['t', 'l'], 'mean': [0, 0], 'scale': [1, 1],
             'minimum': [-1, -1], 'maximum': [1, 1], 'nn_distance_threshold': .1}
        train = pd.DataFrame({'t': [0], 'l': [0]})
        target = pd.DataFrame({'t': [10], 'l': [0]})
        with self.assertRaisesRegex(ValueError, 'support'):
            c.require_supported(train, target, s)

    def test_missing_columns_and_duplicate_facility_offsets(self):
        with self.assertRaisesRegex(ValueError, 'missing'):
            c.validate_dates(pd.DataFrame({'date': ['2024-01-01']}), [2024], ['facility_id', 'date'])
        offsets, pooled = c.historical_offsets(self.frame(), [1, 2])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            c.alternatives(self.frame().query('year == 2025'), pd.concat([offsets, offsets.iloc[:1]]), pooled, {'M1': 2.})

    def test_missing_or_wrong_manifest_input_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / 'a'; f.write_text('a')
            m = {'files': {'a': {'path': str(f), 'sha256': 'wrong'}}}
            with self.assertRaisesRegex(ValueError, 'hash'):
                c.check_manifest(m)
            f.unlink()
            with self.assertRaisesRegex(ValueError, 'missing'):
                c.check_manifest(m)

    def test_wrong_cache_hash_and_unknown_cache_identity(self):
        with tempfile.TemporaryDirectory() as td:
            f = Path(td) / 'a.joblib'; f.write_bytes(b'not a model')
            with self.assertRaisesRegex(ValueError, 'cache'):
                c.validate_cache_entry(f, {'sha256': 'bad', 'model': 'M1', 'subset': 'LOO_1', 'included_sources': [2]}, 1, 'M1', [1, 2])
            with self.assertRaisesRegex(ValueError, 'cache'):
                c.validate_cache_entry(f, {'sha256': c.sha(f), 'model': 'M2', 'subset': 'LOO_1', 'included_sources': [2]}, 1, 'M1', [1, 2])

    def test_cluster_bootstrap_normalizes_facilities(self):
        # Unequal cluster sizes: a single copy of each cluster gives (0+0+9)/3.
        weights = c.facility_cluster_weights(np.array([[1, 1], [2, 0], [0, 2]]), [10, 10, 20], [10, 20])
        values = c.weighted_facility_stat(np.array([0., 0., 9.]), weights)
        np.testing.assert_allclose(values, [3., 0., 9.])
        self.assertNotEqual(values[0], 4.5)

    def test_conditional_offsets_not_refitted(self):
        a = self.frame(); a.loc[0, 'prediction'] = 5.
        offsets, pooled = c.historical_offsets(a, [1, 2])
        selected = a.query('year == 2025 and facility_id == 1')
        out = c.alternatives(selected, offsets, pooled, {'M1': 2.})
        self.assertEqual(out.query("alternative == 'facility'").applied_offset.iloc[0], 2.)
        self.assertEqual(out.query("alternative == 'pooled'").applied_offset.iloc[0], 1.5)


if __name__ == '__main__':
    unittest.main(verbosity=2)

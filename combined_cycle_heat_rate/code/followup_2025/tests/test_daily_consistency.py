import unittest
import pandas as pd
from daily_consistency import inspect_ssod, apply_daily_consistency


def row(mean=20.4, low=8.0, high=114.0, lc='*', hc='*'):
    return pd.DataFrame([{'mean_temperature':mean,'min_temperature':low,
        'max_temperature':high,'min_temperature_Measurement_Code':lc,
        'max_temperature_Measurement_Code':hc}])


class DailyConsistencyTests(unittest.TestCase):
    def test_known_greenville(self):
        r=inspect_ssod(row()).iloc[0]
        self.assertTrue(r.ssod_daily_inconsistent)
        self.assertEqual(r.ssod_consistency_reason,'hourly_derived_max_out_of_range')

    def test_missing_sentinels_not_extremes(self):
        r=inspect_ssod(row(20,-9999.9,-9999.9)).iloc[0]
        self.assertFalse(r.ssod_daily_inconsistent)
        self.assertTrue(pd.isna(r.ssod_min_c))
        self.assertTrue(pd.isna(r.ssod_max_c))

    def test_missing_mean_is_not_contradiction(self):
        r=inspect_ssod(row(-9999.9,8,25)).iloc[0]
        self.assertFalse(r.ssod_daily_inconsistent)

    def test_all_three_order_contradictions(self):
        r=inspect_ssod(row(20,30,10)).iloc[0]
        self.assertEqual(set(r.ssod_consistency_reason.split(';')),
            {'min_above_max','min_above_mean','mean_above_max'})

    def test_noncomparable_order_registered_not_rejected(self):
        r=inspect_ssod(row(20,21,25,'','')).iloc[0]
        self.assertFalse(r.ssod_daily_inconsistent)
        self.assertEqual(r.ssod_order_registered_only,'min_above_mean_period_not_established')

    def test_rounding_intervals_not_contradiction(self):
        self.assertFalse(inspect_ssod(row(20.4,20.5,25)).iloc[0].ssod_daily_inconsistent)
        self.assertFalse(inspect_ssod(row(20.4,8,20.3)).iloc[0].ssod_daily_inconsistent)
        self.assertTrue(inspect_ssod(row(20.4,20.6,25)).iloc[0].ssod_daily_inconsistent)

    def test_non_hourly_range_not_new_rule(self):
        self.assertFalse(inspect_ssod(row(20,8,114,'','')).iloc[0].ssod_daily_inconsistent)

    def test_inclusive_boundaries_and_equal_values(self):
        for mean,low,high in [(0,-60,60),(-60,-60,-60),(60,60,60)]:
            self.assertFalse(inspect_ssod(row(mean,low,high)).iloc[0].ssod_daily_inconsistent)

    def test_negative_hourly_extreme(self):
        self.assertIn('hourly_derived_min_out_of_range',
            inspect_ssod(row(-10,-61,0)).iloc[0].ssod_consistency_reason)

    def test_blank_extrema(self):
        self.assertFalse(inspect_ssod(row(20,'','')).iloc[0].ssod_daily_inconsistent)

    def test_official_local_missing_and_idempotence(self):
        d=inspect_ssod(row()); d['daily_method']='official_ssod_unchanged'
        d['temp_c']=20.4; d['weather_usable']=True
        a=apply_daily_consistency(d)
        self.assertFalse(a.iloc[0].weather_usable)
        self.assertTrue(pd.isna(a.iloc[0].temp_c))
        self.assertTrue(a.iloc[0].ssod_consistency_rejected)
        pd.testing.assert_frame_equal(a,apply_daily_consistency(a))

    def test_unused_ssod_does_not_reject_project(self):
        d=inspect_ssod(row()); d['daily_method']='project_hourly_reconstruction'
        d['temp_c']=12.5; d['weather_usable']=True
        a=apply_daily_consistency(d)
        self.assertTrue(a.iloc[0].weather_usable)
        self.assertEqual(a.iloc[0].temp_c,12.5)
        self.assertFalse(a.iloc[0].ssod_consistency_rejected)

    def test_normal_value_unchanged(self):
        d=inspect_ssod(row(20.4,8,25)); d['daily_method']='official_ssod_unchanged'
        d['temp_c']=20.4; d['weather_usable']=True
        a=apply_daily_consistency(d)
        self.assertTrue(a.iloc[0].weather_usable)
        self.assertEqual(a.iloc[0].temp_c,20.4)

    def test_unknown_numeric_token_fails(self):
        with self.assertRaises(ValueError): inspect_ssod(row(high='unknown'))


if __name__=='__main__': unittest.main()

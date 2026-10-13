import unittest
import numpy as np
import pandas as pd
from historical_mean_control import historical_levels, centered_predictions, facility_metrics


def frame(year, prediction, observed):
    return pd.DataFrame([dict(model=m, facility_id=1, year=year,
        date=f"{year}-01-{i:02d}", prediction=p, observed=y)
        for m in ('M1', 'M2') for i, (p, y) in enumerate(zip(prediction, observed), 1)])


class MeanControlTests(unittest.TestCase):
    def test_arithmetic_mean_and_offset(self):
        levels = historical_levels(frame(2024, [8, 10], [6, 8]), {1})
        self.assertTrue(levels.historical_observed_mean.eq(7).all())
        self.assertTrue(levels.offset_2024.eq(2).all())

    def test_centering_sign_and_algebra(self):
        levels = historical_levels(frame(2024, [8, 10], [6, 8]), {1})
        days = centered_predictions(frame(2025, [9, 11], [7, 9]), levels)
        np.testing.assert_array_equal(days.facility_centered, [7, 9, 7, 9])
        np.testing.assert_array_equal(facility_metrics(days, 'main').mae_centered_minus_historical_mean, [-1, -1])

    def test_constant_model_no_added_change(self):
        for prediction in (0, 9):
            levels = historical_levels(frame(2024, [prediction]*2, [6, 8]), {1})
            days = centered_predictions(frame(2025, [prediction]*2, [8, 9]), levels)
            np.testing.assert_array_equal(days.facility_centered, days.historical_mean)

    def test_model_change_can_harm(self):
        levels = historical_levels(frame(2024, [8, 10], [6, 8]), {1})
        days = centered_predictions(frame(2025, [13, 13], [7, 7]), levels)
        self.assertTrue(facility_metrics(days, 'main').mae_centered_minus_historical_mean.gt(0).all())

    def test_negative_offset(self):
        levels = historical_levels(frame(2024, [5, 7], [6, 8]), {1})
        days = centered_predictions(frame(2025, [6, 7], [7, 8]), levels)
        np.testing.assert_array_equal(days.facility_centered, [7, 8, 7, 8])

    def test_future_history_rejected(self):
        with self.assertRaisesRegex(ValueError, 'leakage'):
            historical_levels(frame(2025, [8, 9], [7, 8]), {1})

    def test_duplicate_day_rejected(self):
        data = frame(2024, [8, 9], [7, 8])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            historical_levels(pd.concat([data, data.iloc[:1]]), {1})

    def test_missing_facility_rejected(self):
        with self.assertRaisesRegex(ValueError, 'missing'):
            historical_levels(frame(2024, [8, 9], [7, 8]), {1, 2})

    def test_nonfinite_rejected(self):
        levels = historical_levels(frame(2024, [8, 9], [7, 8]), {1})
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            centered_predictions(frame(2025, [np.nan], [7]), levels)

    def test_cross_model_observation_mismatch_rejected(self):
        data = frame(2024, [8, 9], [7, 8])
        data.loc[data.model.eq('M2'), 'observed'] += 1
        with self.assertRaisesRegex(ValueError, 'observations'):
            historical_levels(data, {1})

    def test_cross_model_support_mismatch_rejected(self):
        data = frame(2024, [8, 9], [7, 8]).iloc[:-1]
        with self.assertRaisesRegex(ValueError, 'support'):
            historical_levels(data, {1})


if __name__ == '__main__':
    unittest.main(verbosity=2)

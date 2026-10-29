"""Leakage tests for the forecaster and the scheduler's carbon view (synthetic data only)."""
import unittest

import numpy as np

from src.data_sources import generate_synthetic_carbon
from src.errors import DataError
from src.forecasting import (CarbonForecaster, CarbonView, rolling_origin_evaluation, summarise_rolling)
from src.models import SiteSpec


def series(days=24):
    return generate_synthetic_carbon(days, seed=3)["actual"].to_numpy(float)


class LeakageTests(unittest.TestCase):
    def test_fit_ignores_everything_after_n_train(self):
        ci = series()
        n_train = 16 * 48
        corrupted = ci.copy()
        corrupted[n_train:] = np.random.default_rng(0).uniform(0, 1000, len(ci) - n_train)
        a, b = CarbonForecaster(max_iter=20), CarbonForecaster(max_iter=20)
        a.fit(ci, n_train, 0, 0)
        b.fit(corrupted, n_train, 0, 0)
        anchors, ks = np.arange(200, 300), np.array([1, 12, 48])
        np.testing.assert_allclose(a.predict_matrix(ci, anchors, ks), b.predict_matrix(ci, anchors, ks))

    def test_prediction_at_anchor_ignores_future_values(self):
        ci = series()
        fc = CarbonForecaster(max_iter=20)
        fc.fit(ci, 16 * 48, 0, 0)
        anchor = 17 * 48
        future = ci.copy()
        future[anchor + 1:] = 999.0
        ks = np.arange(1, 49)
        np.testing.assert_allclose(fc.predict_matrix(ci, np.array([anchor]), ks),
                                   fc.predict_matrix(future, np.array([anchor]), ks))

    def test_carbon_view_predictions_ignore_future_for_every_site(self):
        ci = series()
        fc = CarbonForecaster(max_iter=20)
        fc.fit(ci, 16 * 48, 0, 0)
        sites = [SiteSpec("a", 1.0, 0), SiteSpec("b", 0.9, -12), SiteSpec("c", 0.8, -24)]
        t, sim_start = 10, 17 * 48
        future = ci.copy()
        future[sim_start + t + 1:] = 999.0
        v1, v2 = CarbonView(ci, sites, fc, sim_start), CarbonView(future, sites, fc, sim_start)
        for s in sites:
            self.assertAlmostEqual(v1.predicted_sum(s.site_id, t, 0, 24), v2.predicted_sum(s.site_id, t, 0, 24))

    def test_rolling_folds_move_forward_and_never_train_on_test_block(self):
        f = rolling_origin_evaluation(series(24), 0, 0, max_iter=20, n_folds=3, block_steps=96, horizons=[1, 12])
        starts = sorted(f.test_start.unique())
        self.assertEqual(len(starts), 3)
        self.assertEqual(starts, sorted(starts))
        self.assertTrue((f.train_steps == f.test_start).all())          # train window ends where test begins
        self.assertEqual(set(f.method), {"gradient_boosting", "persistence", "seasonal_naive_24h"})

    def test_rolling_result_is_invariant_to_values_beyond_each_fold(self):
        # Changing the LAST block must not change the earlier folds' scores.
        ci = series(24)
        base = rolling_origin_evaluation(ci, 0, 0, max_iter=20, n_folds=3, block_steps=96, horizons=[4])
        mod = ci.copy()
        mod[-96:] += 200.0
        alt = rolling_origin_evaluation(mod, 0, 0, max_iter=20, n_folds=3, block_steps=96, horizons=[4])
        first = base[(base.fold == 0) & (base.method == "gradient_boosting")].mae.iloc[0]
        first_alt = alt[(alt.fold == 0) & (alt.method == "gradient_boosting")].mae.iloc[0]
        # fold 0 test block ends before the modified region only for anchors whose target < n-96
        self.assertNotEqual(base[base.fold == 2].mae.tolist(), alt[alt.fold == 2].mae.tolist())
        self.assertGreater(first, 0)
        self.assertTrue(np.isfinite(first_alt))

    def test_too_short_series_raises(self):
        with self.assertRaises(DataError):
            rolling_origin_evaluation(series(5), 0, 0, max_iter=5)

    def test_summary_has_skill_vs_persistence(self):
        f = rolling_origin_evaluation(series(24), 0, 0, max_iter=20, n_folds=2, block_steps=96, horizons=[4, 12])
        s = summarise_rolling(f)
        pers = s[s.method == "persistence"].skill_vs_persistence
        np.testing.assert_allclose(pers, 0.0)
        self.assertIn("folds_model_beats_persistence", s.columns)


if __name__ == "__main__":
    unittest.main()

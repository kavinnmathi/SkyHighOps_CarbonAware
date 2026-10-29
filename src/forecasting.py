"""Carbon-intensity forecasting (scikit-learn regression) and the per-site carbon view.

Target (real, supervised): the grid carbon intensity k steps ahead, learned from
past observations only. The same model is evaluated against two simple baselines
(persistence and 24-hour seasonal naive) on a held-out period.

Everything the scheduler sees about the future comes from this model's
PREDICTIONS; the emissions accounting uses the ACTUAL series.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .errors import DataError
from .models import SiteSpec

LAGS = [0, 1, 2, 3, 4, 6, 8, 12, 24, 48, 96]
MAX_LAG = max(LAGS)
EVAL_HORIZONS = [1, 2, 4, 8, 12, 24, 48]
FEATURE_NAMES = [f"lag_{l}" for l in LAGS] + ["mean_48", "tod_sin", "tod_cos", "dow_sin", "dow_cos", "horizon"]


class CarbonForecaster:
    def __init__(self, max_horizon: int = 48, max_iter: int = 150, learning_rate: float = 0.1,
                 step_minutes: int = 30, seed: int = 42):
        self.H = max_horizon
        self.step_minutes = step_minutes
        self.model = HistGradientBoostingRegressor(max_iter=max_iter, learning_rate=learning_rate,
                                                   min_samples_leaf=200, random_state=seed)
        self.minutes0 = 0
        self.dow0 = 0
        self.fitted = False

    # ---- feature construction (uses ONLY ci[: anchor+1] plus calendar time of the target)
    def _features(self, ci: np.ndarray, anchors: np.ndarray, ks: np.ndarray) -> np.ndarray:
        cs = np.concatenate([[0.0], np.cumsum(ci)])
        a = np.repeat(anchors, len(ks))
        k = np.tile(ks, len(anchors))
        cols = [ci[a - l] for l in LAGS]
        cols.append((cs[a + 1] - cs[a + 1 - 48]) / 48.0)
        total_min = self.minutes0 + (a + k) * self.step_minutes
        tod = (total_min % 1440) / 1440.0
        dow = (self.dow0 + total_min // 1440) % 7
        cols += [np.sin(2 * np.pi * tod), np.cos(2 * np.pi * tod),
                 np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7), k.astype(float)]
        return np.column_stack(cols)

    def fit(self, ci: np.ndarray, n_train: int, minutes0: int, dow0: int) -> None:
        """Train using only ci[:n_train] (targets never reach past n_train-1)."""
        self.minutes0, self.dow0 = int(minutes0), int(dow0)
        last_anchor = n_train - 1 - self.H
        if last_anchor <= MAX_LAG + 24:
            raise DataError("Not enough carbon-intensity history to train the forecaster "
                            f"(need > {MAX_LAG + 24 + self.H} training steps, have {n_train}).")
        anchors = np.arange(MAX_LAG, last_anchor + 1)
        ks = np.arange(1, self.H + 1)
        X = self._features(ci, anchors, ks)
        a_rep = np.repeat(anchors, len(ks))
        k_rep = np.tile(ks, len(anchors))
        # Target = CHANGE from the current value, divided by sqrt(horizon) so that short and
        # long horizons have comparable scale (this made the model beat persistence at 30 min).
        y = (ci[a_rep + k_rep] - ci[a_rep]) / np.sqrt(k_rep)
        self.model.fit(X, y)
        self.fitted = True

    def predict_matrix(self, ci: np.ndarray, anchors: np.ndarray, ks: np.ndarray) -> np.ndarray:
        """Predictions with shape (len(anchors), len(ks))."""
        if not self.fitted:
            raise RuntimeError("Forecaster must be fitted first.")
        X = self._features(ci, anchors, ks)
        current = np.repeat(ci[anchors], len(ks))
        scale = np.sqrt(np.tile(ks, len(anchors)).astype(float))
        return np.clip(current + self.model.predict(X) * scale, 0, None).reshape(len(anchors), len(ks))


def evaluate_forecaster(fc: CarbonForecaster, ci: np.ndarray, test_start: int) -> pd.DataFrame:
    """MAE / RMSE / R2 per horizon for model, persistence and seasonal-naive baselines."""
    rows = []
    for k in EVAL_HORIZONS:
        if k > fc.H:
            continue
        anchors = np.arange(test_start, len(ci) - k)
        if len(anchors) < 10:
            raise DataError("Test period too short to evaluate the forecaster.")
        truth = ci[anchors + k]
        preds = {
            "gradient_boosting": fc.predict_matrix(ci, anchors, np.array([k]))[:, 0],
            "persistence": ci[anchors],
            "seasonal_naive_24h": ci[anchors + k - 48],
        }
        for name, p in preds.items():
            rows.append({"horizon_steps": k, "horizon_hours": k * fc.step_minutes / 60.0, "method": name,
                         "mae": mean_absolute_error(truth, p),
                         "rmse": float(np.sqrt(mean_squared_error(truth, p))),
                         "r2": r2_score(truth, p), "n_test_points": len(anchors)})
    return pd.DataFrame(rows)


class CarbonView:
    """What schedulers can see: current ACTUAL intensity and PREDICTED future intensity per site.

    A site's series is derived from the base grid series:
        ci_site(t) = ci_scale * base(t + ci_shift_steps)      (SIMULATED assumption)
    Predictions never use base values later than the current step.
    """
    EXT = 200

    def __init__(self, base_ci: np.ndarray, sites: list[SiteSpec], forecaster: CarbonForecaster,
                 sim_start: int):
        self.base = base_ci
        self.n = len(base_ci)
        self.sites = {s.site_id: s for s in sites}
        self.fc = forecaster
        self.sim_start = sim_start
        self._base_pred_cache: dict[int, np.ndarray] = {}
        self._cs_cache: dict[tuple, np.ndarray] = {}

    def _idx(self, i):
        return np.clip(i, 0, self.n - 1)

    def actual(self, site_id: str, t: int) -> float:
        s = self.sites[site_id]
        return float(s.ci_scale * self.base[self._idx(self.sim_start + t + s.ci_shift_steps)])

    def _base_pred(self, t_abs: int) -> np.ndarray:
        if t_abs not in self._base_pred_cache:
            H = self.fc.H
            pred = self.fc.predict_matrix(self.base, np.array([t_abs]), np.arange(1, H + 1))[0]
            self._base_pred_cache[t_abs] = np.concatenate([[self.base[t_abs]], pred])
        return self._base_pred_cache[t_abs]

    def _cumsum(self, site_id: str, t: int) -> np.ndarray:
        key = (site_id, t)
        if key not in self._cs_cache:
            s = self.sites[site_id]
            t_abs = self.sim_start + t
            bp = self._base_pred(min(t_abs, self.n - 1 - 0))
            H = self.fc.H
            rel = np.arange(0, H + 1) + s.ci_shift_steps       # base time relative to now
            known = rel <= 0                                    # already observed -> actual
            vals = np.where(known, self.base[self._idx(t_abs + rel)], bp[np.clip(rel, 0, H)])
            path = s.ci_scale * vals
            ext = np.concatenate([path, np.full(self.EXT, path[-1])])
            self._cs_cache[key] = np.concatenate([[0.0], np.cumsum(ext)])
            if len(self._cs_cache) > 4000:                      # keep memory small
                self._cs_cache.pop(next(iter(self._cs_cache)))
        return self._cs_cache[key]

    def predicted_sum(self, site_id: str, t: int, start: int, length: float) -> float:
        """Sum of predicted CI over steps [t+start, t+start+length) (length may be fractional)."""
        cs = self._cumsum(site_id, t)
        L = len(cs) - 1
        a = min(max(start, 0), L)
        whole = int(np.floor(length))
        b = min(a + whole, L)
        total = cs[b] - cs[a]
        frac = length - whole
        if frac > 1e-9 and b < L:
            total += frac * (cs[b + 1] - cs[b])
        return float(total)

    def predicted_at(self, site_id: str, t: int, step: int = 0) -> float:
        return self.predicted_sum(site_id, t, step, 1.0)


def _sums_over_starts(view: "CarbonView", site_id: str, t: int, max_start: int, length: int) -> np.ndarray:
    """Predicted CI summed over `length` steps for every start offset 0..max_start (vectorised)."""
    cs = view._cumsum(site_id, t)
    L = len(cs) - 1
    a = np.clip(np.arange(0, max_start + 1), 0, L)
    b = np.clip(a + length, 0, L)
    return cs[b] - cs[a]


CarbonView.sums_over_starts = lambda self, site_id, t, max_start, length: _sums_over_starts(
    self, site_id, t, max_start, length)


# ------------------------------------------------------------------------------
# Leakage-safe rolling-origin (walk-forward) evaluation
# ------------------------------------------------------------------------------
ROLLING_HORIZONS = [1, 4, 12, 24, 48]


def rolling_origin_evaluation(ci: np.ndarray, minutes0: int, dow0: int, *, max_horizon: int = 48,
                              max_iter: int = 100, learning_rate: float = 0.05, step_minutes: int = 30,
                              seed: int = 42, n_folds: int = 4, block_steps: int = 96,
                              min_train_steps: int = 7 * 48, horizons=None) -> pd.DataFrame:
    """Walk-forward evaluation: for each fold a FRESH model is trained on ci[:origin] only and
    scored on the next ``block_steps`` forecast origins.

    Leakage guards (all covered by tests/test_forecast_eval.py):
      * the model is trained on a physically truncated array ``ci[:origin]`` - later values
        are not even present, so they cannot be used;
      * prediction features for an anchor use only ci[:anchor+1] (array is truncated again
        to the end of the test block);
      * targets are read from the full series only to compute errors;
      * the training window ends strictly before the first test target (no overlap), and
        folds move forward in time (expanding window, never shuffled).
    Returns one row per fold x horizon x method (model, persistence, seasonal naive 24 h).
    """
    ci = np.asarray(ci, dtype=float)
    horizons = [k for k in (horizons or ROLLING_HORIZONS) if k <= max_horizon]
    n = len(ci)
    origins = [n - (n_folds - f) * block_steps for f in range(n_folds)]
    origins = [o for o in origins if o >= min_train_steps]
    if not origins:
        raise DataError(f"Series too short for rolling evaluation ({n} steps; need > "
                        f"{min_train_steps + block_steps}). Use more days of data.")
    rows = []
    for fold, o in enumerate(origins):
        fc = CarbonForecaster(max_horizon, max_iter, learning_rate, step_minutes, seed)
        fc.fit(ci[:o], o, minutes0, dow0)                      # truncated: future is absent
        end = min(o + block_steps, n)
        visible = ci[:end]                                     # features never look past `end`
        for k in horizons:
            anchors = np.arange(o, end)
            anchors = anchors[anchors + k < n]
            if len(anchors) < 10:
                continue
            truth = ci[anchors + k]
            preds = {"gradient_boosting": fc.predict_matrix(visible, anchors, np.array([k]))[:, 0],
                     "persistence": ci[anchors],
                     "seasonal_naive_24h": ci[anchors + k - 48]}
            for name, p in preds.items():
                rows.append({"fold": fold, "train_steps": o, "test_start": o, "horizon_steps": k,
                             "horizon_hours": k * step_minutes / 60.0, "method": name,
                             "mae": float(mean_absolute_error(truth, p)),
                             "rmse": float(np.sqrt(mean_squared_error(truth, p))),
                             "n_test_points": len(anchors)})
    if not rows:
        raise DataError("Rolling evaluation produced no results (test blocks too short).")
    return pd.DataFrame(rows)


def summarise_rolling(folds: pd.DataFrame) -> pd.DataFrame:
    """Mean/std MAE across folds and skill relative to persistence (positive = better)."""
    g = folds.groupby(["horizon_steps", "horizon_hours", "method"]).agg(
        mae_mean=("mae", "mean"), mae_std=("mae", "std"), rmse_mean=("rmse", "mean"),
        n_folds=("fold", "nunique")).reset_index()
    pers = g[g.method == "persistence"].set_index("horizon_steps")["mae_mean"]
    g["skill_vs_persistence"] = 1.0 - g["mae_mean"] / g["horizon_steps"].map(pers)
    wins = folds.pivot_table(index=["horizon_steps", "fold"], columns="method", values="mae").reset_index()
    wins["beats"] = wins["gradient_boosting"] < wins["persistence"]
    g["folds_model_beats_persistence"] = g.apply(
        lambda r: int(wins[wins.horizon_steps == r.horizon_steps]["beats"].sum())
        if r.method == "gradient_boosting" else np.nan, axis=1)
    return g

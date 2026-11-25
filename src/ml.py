"""Machine-learning mortality forecasts.

A tree model fitted on (age, year, sex) cannot forecast: trees can't
extrapolate, so every future year gets the prediction for the last training
year. Instead this module learns **how mortality changes**:

    target  = log m(x, t + h) - log m(x, t)        (change over h years)
    inputs  = age, sex, horizon h, today's level log m(x, t),
              recent improvement (last 5 and 10 years, smoothed across ages),
              long-run improvement since the start of the data

Each training row is "standing in year t, looking h years ahead". Inputs
use only data up to year t, so nothing from the future leaks in.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

FEATURES = ["age", "sex", "h", "level", "imp5", "imp10", "imp_long"]

# Chosen on a validation backtest (origin 2004, forecasting 2005-2014),
# before the 2015-2024 test window was used. See notebook 03.
GBM_PARAMS = dict(max_iter=300, learning_rate=0.03, max_depth=3,
                  min_samples_leaf=20, random_state=0)


def _smooth_ages(s: pd.Series, window: int = 5) -> pd.Series:
    return s.rolling(window, center=True, min_periods=1).mean()


def make_features(log_m: pd.DataFrame, origin: int, h: int, sex: str) -> pd.DataFrame:
    """Inputs for every age, standing in year ``origin`` and looking ``h``
    years ahead. ``log_m`` is an age x year matrix of log death rates."""
    first = log_m.columns.min()
    lm = log_m
    return pd.DataFrame({
        "age": lm.index.to_numpy(),
        "sex": int(sex == "male"),
        "h": h,
        "level": lm[origin].to_numpy(),
        "imp5": _smooth_ages((lm[origin] - lm[origin - 5]) / 5).to_numpy(),
        "imp10": _smooth_ages((lm[origin] - lm[origin - 10]) / 10).to_numpy(),
        "imp_long": ((lm[origin] - lm[first]) / (origin - first)).to_numpy(),
    })


def build_training_set(log_m: dict[str, pd.DataFrame], first_origin: int,
                       last_year: int, horizon: int = 10):
    """Stack (features, target) rows for every sex, origin year and horizon
    whose target year is <= ``last_year``."""
    X, y = [], []
    for sex, lm in log_m.items():
        for origin in range(first_origin, last_year):
            for h in range(1, horizon + 1):
                if origin + h > last_year:
                    break
                X.append(make_features(lm, origin, h, sex))
                y.append((lm[origin + h] - lm[origin]).to_numpy())
    return pd.concat(X, ignore_index=True), np.concatenate(y)


class ImprovementGBM:
    """Gradient boosting on mortality change, one model shared by both sexes."""

    def __init__(self, **params):
        self.params = {**GBM_PARAMS, **params}
        self.model = HistGradientBoostingRegressor(**self.params)

    def fit(self, log_m: dict[str, pd.DataFrame], last_year: int,
            first_origin: int | None = None, horizon: int = 10) -> "ImprovementGBM":
        # need 10 years of history before the first origin for imp10
        if first_origin is None:
            first_origin = min(lm.columns.min() for lm in log_m.values()) + 10
        self.X_, self.y_ = build_training_set(log_m, first_origin, last_year, horizon)
        self.model.fit(self.X_[FEATURES], self.y_)
        self.log_m_, self.origin_ = log_m, last_year
        return self

    def forecast_mx(self, sex: str, horizon: int = 10) -> pd.DataFrame:
        """Death-rate forecast (age x year) from the last training year."""
        lm = self.log_m_[sex]
        out = {}
        for h in range(1, horizon + 1):
            change = self.model.predict(make_features(lm, self.origin_, h, sex)[FEATURES])
            out[self.origin_ + h] = np.exp(lm[self.origin_].to_numpy() + change)
        return pd.DataFrame(out, index=lm.index)


def blend(*forecasts: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight average of forecasts on the log scale."""
    return np.exp(sum(np.log(f) for f in forecasts) / len(forecasts))

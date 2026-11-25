"""Forecast accuracy measures for mortality rates."""
from __future__ import annotations

import numpy as np
import pandas as pd


def log_rmse(actual: pd.DataFrame, forecast: pd.DataFrame) -> float:
    """Root mean squared error of log death rates over all ages and years."""
    diff = np.log(forecast.to_numpy()) - np.log(actual.to_numpy())
    return float(np.sqrt(np.mean(diff**2)))


def mape(actual: pd.DataFrame, forecast: pd.DataFrame) -> float:
    """Mean absolute percentage error of death rates, in percent."""
    a, f = actual.to_numpy(), forecast.to_numpy()
    return float(np.mean(np.abs(f - a) / a) * 100)


def error_by_year(actual: pd.DataFrame, forecast: pd.DataFrame) -> pd.Series:
    """Log RMSE for each forecast year (useful to see the COVID years)."""
    diff = np.log(forecast) - np.log(actual)
    return np.sqrt((diff**2).mean(axis=0))

"""The Lee-Carter (1992) mortality model, written from scratch in NumPy.

    log m(x, t) = a_x + b_x * k_t + error

* a_x : average log death rate at age x (the age shape)
* k_t : a single time index for the overall level of mortality
* b_x : how strongly age x responds when k_t moves

The model is fitted by singular value decomposition (SVD) of the centred
log-rate matrix, using the usual constraints sum(b_x) = 1 and sum(k_t) = 0.
k_t is forecast as a random walk with drift.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class LeeCarter:
    ages: np.ndarray | None = None
    years: np.ndarray | None = None
    a: np.ndarray | None = None
    b: np.ndarray | None = None
    k: np.ndarray | None = None
    drift: float | None = None
    sigma: float | None = None
    explained_variance: float | None = None

    def fit(self, mx: pd.DataFrame) -> "LeeCarter":
        """Fit on an age x year DataFrame of central death rates mx."""
        self.ages = mx.index.to_numpy()
        self.years = mx.columns.to_numpy()
        log_m = np.log(mx.to_numpy())

        self.a = log_m.mean(axis=1)
        centred = log_m - self.a[:, None]
        U, s, Vt = np.linalg.svd(centred, full_matrices=False)

        b = U[:, 0]
        k = s[0] * Vt[0]
        # identification constraints: sum(b) = 1
        scale = b.sum()
        self.b = b / scale
        self.k = k * scale
        self.explained_variance = float(s[0] ** 2 / np.sum(s**2))

        dk = np.diff(self.k)
        self.drift = float(dk.mean())
        self.sigma = float(dk.std(ddof=1))
        return self

    def fitted_log_m(self) -> pd.DataFrame:
        vals = self.a[:, None] + np.outer(self.b, self.k)
        return pd.DataFrame(vals, index=self.ages, columns=self.years)

    def forecast_k(self, horizon: int, level: float = 0.95) -> pd.DataFrame:
        """Random walk with drift forecast of k_t, with a prediction interval
        that reflects innovation noise and drift-estimate uncertainty."""
        from statistics import NormalDist

        z = NormalDist().inv_cdf(0.5 + level / 2)
        h = np.arange(1, horizon + 1)
        n = len(self.k) - 1
        centre = self.k[-1] + self.drift * h
        se = self.sigma * np.sqrt(h + h**2 / n)
        years = self.years[-1] + h
        return pd.DataFrame(
            {"k": centre, "lower": centre - z * se, "upper": centre + z * se},
            index=years,
        )

    def forecast_mx(self, horizon: int) -> pd.DataFrame:
        """Central forecast of mx, age x future year."""
        k = self.forecast_k(horizon)["k"].to_numpy()
        log_m = self.a[:, None] + np.outer(self.b, k)
        return pd.DataFrame(
            np.exp(log_m), index=self.ages, columns=self.years[-1] + np.arange(1, horizon + 1)
        )


def random_walk_forecast(mx: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Naive benchmark: every age keeps its own average log-rate trend
    (a random walk with drift fitted separately at each age)."""
    log_m = np.log(mx.to_numpy())
    drift = (log_m[:, -1] - log_m[:, 0]) / (log_m.shape[1] - 1)
    h = np.arange(1, horizon + 1)
    out = log_m[:, -1][:, None] + np.outer(drift, h)
    return pd.DataFrame(np.exp(out), index=mx.index, columns=mx.columns[-1] + h)

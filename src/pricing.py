"""Net premiums for level term life insurance.

These are *net* (risk) premiums: expected claims only, no expenses,
commissions, lapses, profit margin or underwriting selection. They are
useful for comparing mortality assumptions, not as market prices.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import mx_to_qx
from src.lifetable import annuity_factor, term_insurance_epv


def cohort_qx(mx: pd.DataFrame, issue_age: int, issue_year: int, term: int) -> np.ndarray:
    """qx along a policyholder's path: age issue_age + j in year issue_year + j.

    ``mx`` is an age x year matrix (actual or forecast) of central death rates.
    """
    return np.array([
        float(mx_to_qx(mx.loc[issue_age + j, issue_year + j])) for j in range(term)
    ])


def net_level_premium(qx_path: np.ndarray, face: float, rate: float) -> float:
    """Annual premium, paid at the start of each year while alive, that equals
    the expected present value of a death benefit paid at the end of the year
    of death (equivalence principle)."""
    term = len(qx_path)
    benefit = term_insurance_epv(qx_path, 0, rate, term)
    annuity = annuity_factor(qx_path, 0, rate, term)
    return face * benefit / annuity


def pricing_shortfall(premium: float, actual_qx_path: np.ndarray, face: float, rate: float) -> float:
    """Present value, per policy issued, of claims minus premiums when the
    premium was set with one assumption but deaths followed ``actual_qx_path``.
    Positive = the insurer lost money."""
    term = len(actual_qx_path)
    claims = face * term_insurance_epv(actual_qx_path, 0, rate, term)
    premiums = premium * annuity_factor(actual_qx_path, 0, rate, term)
    return claims - premiums

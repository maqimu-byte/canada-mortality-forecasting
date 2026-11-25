"""Period life-table calculations from single-year qx."""
from __future__ import annotations

import numpy as np


def life_expectancy(qx: np.ndarray, a0: float = 0.1, radix: float = 100_000.0) -> np.ndarray:
    """Period life expectancy e_x at every age, from a vector of qx.

    Parameters
    ----------
    qx : probabilities of dying, ages 0..omega. The last age is treated as
        an open interval (everyone dies in it).
    a0 : average fraction of the first year lived by infants who die.
        Infant deaths cluster in the first weeks of life, so this is well
        below 0.5; 0.1 is a common working value for low-mortality countries.

    Notes
    -----
    Deaths at ages 1+ are assumed to fall, on average, at mid-year.
    In the open age interval, person-years lived are approximated by
    l_omega / m_omega with m_omega = -ln(1 - q) capped, which keeps the
    result finite when q_omega = 1.
    """
    qx = np.asarray(qx, dtype=float)
    n = len(qx)
    lx = radix * np.concatenate([[1.0], np.cumprod(1 - qx)[:-1]])
    dx = lx * qx
    ax = np.full(n, 0.5)
    ax[0] = a0
    Lx = lx - dx + ax * dx
    # open interval: use the mortality level of the last closed age
    m_last = -np.log(max(1 - qx[-2], 1e-12))
    Lx[-1] = lx[-1] / m_last if m_last > 0 else lx[-1] * 0.5
    Tx = np.cumsum(Lx[::-1])[::-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(lx > 0, Tx / lx, 0.0)


def e0(qx: np.ndarray, **kwargs) -> float:
    """Life expectancy at birth."""
    return float(life_expectancy(qx, **kwargs)[0])


def annuity_factor(qx: np.ndarray, age: int, rate: float, term: int) -> float:
    """Expected present value of 1 paid at the start of each year while alive
    (annuity-due), for up to ``term`` years from ``age``."""
    q = np.asarray(qx, dtype=float)[age : age + term]
    survival = np.concatenate([[1.0], np.cumprod(1 - q)[:-1]])
    v = (1 + rate) ** -np.arange(len(q))
    return float(np.sum(survival * v))


def term_insurance_epv(qx: np.ndarray, age: int, rate: float, term: int) -> float:
    """Expected present value of 1 paid at the end of the year of death,
    if death happens within ``term`` years of ``age``."""
    q = np.asarray(qx, dtype=float)[age : age + term]
    survival = np.concatenate([[1.0], np.cumprod(1 - q)[:-1]])
    v = (1 + rate) ** -np.arange(1, len(q) + 1)
    return float(np.sum(survival * q * v))

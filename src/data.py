"""Loading and reshaping Statistics Canada life-table data (Table 13-10-0837-01).

The processed CSV in ``data/processed/`` is in long format:

    year, sex, age, qx

where ``qx`` is the probability that a person aged ``age`` dies before
reaching ``age + 1`` (Canada, single years of age 0-110+, 1980-2024).

``fetch_qx_from_statcan`` shows how that file was built from the
Statistics Canada Web Data Service (WDS), so the data step can be re-run.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
QX_CSV = REPO_ROOT / "data" / "processed" / "canada_qx_1980_2024.csv"
EX_CSV = REPO_ROOT / "data" / "processed" / "canada_ex_official_1980_2024.csv"

# --- StatCan WDS coordinates for Table 13-10-0837-01 -------------------------
# coordinate = "{geo}.{age}.{sex}.{element}.0.0.0.0.0.0"
#   geo 1 = Canada
#   age member 1..111 = ages 0..110+
#   sex 2 = Males, 3 = Females
#   element 3 = qx (probability of dying), 8 = ex (life expectancy)
PRODUCT_ID = 13100837
WDS_URL = (
    "https://www150.statcan.gc.ca/t1/wds/rest/"
    "getDataFromCubePidCoordAndLatestNPeriods"
)
SEX_MEMBER = {"male": 2, "female": 3}
ELEMENT_QX = 3


def load_qx(path: Path | str = QX_CSV) -> pd.DataFrame:
    """Return the long-format qx table (year, sex, age, qx)."""
    return pd.read_csv(path)


def load_official_ex(path: Path | str = EX_CSV) -> pd.DataFrame:
    """Return StatCan's published life expectancy at ages 0 and 65."""
    return pd.read_csv(path)


def qx_matrix(df: pd.DataFrame, sex: str, max_age: int = 110) -> pd.DataFrame:
    """Reshape to an age x year matrix of qx for one sex."""
    sub = df[(df["sex"] == sex) & (df["age"] <= max_age)]
    return sub.pivot(index="age", columns="year", values="qx").sort_index()


def qx_to_mx(qx: np.ndarray | pd.DataFrame) -> np.ndarray | pd.DataFrame:
    """Convert qx to a central death rate mx, assuming a constant force of
    mortality within each year of age: mx = -ln(1 - qx)."""
    return -np.log1p(-np.clip(qx, None, 1 - 1e-12))


def mx_to_qx(mx: np.ndarray | pd.DataFrame) -> np.ndarray | pd.DataFrame:
    """Inverse of ``qx_to_mx``: qx = 1 - exp(-mx)."""
    return -np.expm1(-mx)


def fetch_qx_from_statcan(latest_n: int = 45, timeout: int = 60) -> pd.DataFrame:
    """Download Canada qx by single year of age and sex from the StatCan WDS.

    Requires internet access. Returns the same long format as ``load_qx``.
    """
    rows = []
    for sex, sex_member in SEX_MEMBER.items():
        payload = [
            {
                "productId": PRODUCT_ID,
                "coordinate": f"1.{age_member}.{sex_member}.{ELEMENT_QX}.0.0.0.0.0.0",
                "latestN": latest_n,
            }
            for age_member in range(1, 112)
        ]
        req = urllib.request.Request(
            WDS_URL,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            results = json.load(resp)
        rows.extend(_parse_wds(results, sex))
    out = pd.DataFrame(rows, columns=["year", "sex", "age", "qx"])
    return out.sort_values(["sex", "year", "age"], ascending=[False, True, True])


def _parse_wds(results: list[dict], sex: str) -> list[tuple]:
    """Parse a WDS response. Results can come back in any order, so each
    one is matched to its age through its own coordinate."""
    rows = []
    for item in results:
        if item.get("status") != "SUCCESS":
            raise RuntimeError(f"WDS request failed: {item}")
        obj = item["object"]
        age = int(obj["coordinate"].split(".")[1]) - 1
        for point in obj["vectorDataPoint"]:
            year = int(str(point["refPer"])[:4])
            rows.append((year, sex, age, round(float(point["value"]), 5)))
    return rows

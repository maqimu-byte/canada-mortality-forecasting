import numpy as np
import pandas as pd
import pytest

from src.data import _parse_wds, load_official_ex, load_qx, mx_to_qx, qx_matrix, qx_to_mx
from src.lee_carter import LeeCarter
from src.lifetable import e0, term_insurance_epv


def test_data_shape():
    df = load_qx()
    assert len(df) == 45 * 111 * 2
    assert df["qx"].between(0, 1).all()
    assert set(df["year"]) == set(range(1980, 2025))


@pytest.mark.parametrize("sex", ["male", "female"])
def test_e0_matches_statcan(sex):
    """Our life-table code should reproduce StatCan's published e0
    to within 0.02 years in every year."""
    q = qx_matrix(load_qx(), sex)
    official = load_official_ex().query("sex == @sex and age == 0").set_index("year")["ex"]
    ours = pd.Series({y: e0(q[y].to_numpy()) for y in q.columns})
    assert (ours - official).abs().max() < 0.02


def test_qx_mx_roundtrip():
    q = np.array([0.001, 0.05, 0.3])
    assert np.allclose(mx_to_qx(qx_to_mx(q)), q)


def test_lee_carter_constraints_and_recovery():
    rng = np.random.default_rng(0)
    ages, years = np.arange(0, 50), np.arange(2000, 2030)
    a = np.linspace(-8, -3, len(ages))
    b = np.full(len(ages), 1 / len(ages))
    k = np.linspace(10, -10, len(years))
    log_m = a[:, None] + np.outer(b, k) + rng.normal(0, 1e-4, (len(ages), len(years)))
    lc = LeeCarter().fit(pd.DataFrame(np.exp(log_m), index=ages, columns=years))
    assert np.isclose(lc.b.sum(), 1)
    assert np.isclose(lc.k.sum(), 0, atol=1e-6)
    assert np.allclose(lc.k, k, atol=0.05)
    assert lc.explained_variance > 0.99


def test_term_epv_simple():
    q = np.full(10, 0.01)
    # one-year term at 0% interest pays q
    assert np.isclose(term_insurance_epv(q, 0, 0.0, 1), 0.01)


def test_parse_wds_maps_by_coordinate():
    results = [
        {"status": "SUCCESS", "object": {"coordinate": "1.3.2.3.0.0.0.0.0.0",
         "vectorDataPoint": [{"refPer": "2024-01-01", "value": 0.0002}]}},
        {"status": "SUCCESS", "object": {"coordinate": "1.1.2.3.0.0.0.0.0.0",
         "vectorDataPoint": [{"refPer": "2024-01-01", "value": 0.0045}]}},
    ]
    rows = _parse_wds(results, "male")
    assert rows == [(2024, "male", 2, 0.0002), (2024, "male", 0, 0.0045)]


def test_ml_features_use_no_future_data():
    """Changing data after the origin year must not change the inputs."""
    from src.ml import make_features
    q = qx_matrix(load_qx(), "male").loc[0:100]
    lm = np.log(qx_to_mx(q))
    before = make_features(lm, 2010, 3, "male")
    lm2 = lm.copy()
    lm2.loc[:, 2011:] = 0.0
    pd.testing.assert_frame_equal(before, make_features(lm2, 2010, 3, "male"))


def test_ml_training_targets_stop_at_last_year():
    from src.ml import build_training_set
    lm = {"male": np.log(qx_to_mx(qx_matrix(load_qx(), "male").loc[0:100]))}
    X, y = build_training_set(lm, 1990, 2000, horizon=10)
    assert X["h"].max() <= 10
    # origin + h <= 2000 for every row: 10 origins, horizons 10..1 -> 55 per age
    assert len(X) == 55 * 101

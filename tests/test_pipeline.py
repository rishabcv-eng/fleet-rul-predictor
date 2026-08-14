"""
Correctness checks for the data pipeline and API.

Run with:  pytest -q

The most important test here is test_no_lifetime_leakage. An earlier version of
this project scored an RMSE of 1.8 because a feature encoded each unit's total
lifetime -- which is the answer. That test exists so it cannot come back.
"""

import numpy as np
import pytest

from backend import data as D


@pytest.fixture(scope="module")
def train():
    try:
        return D.add_training_labels(D.load_raw("train"))
    except FileNotFoundError:
        pytest.skip("No dataset. Run: python scripts/generate_data.py")


def test_rul_hits_zero_at_failure(train):
    """The final cycle of a run-to-failure unit must have RUL 0."""
    final = train.groupby("unit")["rul"].min()
    assert (final == 0).all()


def test_rul_decreases_over_time(train):
    """Remaining life never goes up as a unit accumulates cycles."""
    for _unit, group in train.sort_values("cycle").groupby("unit"):
        assert group["rul"].is_monotonic_decreasing


def test_rul_respects_cap(train):
    assert train["rul"].max() <= D.RUL_CAP


def test_constant_sensors_are_dropped(train):
    sensors = D.useful_sensors(train)
    assert len(sensors) < len(D.SENSOR_NAMES)
    assert all(train[sensor].std() > 0 for sensor in sensors)


def test_no_lifetime_leakage(train):
    """
    No feature may correlate near-perfectly with the unit's total lifetime.

    A feature that encodes how long a unit ultimately lasts is leakage: at
    prediction time on live equipment, that value is unknowable.
    """
    sensors = D.useful_sensors(train)
    features = D.build_features(train, sensors)
    columns = D.feature_columns(features, sensors)

    lifetime = features.groupby("unit")["cycle"].transform("max").to_numpy(dtype=float)

    for column in columns:
        values = features[column].to_numpy(dtype=float)
        if np.std(values) < 1e-9:
            continue
        correlation = abs(np.corrcoef(values, lifetime)[0, 1])
        assert correlation < 0.95, f"{column} looks like it leaks unit lifetime"


def test_features_use_no_future_information(train):
    """
    Truncating a unit's history must not change the features already computed.

    If it does, some feature is looking ahead -- which works in a notebook and
    breaks the moment the model sees equipment that has not failed yet.
    """
    sensors = D.useful_sensors(train)
    one_unit = train[train["unit"] == train["unit"].iloc[0]].copy()

    full = D.build_features(one_unit, sensors)
    truncated = D.build_features(one_unit.head(60), sensors)
    columns = D.feature_columns(full, sensors)

    np.testing.assert_allclose(
        full[columns].head(60).to_numpy(dtype=float),
        truncated[columns].to_numpy(dtype=float),
        rtol=1e-6,
        atol=1e-6,
    )


def test_last_cycle_per_unit_picks_the_latest(train):
    latest = D.last_cycle_per_unit(train)
    assert len(latest) == train["unit"].nunique()
    for row in latest.itertuples():
        assert row.cycle == train[train["unit"] == row.unit]["cycle"].max()


# --------------------------------------------------------------------- API

@pytest.fixture(scope="module")
def client():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from backend.api import MODEL_PATH, app

    if not MODEL_PATH.exists():
        pytest.skip("No trained model. Run: python -m backend.train")
    return TestClient(app)


def test_health_endpoint(client):
    payload = client.get("/api/health").json()
    assert payload["status"] == "ok"


def test_fleet_is_sorted_by_urgency(client):
    payload = client.get("/api/fleet").json()
    ruls = [asset["predicted_rul"] for asset in payload["assets"]]
    assert ruls == sorted(ruls)
    assert payload["summary"]["total"] == len(payload["assets"])


def test_fleet_statuses_match_thresholds(client):
    from backend.api import CRITICAL_THRESHOLD, WARNING_THRESHOLD

    for asset in client.get("/api/fleet").json()["assets"]:
        rul = asset["predicted_rul"]
        expected = (
            "critical" if rul <= CRITICAL_THRESHOLD
            else "warning" if rul <= WARNING_THRESHOLD
            else "healthy"
        )
        assert asset["status"] == expected


def test_asset_detail_series_align(client):
    unit = client.get("/api/fleet").json()["assets"][0]["unit"]
    asset = client.get(f"/api/asset/{unit}").json()

    length = len(asset["cycles"])
    assert len(asset["rul_trajectory"]) == length
    for series in asset["sensors"].values():
        assert len(series["raw"]) == length
        assert len(series["smoothed"]) == length


def test_unknown_asset_returns_404(client):
    assert client.get("/api/asset/999999").status_code == 404

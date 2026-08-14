"""
FastAPI service for the predictive maintenance dashboard.

    uvicorn backend.api:app --reload

Interactive API docs are generated automatically at /docs.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend import data as D

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "rul_model.joblib"
METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

# Maintenance urgency bands, in cycles of remaining life.
CRITICAL_THRESHOLD = 25
WARNING_THRESHOLD = 60

# Cosmetic only: the dataset is turbofan engines, but the problem statement is
# military equipment, so we label units as assets an operator would recognise.
ASSET_TYPES = [
    ("Transport Aircraft", "Engine"),
    ("Armoured Vehicle", "Powerpack"),
    ("Field Generator", "Turbine"),
    ("Naval Patrol Craft", "Gas Turbine"),
    ("Utility Helicopter", "Engine"),
]
BASES = ["Forward Base Alpha", "Depot Bravo", "Airfield Charlie",
         "Naval Station Delta", "Garrison Echo"]

app = FastAPI(
    title="Predictive Maintenance API",
    description="Remaining-useful-life predictions for a fleet of equipment.",
    version="1.0.0",
)


@functools.lru_cache(maxsize=1)
def load_bundle() -> dict:
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="Model not trained yet. Run: python -m backend.train",
        )
    return joblib.load(MODEL_PATH)


@functools.lru_cache(maxsize=1)
def load_fleet() -> pd.DataFrame:
    """
    Score every unit in the test set and cache the result.

    Test units are truncated mid-life, which is exactly the production
    situation: equipment currently in service, failure date unknown.
    """
    bundle = load_bundle()
    frame = D.build_features(D.load_raw("test"), bundle["sensors"])

    predictions = bundle["model"].predict(
        frame[bundle["features"]].to_numpy(dtype=np.float32)
    )
    frame["predicted_rul"] = np.clip(predictions, 0, bundle["rul_cap"])
    return frame


def describe_asset(unit_id: int) -> dict:
    asset_type, component = ASSET_TYPES[unit_id % len(ASSET_TYPES)]
    return {
        "asset_id": f"{asset_type.split()[0][:3].upper()}-{unit_id:03d}",
        "asset_type": asset_type,
        "component": component,
        "base": BASES[unit_id % len(BASES)],
    }


def classify(rul: float) -> tuple[str, int]:
    """Map remaining life to a status band and a 0-100 health score."""
    if rul <= CRITICAL_THRESHOLD:
        status = "critical"
    elif rul <= WARNING_THRESHOLD:
        status = "warning"
    else:
        status = "healthy"
    cap = load_bundle()["rul_cap"]
    return status, int(round(100 * min(rul, cap) / cap))


@app.get("/api/health")
def health() -> dict:
    """Liveness check -- also tells you whether a trained model is present."""
    return {"status": "ok", "model_trained": MODEL_PATH.exists()}


@app.get("/api/metrics")
def metrics() -> dict:
    """Model evaluation metrics produced during training."""
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail="Train the model first.")
    return json.loads(METRICS_PATH.read_text())


@app.get("/api/fleet")
def fleet() -> dict:
    """
    Every asset with its current risk, sorted most urgent first.

    This is what the dashboard's main table renders.
    """
    latest = D.last_cycle_per_unit(load_fleet())
    assets = []

    for row in latest.itertuples():
        status, health_score = classify(row.predicted_rul)
        assets.append(
            {
                **describe_asset(row.unit),
                "unit": int(row.unit),
                "cycles_in_service": int(row.cycle),
                "predicted_rul": round(float(row.predicted_rul), 1),
                "health_score": health_score,
                "status": status,
            }
        )

    assets.sort(key=lambda asset: asset["predicted_rul"])
    counts = {level: 0 for level in ("critical", "warning", "healthy")}
    for asset in assets:
        counts[asset["status"]] += 1

    return {
        "assets": assets,
        "summary": {
            "total": len(assets),
            **counts,
            "mean_health": round(
                sum(a["health_score"] for a in assets) / max(len(assets), 1), 1
            ),
        },
    }


@app.get("/api/asset/{unit}")
def asset_detail(unit: int) -> dict:
    """
    Full history for one asset: RUL trajectory and sensor traces.

    The RUL trajectory is the useful plot -- a line falling steadily towards
    zero is a unit degrading on schedule; a sudden drop is worth a look.
    """
    frame = load_fleet()
    history = frame[frame["unit"] == unit].sort_values("cycle")
    if history.empty:
        raise HTTPException(status_code=404, detail=f"No asset with unit id {unit}")

    current_rul = float(history["predicted_rul"].iloc[-1])
    status, health_score = classify(current_rul)

    # Show the sensors the model leaned on most, not all fifteen.
    top_sensors = ranked_sensors()[:6]

    return {
        **describe_asset(unit),
        "unit": unit,
        "cycles_in_service": int(history["cycle"].iloc[-1]),
        "predicted_rul": round(current_rul, 1),
        "health_score": health_score,
        "status": status,
        "cycles": [int(c) for c in history["cycle"]],
        "rul_trajectory": [round(float(v), 1) for v in history["predicted_rul"]],
        "sensors": {
            sensor: {
                "raw": [round(float(v), 3) for v in history[sensor]],
                "smoothed": [
                    round(float(v), 3) for v in history[f"{sensor}_roll_mean"]
                ],
            }
            for sensor in top_sensors
        },
    }


@functools.lru_cache(maxsize=1)
def ranked_sensors() -> list[str]:
    """Sensors ordered by how much the model relied on them."""
    bundle = load_bundle()
    if not METRICS_PATH.exists():
        return bundle["sensors"]

    importances = json.loads(METRICS_PATH.read_text()).get("top_features", [])
    ordered: list[str] = []
    for entry in importances:
        for sensor in bundle["sensors"]:
            if entry["feature"].startswith(sensor) and sensor not in ordered:
                ordered.append(sensor)
    # Anything not ranked goes on the end so we never return an empty list.
    return ordered + [s for s in bundle["sensors"] if s not in ordered]


@app.get("/api/importance")
def importance() -> dict:
    """Which features drive the predictions -- the explainability view."""
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail="Train the model first.")
    return {"features": json.loads(METRICS_PATH.read_text()).get("top_features", [])}


# Serve the dashboard from the same origin as the API, so there is no CORS
# setup and no second server to run.
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

    @app.get("/")
    def dashboard() -> FileResponse:
        return FileResponse(FRONTEND_DIR / "index.html")

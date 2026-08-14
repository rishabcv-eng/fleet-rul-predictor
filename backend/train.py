"""
Train the Remaining Useful Life (RUL) model.

    python -m backend.train

Saves the fitted model plus its metadata to models/rul_model.joblib and
writes evaluation metrics to models/metrics.json.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict

from backend import data as D

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_PATH = MODEL_DIR / "rul_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"


def phm_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    The asymmetric scoring function from the PHM08 prognostics challenge.

    Predicting failure *late* is penalised much harder than predicting it
    early, because a late prediction means the equipment broke before you
    serviced it. Lower is better.
    """
    error = y_pred - y_true
    late = error > 0
    scores = np.where(late, np.exp(error / 10.0) - 1.0, np.exp(-error / 13.0) - 1.0)
    return float(scores.sum())


def build_model() -> HistGradientBoostingRegressor:
    """
    Gradient-boosted trees.

    Chosen over a neural network on purpose: this dataset is small, trees
    handle the mixed feature scales without normalisation, they train in
    seconds on a laptop, and the result is reproducible. An LSTM would be
    the natural next step if you want sequence modelling.
    """
    return HistGradientBoostingRegressor(
        max_iter=400,
        learning_rate=0.06,
        max_depth=6,
        min_samples_leaf=25,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42,
    )


def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_pred = np.clip(y_pred, 0, D.RUL_CAP)
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "phm_score": phm_score(y_true, y_pred),
        "n": int(len(y_true)),
    }
    print(
        f"  {name:<28} RMSE {metrics['rmse']:6.2f}   "
        f"MAE {metrics['mae']:6.2f}   R2 {metrics['r2']:5.3f}"
    )
    return metrics


def main() -> None:
    import joblib  # imported here so `--help` works without the dependency

    print("Loading data...")
    train_raw = D.load_raw("train")
    train = D.add_training_labels(train_raw)
    sensors = D.useful_sensors(train)
    dropped = len(D.SENSOR_NAMES) - len(sensors)
    print(f"  {train['unit'].nunique()} units, {len(train):,} cycles")
    print(f"  {len(sensors)} informative sensors ({dropped} constant ones dropped)")

    print("Engineering features...")
    train = D.build_features(train, sensors)
    features = D.feature_columns(train, sensors)
    X = train[features].to_numpy(dtype=np.float32)
    y = train["rul"].to_numpy(dtype=np.float32)
    groups = train["unit"].to_numpy()
    print(f"  {len(features)} features")

    # Group the folds by unit. Splitting rows at random would put cycles from
    # the same engine in both train and validation, and consecutive cycles are
    # nearly identical -- the score would look great and mean nothing.
    print("Cross-validating (grouped by unit, 5 folds)...")
    started = time.time()
    oof_predictions = cross_val_predict(
        build_model(), X, y, groups=groups, cv=GroupKFold(n_splits=5), n_jobs=-1
    )
    cv_metrics = evaluate("cross-validation", y, oof_predictions)
    print(f"  took {time.time() - started:.1f}s")

    print("Fitting final model on all data...")
    model = build_model().fit(X, y)

    # Held-out check: test units are cut off mid-life, so we score only their
    # most recent cycle against the true remaining life.
    holdout_metrics = None
    truth = D.true_test_rul()
    if truth is not None:
        test = D.build_features(D.load_raw("test"), sensors)
        latest = D.last_cycle_per_unit(test)
        predictions = model.predict(latest[features].to_numpy(dtype=np.float32))
        holdout_metrics = evaluate(
            "held-out test units", np.clip(truth, 0, D.RUL_CAP), predictions
        )

    importances = permutation_importance_summary(model, X, y, features)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "features": features,
            "sensors": sensors,
            "rul_cap": D.RUL_CAP,
            "trained_at": pd.Timestamp.now("UTC").isoformat(),
        },
        MODEL_PATH,
    )
    METRICS_PATH.write_text(
        json.dumps(
            {
                "cross_validation": cv_metrics,
                "holdout": holdout_metrics,
                "n_features": len(features),
                "n_train_units": int(train["unit"].nunique()),
                "top_features": importances,
            },
            indent=2,
        )
    )
    print(f"\nSaved model  -> {MODEL_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")
    print("\nNext: uvicorn backend.api:app --reload")


def permutation_importance_summary(model, X, y, features, top: int = 12) -> list[dict]:
    """Which features actually mattered, measured by shuffling each one."""
    from sklearn.inspection import permutation_importance

    # Subsample -- permutation importance is slow and we only need a ranking.
    rng = np.random.default_rng(0)
    index = rng.choice(len(X), size=min(3000, len(X)), replace=False)
    result = permutation_importance(
        model, X[index], y[index], n_repeats=3, random_state=0, n_jobs=-1
    )
    order = np.argsort(result.importances_mean)[::-1][:top]
    return [
        {"feature": features[i], "importance": float(result.importances_mean[i])}
        for i in order
    ]


if __name__ == "__main__":
    main()

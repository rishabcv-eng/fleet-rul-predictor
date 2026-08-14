"""Loading and feature engineering for the C-MAPSS-format sensor data."""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Column layout of the C-MAPSS text files: unit, cycle, 3 operating settings,
# then 19 sensor channels.
SENSOR_NAMES = [
    "fan_inlet_temp", "lpc_outlet_temp", "hpc_outlet_temp", "lpt_outlet_temp",
    "fan_inlet_pressure", "bypass_duct_pressure", "hpc_outlet_pressure",
    "physical_fan_speed", "physical_core_speed", "engine_pressure_ratio",
    "static_pressure", "fuel_flow_ratio", "corrected_fan_speed",
    "corrected_core_speed", "bypass_ratio", "burner_fuel_air_ratio",
    "bleed_enthalpy", "hpt_coolant_bleed", "lpt_coolant_bleed",
]
COLUMNS = ["unit", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"] + SENSOR_NAMES

# Past this many cycles from failure we treat remaining life as "plenty".
# Without this the model wastes capacity trying to tell 300 cycles-to-failure
# apart from 280, which nobody cares about, and which the sensors cannot
# actually distinguish while the unit is still healthy.
RUL_CAP = 125

ROLLING_WINDOW = 20


def load_raw(split: str) -> pd.DataFrame:
    """Read train_FD001.txt or test_FD001.txt into a DataFrame."""
    path = RAW_DIR / f"{split}_FD001.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found.\n"
            "Run `python scripts/generate_data.py` for synthetic data, or "
            "`python scripts/download_data.py` for the real NASA dataset."
        )
    frame = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS)
    return frame.astype({"unit": int, "cycle": int})


def add_training_labels(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Add the RUL target column to run-to-failure training data.

    Every training unit runs until it breaks, so the last cycle we see for a
    unit is the moment of failure: RUL at any earlier cycle is simply the
    distance to that last cycle.
    """
    frame = frame.copy()
    last_cycle = frame.groupby("unit")["cycle"].transform("max")
    frame["rul"] = (last_cycle - frame["cycle"]).clip(upper=RUL_CAP)
    return frame


def useful_sensors(frame: pd.DataFrame) -> list[str]:
    """
    Drop sensors that never move.

    Some channels are constant across the entire fleet -- they carry no
    information about wear and only add noise and training time.
    """
    return [name for name in SENSOR_NAMES if frame[name].std() > 1e-6]


def build_features(frame: pd.DataFrame, sensors: list[str]) -> pd.DataFrame:
    """
    Turn raw per-cycle readings into features the model can learn from.

    A single snapshot of a sensor is weak evidence -- absolute values vary
    between units because of manufacturing differences. What actually
    signals wear is *change over time*, so for each sensor we add:

      _roll_mean  smoothed level, which cuts sensor noise
      _roll_std   volatility, which tends to rise as things wear out
      _delta      drift from the unit's own healthy baseline
      _slope      short-term rate of change

    Everything is computed per unit and only from past readings, so no
    information leaks backwards from the future.
    """
    frame = frame.sort_values(["unit", "cycle"]).copy()
    grouped = frame.groupby("unit")

    # Deliberately NOT included: cycle / max(cycle). That feature leaks the
    # answer -- in training data a unit's last cycle IS the failure cycle, so
    # dividing by it hands the model the very thing it is supposed to predict.
    # It scores beautifully in validation and collapses on real equipment.
    new_columns: dict[str, pd.Series] = {}

    for sensor in sensors:
        series = grouped[sensor]
        rolling = series.rolling(ROLLING_WINDOW, min_periods=1)

        new_columns[f"{sensor}_roll_mean"] = rolling.mean().reset_index(level=0, drop=True)
        new_columns[f"{sensor}_roll_std"] = (
            rolling.std().reset_index(level=0, drop=True).fillna(0.0)
        )
        # Baseline = average of the first few cycles, when the unit was healthy.
        baseline = series.transform(lambda s: s.head(5).mean())
        new_columns[f"{sensor}_delta"] = frame[sensor] - baseline
        new_columns[f"{sensor}_slope"] = (
            series.diff(5).reset_index(level=0, drop=True).fillna(0.0) / 5.0
        )

    return pd.concat([frame, pd.DataFrame(new_columns, index=frame.index)], axis=1)


def feature_columns(frame: pd.DataFrame, sensors: list[str]) -> list[str]:
    """The exact column list fed to the model, in a stable order."""
    columns = ["cycle"]
    for sensor in sensors:
        columns += [
            sensor,
            f"{sensor}_roll_mean",
            f"{sensor}_roll_std",
            f"{sensor}_delta",
            f"{sensor}_slope",
        ]
    return [column for column in columns if column in frame.columns]


def last_cycle_per_unit(frame: pd.DataFrame) -> pd.DataFrame:
    """Most recent reading for each unit -- what you'd score in production."""
    return frame.loc[frame.groupby("unit")["cycle"].idxmax()].reset_index(drop=True)


def true_test_rul() -> np.ndarray | None:
    """Ground-truth remaining life for the test units, if the file exists."""
    path = RAW_DIR / "RUL_FD001.txt"
    if not path.exists():
        return None
    return np.loadtxt(path).astype(int)

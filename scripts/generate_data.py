"""
Generate a synthetic run-to-failure dataset modelled on NASA's C-MAPSS
turbofan degradation benchmark.

Each "unit" is one piece of equipment that runs for a number of operating
cycles and then fails. Sensors drift as the unit degrades, some linearly,
some exponentially, some not at all (those are the useless sensors -- real
datasets have them too, and dropping them is part of the job).

Output: data/raw/train_FD001.txt and data/raw/test_FD001.txt in the same
whitespace-delimited layout as the real C-MAPSS files, so this code works
unchanged if you swap in the real data.
"""

import argparse
from pathlib import Path

import numpy as np

# Sensor behaviour: (name, baseline, noise_sd, degradation_mode, magnitude)
# modes: "up" rises with wear, "down" falls with wear, "flat" is uninformative
SENSORS = [
    ("fan_inlet_temp",        518.67, 0.00,  "flat", 0.0),
    ("lpc_outlet_temp",       642.68, 0.50,  "up",   6.0),
    ("hpc_outlet_temp",      1590.52, 6.00,  "up",  40.0),
    ("lpt_outlet_temp",      1408.93, 5.00,  "up",  28.0),
    ("fan_inlet_pressure",     14.62, 0.00,  "flat", 0.0),
    ("bypass_duct_pressure",   21.61, 0.02,  "down", 0.3),
    ("hpc_outlet_pressure",   553.37, 0.90,  "down", 9.0),
    ("physical_fan_speed",   2388.06, 0.07,  "up",   0.6),
    ("physical_core_speed",   9065.24, 25.0, "up",  70.0),
    ("engine_pressure_ratio",   1.30, 0.00,  "flat", 0.0),
    ("static_pressure",        47.35, 0.30,  "up",   1.8),
    ("fuel_flow_ratio",       521.66, 0.80,  "down", 8.0),
    ("corrected_fan_speed",  2388.02, 0.07,  "up",   0.5),
    ("corrected_core_speed",  8138.62, 22.0, "down", 60.0),
    ("bypass_ratio",            8.44, 0.04,  "up",   0.35),
    ("burner_fuel_air_ratio",   0.03, 0.00,  "flat", 0.0),
    ("bleed_enthalpy",        392.00, 1.20,  "down", 12.0),
    ("hpt_coolant_bleed",      38.86, 0.10,  "down", 0.9),
    ("lpt_coolant_bleed",      23.32, 0.08,  "up",   0.7),
]


def degradation_curve(n_cycles: int, rng: np.random.Generator) -> np.ndarray:
    """
    Wear fraction from 0 (healthy) to 1 (failed).

    Real equipment does not degrade linearly: it holds steady for a while,
    then deteriorates faster and faster. We model that with an exponential
    that stays near zero for the first chunk of life.
    """
    t = np.linspace(0.0, 1.0, n_cycles)
    healthy_fraction = rng.uniform(0.35, 0.60)  # how long it stays "fine"
    sharpness = rng.uniform(2.5, 4.5)

    wear = np.zeros(n_cycles)
    degrading = t > healthy_fraction
    scaled = (t[degrading] - healthy_fraction) / (1.0 - healthy_fraction)
    wear[degrading] = scaled ** sharpness
    return wear


def simulate_unit(unit_id: int, rng: np.random.Generator) -> np.ndarray:
    """Simulate one unit from new until failure. Returns a 2D array of rows."""
    n_cycles = int(rng.normal(206, 46))
    n_cycles = int(np.clip(n_cycles, 128, 362))

    wear = degradation_curve(n_cycles, rng)
    cycles = np.arange(1, n_cycles + 1)

    # Three operating-condition columns, as in the real dataset.
    op1 = rng.normal(0.0, 0.0020, n_cycles)
    op2 = rng.normal(0.0, 0.0003, n_cycles)
    op3 = np.full(n_cycles, 100.0)

    columns = [np.full(n_cycles, unit_id, dtype=float), cycles.astype(float),
               op1, op2, op3]

    for _name, baseline, noise_sd, mode, magnitude in SENSORS:
        signal = np.full(n_cycles, baseline, dtype=float)
        if mode == "up":
            signal = signal + magnitude * wear
        elif mode == "down":
            signal = signal - magnitude * wear

        # Unit-to-unit manufacturing variation, then per-reading sensor noise.
        signal = signal + rng.normal(0.0, noise_sd * 0.5)
        signal = signal + rng.normal(0.0, noise_sd, n_cycles)
        columns.append(signal)

    return np.column_stack(columns)


def build_split(n_units: int, rng: np.random.Generator, truncate: bool):
    """
    Build a set of units.

    Training units run all the way to failure. Test units are cut off at a
    random point mid-life, which is the realistic case: the equipment is
    still in service and you must predict how much life is left.
    """
    rows, remaining_life = [], []
    for unit_id in range(1, n_units + 1):
        unit = simulate_unit(unit_id, rng)
        if truncate:
            total = len(unit)
            # Leave at least 15 cycles of life so the task stays meaningful.
            cut = rng.integers(int(total * 0.35), total - 15)
            remaining_life.append(total - cut)
            unit = unit[:cut]
        rows.append(unit)
    return np.vstack(rows), np.array(remaining_life)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-units", type=int, default=100)
    parser.add_argument("--test-units", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data" / "raw")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    train, _ = build_split(args.train_units, rng, truncate=False)
    test, test_rul = build_split(args.test_units, rng, truncate=True)

    fmt = "%.5f"
    np.savetxt(args.out / "train_FD001.txt", train, fmt=fmt)
    np.savetxt(args.out / "test_FD001.txt", test, fmt=fmt)
    np.savetxt(args.out / "RUL_FD001.txt", test_rul, fmt="%d")

    print(f"Wrote {len(train):,} training rows across {args.train_units} units")
    print(f"Wrote {len(test):,} test rows across {args.test_units} units")
    print(f"Output directory: {args.out}")
    print("\nNOTE: this is simulated data. To use the real NASA C-MAPSS "
          "benchmark instead, run scripts/download_data.py")


if __name__ == "__main__":
    main()

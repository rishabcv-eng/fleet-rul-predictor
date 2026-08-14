# Contributing

Thanks for taking an interest. This is a small project, so the process is
correspondingly small: open an issue if you want to discuss something first,
otherwise send a pull request.

## Getting set up

Requires Python 3.10 or newer.

```bash
git clone https://github.com/YOUR-USERNAME/predictive-maintenance.git
cd predictive-maintenance

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/generate_data.py    # create the dataset
python -m backend.train            # train the model (~30 seconds)
uvicorn backend.api:app --reload   # start the server
```

The dataset and trained model are both generated locally and are not committed
(see `.gitignore`). If tests skip with "No dataset", you have not run
`scripts/generate_data.py` yet.

## Before you open a pull request

```bash
pip install pytest
pytest -q
```

All tests must pass. If you change feature engineering, the model, or an API
response shape, add a test covering it.

## The one rule that matters: no leakage

`tests/test_pipeline.py::test_no_lifetime_leakage` exists because an early
version of this project scored RMSE 1.8 — a feature computed as
`cycle / max(cycle)` per unit handed the model the very thing it was meant to
predict, since in training data a unit's last cycle *is* its failure cycle.

So, when adding or changing a feature:

- A feature may only use information available **at or before the current
  cycle** for that unit. No `max()`, no `len()`, no reverse-indexing, no
  normalising by anything computed over the unit's whole history.
- Never delete or weaken the leakage test to make a change pass. If it fires,
  the feature is wrong, not the test.
- Evaluate with cross-validation **grouped by unit**. Consecutive cycles from
  one engine are nearly identical; a random row split puts near-duplicates on
  both sides and produces a score that looks excellent and means nothing.

A suspiciously good metric is a bug report, not a result. If RMSE drops sharply,
find out why before celebrating.

## Changes that affect the model

If a change moves the numbers, include the before/after metrics from
`models/metrics.json` in the pull request description, along with what you
believe caused the difference. The table in the README should be updated in the
same PR.

## Code style

Match what's already there rather than introducing new tooling:

- Standard library imports first, then third-party, then `backend.*`.
- A module docstring on every module saying what it is for.
- Comments explain *why*, not *what* — the existing comments on `RUL_CAP` and
  the dropped constant sensors are the model to follow.
- Descriptive names (`ROLLING_WINDOW`, not `w`). Sensor channels keep their
  C-MAPSS-derived names.
- Type hints on public functions where they clarify the contract.

The frontend is deliberately a single HTML file with hand-written CSS, JS and
SVG charts — no build step, no npm, no CDN. Please keep it that way; a
dependency-free clone-and-run dashboard is a feature of this project, not an
oversight.

## Good places to start

The README's "What I'd add next" section lists the open directions: a sequence
model (LSTM/GRU), prediction intervals, anomaly detection for sudden faults, and
work-order generation. Any of those is a welcome contribution — open an issue
first so we can agree on the shape of it.

## Licence

By contributing, you agree that your contributions are licensed under the MIT
Licence, the same terms as the rest of the project.

# Getting this onto your GitHub

Written for a first repository. If a step already looks familiar, skip it.

## 1. Install Git and check it works

```bash
git --version
```

If that errors, install Git from https://git-scm.com/downloads, then reopen your
terminal.

Set your name and email once (they get stamped on every commit):

```bash
git config --global user.name "Rishab"
git config --global user.email "rishabcv@gmail.com"
```

Use the same email as your GitHub account and your commits will link to your
profile — that's what fills in the green contribution squares.

## 2. Check the project runs before you push

Do not skip this. A repo that fails on first run is worse than no repo.

```bash
cd predictive-maintenance

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/generate_data.py
python -m backend.train
pytest -q
uvicorn backend.api:app --reload
```

Open http://localhost:8000 and confirm the dashboard loads. Stop the server with
`Ctrl+C`.

## 3. Make the first commit

```bash
git init
git add .
git status
```

Read the `git status` output before committing. You should see source files, the
README, and `docs/`. You should **not** see `data/raw/`, `.venv/`, or
`models/*.joblib` — `.gitignore` excludes them, because generated files and
virtual environments don't belong in a repo. If they show up anyway, stop and
check that `.gitignore` is in the project root.

```bash
git commit -m "Predictive maintenance system for equipment RUL prediction"
```

## 4. Create the repo on GitHub

Go to https://github.com/new and:

- **Repository name:** `fleet-rul-predictor`
- **Description:** `Predicting remaining useful life of equipment from sensor telemetry — ML pipeline + dashboard`
- **Public**
- **Do not** tick "Add a README", "Add .gitignore", or "Choose a license" — you
  already have all three locally, and ticking them creates a conflict you'd then
  have to resolve.

Click **Create repository**.

## 5. Push

GitHub shows you these commands after creating the repo. For this project the
remote is already set to `rishabcv-eng/fleet-rul-predictor`:

```bash
git remote add origin https://github.com/rishabcv-eng/fleet-rul-predictor.git
git branch -M main
git push -u origin main
```

When it asks for a password, your GitHub account password will **not** work. You
need a personal access token:

1. https://github.com/settings/tokens → **Generate new token (classic)**
2. Give it a name, set an expiry, tick the **`repo`** scope
3. Generate it and copy the token immediately — it is shown once
4. Paste it as the password

(Alternative: install the GitHub CLI, run `gh auth login`, and it handles this for
you.)

## 6. Finish the repo page

Refresh your repo on GitHub. Two small things that make a real difference:

**Topics.** Click the gear next to "About" and add: `machine-learning`,
`predictive-maintenance`, `fastapi`, `scikit-learn`, `dashboard`, `python`,
`time-series`. These make the repo findable.

**Check the screenshot rendered.** The README references `docs/dashboard.png`. If
you see a broken image icon, the file didn't get committed — run
`git add docs/ && git commit -m "Add dashboard screenshot" && git push`.

## 7. Fix your commit name

The first commits on this repo were authored as `unknown` because `user.name` was
never set — only `user.email` was. Set it once so future commits carry your name:

```bash
git config --global user.name "Rishab"
```

Past commits keep the old name; that's fine, and not worth rewriting history for.

---

## Working on it from here

The everyday loop is three commands:

```bash
git add .
git commit -m "Describe what changed"
git push
```

Commit when you finish something that works, not when you finish for the day.
"Add prediction intervals to RUL output" is a useful message; "update" and "fixes"
are not — six months from now they tell you nothing.

## If something goes wrong

**Committed a file you didn't mean to** (before pushing):

```bash
git rm --cached path/to/file
echo "path/to/file" >> .gitignore
git commit -m "Remove file that should not be tracked"
```

**Pushed something secret** — an API key, a password. Don't just delete it in a new
commit; it stays in the history. Revoke the credential immediately, then rewrite
history or delete and recreate the repo. Assume anything pushed publicly was seen.

**`error: failed to push some refs`** — the remote has commits you don't have
locally. Usually this means you ticked one of the "Add a README" boxes in step 4.
Fix:

```bash
git pull --rebase origin main
git push
```

**Want to undo the last commit but keep your changes:**

```bash
git reset --soft HEAD~1
```

## Talking about the project

When someone asks about it — in an interview, on your profile — the interesting
part is not that you trained a model. It's the leakage bug documented in the
README: you built a feature that scored RMSE 1.8, recognised the score was too good
to be true, found that it encoded the answer, and removed it. Noticing that a
result is suspiciously good is a more valuable instinct than getting a good result,
and most people who build a first ML project never demonstrate it.

The second thing worth mentioning: cross-validation grouped by unit rather than
split at random. It's a small detail that shows you understood why the naive
approach inflates the score.

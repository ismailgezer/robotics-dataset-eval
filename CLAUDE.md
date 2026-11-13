# robotics-dataset-eval

Human-evaluation website (Streamlit, `app_user.py` + `rde_core.py`). Annotators label items from
AmbiK, SaGC and SafeAgentBench; answers are saved to a Google Sheet. The labels go into a research
paper: correctness and traceability matter more than features.

- Deployed from `main` on Streamlit Community Cloud, Python 3.14. Pushing `main` redeploys the live site.
- `main` holds interface v2: fixed evaluation subset (`eval_subset.json`), dataset selection with
  per-dataset targets (`study_config.json`), name-only sign-in, no preselected answers, training set
  (`training_tasks.json`, `version` = Training_Version), Interface_Version/Training_Version columns
  (L/M), allocations worksheet. Interface v1 is in git history before the v2 merge (`ef558e2`).
- Human-label documentation: `HUMAN_LABELS.md` in `~/multivote-triage-dev` (branch `dev`).

## Rules

- Work on a branch, never directly on `main`. Commit, never push; the user merges and pushes.
- Never touch the live Sheet. Test with the in-memory fake Sheet (`tests/fake_sheet.py`) and AppTest.
- Never change dataset labels, task text, item IDs, the evaluation subset or the answer options.
- Training: never change expected answers; bump `version` in `training_tasks.json` only when what
  annotators see in training changes. Training is done if a training-complete row exists for the
  (normalized) ID with the current Training_Version.
- Column F records exactly the context shown (`rde_core.context_text`).
- `app_user.py` has CRLF line endings; keep them.

## Checks

- Tests on both envs: `~/envs/rde314/bin/python -m unittest discover -s tests` and
  `~/envs/rde/bin/python -m unittest discover -s tests` (3.11). Optional:
  `MVT_DIR=~/multivote-triage-dev` runs the checks against the multivote-triage loaders and lists.
- Boot check: `streamlit run app_user.py --server.headless true` and `/_stcore/health` returns ok.

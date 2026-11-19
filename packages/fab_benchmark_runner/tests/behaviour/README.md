# Behaviour tests

These tests describe what the runner is meant to do, and why, through the entry points
people use: `create_experiment`, `raise_max_tokens`, and a job running a run config
(what `fab-bench run` calls). Only model calls and GCS are faked. Each module starts
with the decisions it describes; each test name states one behaviour.

| Module                      | Describes                                                      |
| --------------------------- | -------------------------------------------------------------- |
| `test_experiments.py`       | Inputs stored once: what models get, IDs, images               |
| `test_running_models.py`    | Results per model, reuse, resuming, errors, run configs        |
| `test_token_limits.py`      | `max_tokens`: recorded per row, raised in place, never lowered |
| `test_jobs_on_gcs.py`       | Many jobs on one experiment on GCS, without overwrites         |
| `test_recorded_versions.py` | What each run records, and when dry runs warn                  |

`conftest.py` holds the harness: a `Bench` with a model registry, a fake GCS, and
`FakeModels`, which answer every call unless a test makes one fail, crash or depend on
the token limit.

The detailed mechanics, such as validation messages, checkpoint order and upload
retries, are pinned in `tests/unit/`.

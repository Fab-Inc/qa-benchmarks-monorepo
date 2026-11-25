# Behaviour tests

These tests describe what the runner is meant to do, and why, through the entry points
people use: `create_experiment`, `check_inputs_file`, and a job running a run config
(what `fab-bench run` calls). Only model calls and GCS are faked. Each module starts
with the decisions it describes; each test name states one behaviour.

| Module                      | Describes                                                       |
| --------------------------- | --------------------------------------------------------------- |
| `test_inputs_file.py`       | The inputs file format, rule by rule, as `INPUTS.md` gives it   |
| `test_experiments.py`       | Inputs stored once: what models get, IDs, images                |
| `test_images.py`            | Images: what models are sent, checked before any call           |
| `test_running_models.py`    | Results per model, reuse, resuming, errors, run configs, memory |
| `test_token_limits.py`      | `max_tokens`: fixed per experiment, recorded per row            |
| `test_jobs_on_gcs.py`       | Many jobs on one experiment on GCS, without overwrites          |
| `test_recorded_versions.py` | What each run records, and when dry runs warn                   |

`conftest.py` holds the harness: a `Bench` with a model registry, a fake GCS, and
`FakeModels`, which answer every call unless a test makes one fail, crash or depend on
the token limit. Tests about what FDLLM is handed, such as images and a missing token
limit, run through FDLLM with `FakeProviders` standing in for the providers behind it.

The detailed mechanics, such as validation messages, checkpoint order and upload
retries, are pinned in `tests/unit/`.

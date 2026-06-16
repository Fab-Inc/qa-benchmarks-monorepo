# Fab Benchmarks Monorepo

This is a proposed monorepo layout for new benchmarks, specifically the QA facility
text-based benchmarks.

It contains:

- `packages/fab_benchmark_publisher`: shared result contracts, result JSON helpers, and
  a small GCS upload helper.
- `benchmarks/phonics_benchmark`: benchmark-specific package for the phonics benchmark.
- `benchmarks/language_knowledge_benchmark`: benchmark-specific package for the language
  knowledge benchmark.

The benchmark packages are currently cookiecutter-style placeholders with only a CLI
module and CLI tests. The shared package is intentionally small and follows the existing
`fab-benchmarks-configs` pattern: GitHub Actions authenticates to Google Cloud, sets
`BUCKET_NAME` and `FILE_PREFIX`, and Python uploads JSON to GCS with
`google.cloud.storage.Client`.

## Layout

```text
fab-benchmarks-monorepo/
  pyproject.toml
  uv.lock

  packages/
    fab_benchmark_publisher/
      pyproject.toml
      src/fab_benchmark_publisher/
        contracts.py
        publisher.py
        results.py

  benchmarks/
    __init__.py
    phonics_benchmark/
      __init__.py
      pyproject.toml
      src/phonics_benchmark/
        __init__.py
        cmd.py
      scripts/
        results_for_website.py
      tests/
        __init__.py
        test_cmd.py

    language_knowledge_benchmark/
      __init__.py
      pyproject.toml
      src/language_knowledge_benchmark/
        __init__.py
        cmd.py
      scripts/
        results_for_website.py
      tests/
        __init__.py
        test_cmd.py
```

## Development

Install all workspace dependencies:

```bash
uv sync --all-packages
```

For developers contributing to this project, please run

```sh
uv run pre-commit install
```

which will set up the pre-commit hooks that enforce code quality by running automated
checks and fixes — formatting, linting, end-of-file fixes, etc. — defined in
`.pre-commit-config.yaml`.

Note: Despite the name, this `install` step doesn't install pre-commit itself (uv did
that during `uv sync`) — it just adds a git hook to your local `.git/`. So you need to
run it in every new project, even if you've used pre-commit elsewhere before. Editing
the config later doesn't require re-running — changes are picked up automatically.

Run all tests:

```bash
uv run pytest
```

The publisher package expects the upload environment to provide the same basic variables
used by `fab-benchmarks-configs`:

```bash
export BUCKET_NAME="fab-benchmarks-485014-benchmark-results"
export FILE_PREFIX="dev"
```

Then a benchmark can publish records with:

```python
from fab_benchmark_publisher import LeaderboardRecord, publish_records

publish_records(
    benchmark_domain="PHONICS",
    artifact_name="leaderboard_data.json",
    records=[
        LeaderboardRecord(
            benchmark_domain="PHONICS",
            model="example-model",
            accuracy=88.0,
        )
    ],
)
```

This writes to `gs://$BUCKET_NAME/$FILE_PREFIX/PHONICS/leaderboard_data.json`.

## Dependency Model

New benchmark packages use the shared package as a workspace dependency:

```toml
[tool.uv.sources]
fab-benchmark-publisher = { workspace = true }
```

Old standalone repositories can depend on the shared package via a Git subdirectory
dependency:

```toml
[project]
dependencies = [
  "fab-benchmark-publisher",
]

[tool.uv.sources]
fab-benchmark-publisher = {
  git = "https://github.com/Fab-Inc/fab-benchmarks-monorepo.git",
  subdirectory = "packages/fab_benchmark_publisher",
  tag = "publisher-v0.1.0"
}
```

To update the dependency to a newer version, update the `tag` and then run

```bash
uv sync --upgrade-package fab-benchmark-publisher
```

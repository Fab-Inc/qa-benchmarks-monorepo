# Fab Benchmarks Monorepo

This is a monorepo for new benchmarks, specifically the QA Facility text-based
benchmarks.

It contains:

- `packages/fab_benchmark_publisher`: shared result contracts, result JSON helpers, and
  a small GCS upload helper.
- `packages/decodability`: standalone scaffolding for language-specific decodability
  tooling.
- `benchmarks/phonics_benchmark`: benchmark-specific package for the phonics benchmark.
- `benchmarks/linguistic_knowledge_benchmark`: benchmark-specific package for the
  linguistic knowledge benchmark.

The benchmark packages are currently cookiecutter-style placeholders with only a CLI
module and CLI tests. The shared package is intentionally small and follows the existing
`fab-benchmarks-configs` pattern: GitHub Actions authenticates to Google Cloud Storage
(GCS), sets `BUCKET_NAME` and `FILE_PREFIX`, and Python uploads JSON to GCS with
`google.cloud.storage.Client`.

## Development

This project uses [uv](https://docs.astral.sh/uv/) as its Python package and project
manager. If you don't already have uv installed, follow the
[uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/).

This project enforces a minimum required uv version via the `[tool.uv] required-version`
line of the root `pyproject.toml`. So please update your uv if required, and especially
if you are experiencing any errors.

To install all workspace dependencies (and whenever you change branch or edit
`pyproject.toml`):

```bash
uv sync --all-packages
```

which will install this project and its dependencies, including an appropriate Python
version. (All of this is installed into the hidden `.venv` virtual environment folder at
the project root.)

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

To run all tests:

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

### Common development commands:

```sh
# Install dependencies (e.g. after editing pyproject.toml or switching branch)
uv sync --all-packages

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov

# Run pre-commit checks (staged files only)
uv run pre-commit

# Run pre-commit checks (all files)
uv run pre-commit run -a

# Format code
uv run ruff format .

# Check and fix linting issues
uv run ruff check --fix .
```

## Dependency Model

New benchmark packages use the shared packages as a workspace dependency (defined in
root-level `pyproject.yaml`):

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

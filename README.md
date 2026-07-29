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

## Installation

This project uses [uv](https://docs.astral.sh/uv/) as its Python package and project
manager. If you don't already have uv installed, follow the
[uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/).

This project enforces a minimum required uv version via the `[tool.uv] required-version`
line of the root `pyproject.toml`. So please update your uv if required, and especially
if you are experiencing any errors.

Once uv is installed, clone this project and change into its directory, then run:

```bash
uv sync --all-packages
```

This installs the project and all dependencies (including an appropriate Python version)
into a hidden `.venv` virtual environment folder at the project root. The
`--all-packages` flag ensures nested workspace dependencies are included too.

To run the benchmarks against real models, you will need to copy `.env.example` to
`.env` (in the monorepo root) by running

```sh
cp .env.example .env
```

and then fill in the API keys for any model providers you intend to use. Ask the project
leads for the appropriate keys if you need them.

## Development

### Pre-commit hooks

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

### Tests

To run all tests:

```bash
uv run pytest
```

You are encouraged to run tests before committing and especially before pushing. CI/CD
runs the same suite, but catching failures locally is faster and cleaner.

### IDE setup & type checking

For full IDE support (autocomplete, type checking, code navigation, etc.), point your
editor at the project's Python interpreter, located at `.venv/bin/python` (on
macOS/Linux) or `.venv\Scripts\python.exe` (on Windows). To do this in VS Code, open the
command palette and run **Python: Select Interpreter**, then choose the entry inside
`.venv`.

This project aims to pass type checks at Pylance's **standard** level, so please set
`"python.analysis.typeCheckingMode": "standard"` in your VS Code settings (or **strict**
if you prefer). If you use a different type checker, configure a comparable level of
strictness. This is not currently CI-enforced but may be in future.

### Publishing benchmark results

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

### GCP local setup

To publish results to GCS from your local machine, you need the
[gcloud CLI](https://cloud.google.com/sdk/docs/install). Once installed, run:

```sh
gcloud auth application-default login
```

This stores user credentials that `google-auth` will use to impersonate the project's
service account. Then set `GOOGLE_CLOUD_IMPERSONATE_SERVICE_ACCOUNT` in the repo-root
`.env` — ask the project leads for the service account email. No code changes are
needed; `google-auth` reads this env var automatically when `Client()` is called. In
production, GitHub Actions handles authentication directly and this env var is not used.

Each benchmark has its own GCS bucket. Bucket config lives in a benchmark-specific
`.env` file rather than the repo root. For the phonics benchmark:

```sh
cp benchmarks/phonics_benchmark/.env.example benchmarks/phonics_benchmark/.env
```

Then fill in `BUCKET_NAME` (ask the project leads for the value).

To verify your full GCS setup, run the integration tests:

```sh
uv run pytest -m integration
```

The tests load both `.env` files automatically. A passing run confirms your credentials
can reach the bucket.

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

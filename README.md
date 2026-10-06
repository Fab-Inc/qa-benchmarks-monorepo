# Fab Benchmarks Monorepo

This is a monorepo for new benchmarks, specifically the QA Facility text-based
benchmarks.

It contains:

- `packages/fab_benchmark_publisher`: shared result contracts, result JSON helpers, the
  leaderboard bundle builder, and a small GCS upload helper.
- `packages/decodability`: standalone scaffolding for language-specific decodability
  tooling.
- `benchmarks/phonics_benchmark`: benchmark-specific package for the phonics benchmark.
- `benchmarks/decodable_text_generation`: the decodable text generation benchmarks
  (child and curriculum), reusing the phonics run harness.
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

### Publishing to the leaderboard website

The website reads five JSON files. We build four of them: `configmeta`, `data`,
`data-costfrontier` and `data-sizefrontier`. Each of these covers every benchmark ×
language pair. The fifth file, `providermeta`, comes from somewhere else. See below.

Publishing has two stages:

1. Each benchmark turns its own results into a small **records file**.
1. One shared command reads all the records files and builds the four JSON files.

```
LK      summary CSVs   ─┐
                         ├─▶  leaderboard_records_<pair>.json  ─▶  the 4 JSON files
phonics summary CSVs   ─┘              (stage 1)                        (stage 2)
```

Run all commands from the repo root, after `uv sync --all-packages`.

**About `models.csv` and `providers.csv`.** These two files list every model's provider,
price, size and colour. They are not committed to the repo. They are downloaded from the
`fab-open-configs` bucket, so a fresh checkout does not have them. Stage 2 downloads
them for you if they are missing, so usually there is nothing to do. LK's stage 1 script
is the exception: it expects the files to be there already. Run any LK command once to
download them, for example `uv run linguistic-knowledge-benchmark run-bench --help`.
Phonics' stage 1 does not need either file.

**About `providermeta.json`.** This file is the leaderboard's catalogue of models and
providers. It holds each provider's colour and logo, and each model's display name,
price, size and release date. Every benchmark on the site shares it, and we do not build
or upload it. It lives in the public `ai-for-ed-public` container on Azure:

```
https://fabdatastorage.blob.core.windows.net/ai-for-ed-public/providermeta.json
```

We need it because our `data` rows only contain ids, like
`{"modelId": "303", "providerId": "0", "accuracy": 71.2}`. The website looks up each id
in `providermeta` to find the model's name and price. The ids are row numbers in
`models.csv` and `providers.csv`. They match because `providermeta` is built from those
same two files.

Stage 2 downloads `providermeta` and checks every model in the bundle against it. If a
model is not in the file, stage 2 drops it and prints a warning. Without this check, the
website would show a row with no name and no price. If a model you scored gets dropped,
ask for `providermeta` to be updated — it is usually behind `models.csv`.

You do not need credentials, because the container is public. If that changes, set
`AZURE_SAS_TOKEN` in the repo-root `.env`.

#### Stage 1 — one records file per benchmark × language pair

A records file holds one score per model for one pair, and says which run the scores
came from.

**Coming soon:** records files will be downloaded from GCS, so you will be able to build
a bundle without running anyone else's benchmark. For now each benchmark builds its own
file locally. Each one has a `results_for_website.py` script that reads its own summary
CSV:

```bash
# LK Kiswahili + LK English, from the summary CSVs pinned in the script's SOURCES table
uv run --package linguistic-knowledge-benchmark python benchmarks/linguistic_knowledge_benchmark/scripts/results_for_website.py

# DA English + DA Kiswahili, every pair the phonics run reported
uv run --package phonics-benchmark python benchmarks/phonics_benchmark/scripts/results_for_website.py --run-id run-test
```

Each script writes `leaderboard_records_<pair>.json` into its own benchmark's
`results/web/` folder. It overwrites the previous file, so the folder always holds the
latest scores. That is why stage 2 can simply read the folder.

**Only write your own benchmark's records file.** This is what lets two people publish
separately. Your file is not changed by their run, so a rebuild always picks up the
newest version of both.

#### Stage 2 — build the four JSON files

```bash
uv run fab-leaderboard publish --records-dir benchmarks/linguistic_knowledge_benchmark/results/web --records-dir benchmarks/phonics_benchmark/results/web --outdir results/web
```

This command is the only place the bundle is built. It knows nothing about any
individual benchmark: it reads the records files and `leaderboard_pairs.json`, and
nothing else. It writes to `results/web/<timestamp>/`:

| File                                           | Pushed to Azure?                                              |
| ---------------------------------------------- | ------------------------------------------------------------- |
| `<timestamp>_<cluster>_configmeta.json`        | yes                                                           |
| `<timestamp>_<cluster>_data.json`              | yes                                                           |
| `<timestamp>_<cluster>_data-costfrontier.json` | yes                                                           |
| `<timestamp>_<cluster>_data-sizefrontier.json` | yes                                                           |
| `benchmark_default_models.json`                | no — a record of which models got pre-selected                |
| `manifest.json`                                | no — which records file fed each column, and what was dropped |

`leaderboard_pairs.json` in the repo root lists the expected columns and the order they
appear in on the website. Listing them explicitly, rather than reading whatever files
happen to be in a folder, gives two benefits. The column order never changes by
accident. And if a pair's records file is missing, the command reports it as `MISSING`
instead of silently leaving that column off the site.

**A model can have a score and still not be published.** This happens for two reasons:
the model is not in `models.csv`, or it is not in Azure's `providermeta`. Both cases
print a warning and are written to `manifest.json` under `dropped`. The manifest is the
only record of this, because a dropped model leaves no trace on the website.

#### Stage 3 — push to Azure

Not built yet. For now the four JSON files are written locally and uploaded by hand to
the `ai-for-ed-public` container. The API reads them from there and publishes them to
the website.

#### Adding a new benchmark × language pair

1. Add a row to `leaderboard_pairs.json` with `benchmark_domain` (for example `DA-sw`)
   and the `display_name` the site shows. The pair reports as `MISSING` until a records
   file exists, which is expected.
1. Make that benchmark write `leaderboard_records_DA-sw.json`. Copy the closest
   `results_for_website.py`. Each one is about 100 lines, and most of that is reading
   one CSV.

#### One thing to keep in mind when comparing columns

The columns do not all measure the same thing. Linguistic knowledge reports accuracy,
which is the plain proportion of correct answers. Phonics reports an adjusted balanced
accuracy, which averages recall and specificity after adjusting for vocabulary coverage.
Both appear on the same axis, labelled accuracy.

`bad_format` is also different. It is a real measurement for LK, but phonics has no
equivalent, so it is published as `0.0`. That looks like a perfect score when it
actually means "not measured". Mention this whenever you present the leaderboard.

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

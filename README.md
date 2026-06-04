# Fab Benchmarks Monorepo

This is a proposed monorepo layout for new benchmarks, specifically the QA Facility
text-based benchmarks.

It contains:

- `packages/fab_benchmark_publisher`: shared result contracts, result JSON helpers, and a small GCS upload helper.
- `packages/decodability`: standalone scaffolding for language-specific decodability tooling.
- `benchmarks/phonics_benchmark`: benchmark-specific package for the phonics benchmark.
- `benchmarks/linguistic_knowledge_benchmark`: benchmark-specific package for the linguistic knowledge benchmark.

The benchmark packages are currently cookiecutter-style placeholders with only a CLI module and CLI tests. The shared package is intentionally small and follows the existing `fab-benchmarks-configs` pattern: GitHub Actions authenticates to Google Cloud, sets `BUCKET_NAME` and `FILE_PREFIX`, and Python uploads JSON to GCS with `google.cloud.storage.Client`.

## Development

Install all workspace dependencies:

```bash
uv sync --all-packages
```

Run all tests:

```bash
uv run pytest
```

The publisher package expects the upload environment to provide the same basic variables used by `fab-benchmarks-configs`:

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

Old standalone repositories can depend on the shared package via a Git subdirectory dependency:

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

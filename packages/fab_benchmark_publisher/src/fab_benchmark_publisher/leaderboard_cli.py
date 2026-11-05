"""`fab-leaderboard` -- build the cluster-wide leaderboard bundle from records files.

Every benchmark writes its own leaderboard_records_<domain>.json; this command reads
all of them and produces the four JSON files the website loads. It is the ONLY place
the bundle is built, because the bundle spans every benchmark x language pair: two
benchmarks each publishing their own bundle would replace each other's columns rather
than merge.

The site reads a fifth file, providermeta.json, which resolves the ids in this bundle
into names, prices and provider colours. That one is shared across every benchmark
cluster and lives in an Azure container we do not own, so this command only downloads
it -- to drop any model it cannot resolve rather than publish a nameless row.

Nothing here knows about any benchmark's results layout. It reads records files and
the pairs config, and that is all.

One --records-dir per benchmark; the README's "Publishing to the leaderboard website"
section carries the full commands, ready to paste.

    # everything local, no upload
    uv run fab-leaderboard publish --records-dir DIR --outdir results/web

Run from the repo root: the config and CSV defaults are relative to the working
directory.

TODO: pulling records from and pushing the bundle to GCS. The pieces are in place --
publisher.download_file / publisher.upload_file -- but for now both sides are local.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from fab_benchmark_publisher.contracts import BenchmarkPair, RecordsFile
from fab_benchmark_publisher.leaderboard import (
    DEFAULT_MODELS_ADD_MISSING,
    DEFAULT_MODELS_FILTER_BY_FAMILY,
    DEFAULT_MODELS_MAX,
    build_leaderboard_bundle,
    find_records_files,
    read_records_files,
    write_leaderboard_bundle,
)
from fab_benchmark_publisher.publisher import (
    AZURE_PROVIDERMETA_URL,
    ensure_shared_config,
    fetch_providermeta,
)

DEFAULT_PAIRS_FILE = "leaderboard_pairs.json"
MANIFEST_NAME = "manifest.json"

# Model and provider metadata always comes from the shared-config bucket, kept at the
# repo root. There is deliberately no flag to point at a local copy instead: the
# leaderboard's model ids are row positions in these exact files, so a bundle built
# against someone's private edit would carry ids that mean something else on the site.
MODELS_CSV = "models.csv"
PROVIDERS_CSV = "providers.csv"


@dataclass(frozen=True)
class ClusterConfig:
    """The leaderboard's columns and their order, from leaderboard_pairs.json."""

    cluster_name: str
    pairs: list[BenchmarkPair]


def load_cluster_config(path: str | Path) -> ClusterConfig:
    """Read the pairs config, which owns the column order and the display labels."""
    path = Path(path)
    if not path.is_file():
        raise SystemExit(
            f"ERROR: pairs config not found: {path}\n"
            "       run from the repo root, or pass --pairs"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        pairs = [
            BenchmarkPair(
                benchmark_domain=p["benchmark_domain"], display_name=p["display_name"]
            )
            for p in payload["pairs"]
        ]
    except (KeyError, TypeError) as e:
        raise SystemExit(f"ERROR: {path} is malformed: {e}") from e
    if not pairs:
        raise SystemExit(f"ERROR: {path} lists no pairs")
    domains = [p.benchmark_domain for p in pairs]
    duplicates = sorted({d for d in domains if domains.count(d) > 1})
    if duplicates:
        raise SystemExit(f"ERROR: {path} repeats pair(s): {', '.join(duplicates)}")
    return ClusterConfig(
        cluster_name=payload.get("cluster_name", "cluster"), pairs=pairs
    )


def order_contributions(
    config: ClusterConfig, contributions: Sequence[RecordsFile]
) -> tuple[list[RecordsFile], list[BenchmarkPair]]:
    """Put the contributions in the config's column order; report unexpected ones.

    Returns (ordered contributions, pairs listed in the config with no records). The
    display name comes from the config even when a records file disagrees: one place
    decides what a column is called, or two benchmarks can rename each other's.
    """
    by_domain = {c.benchmark.benchmark_domain: c for c in contributions}
    unexpected = sorted(set(by_domain) - {p.benchmark_domain for p in config.pairs})
    if unexpected:
        raise SystemExit(
            f"ERROR: records file(s) for pair(s) not in the config: "
            f"{', '.join(unexpected)}\n"
            "       add them to the pairs config, or remove the files"
        )
    ordered: list[RecordsFile] = []
    missing: list[BenchmarkPair] = []
    for pair in config.pairs:
        contribution = by_domain.get(pair.benchmark_domain)
        if contribution is None:
            missing.append(pair)
            continue
        ordered.append(
            RecordsFile(
                benchmark=pair,
                records=contribution.records,
                source=contribution.source,
                generated_at=contribution.generated_at,
            )
        )
    return ordered, missing


def write_manifest(
    run_dir: Path,
    contributions: Sequence[RecordsFile],
    missing: Sequence[BenchmarkPair],
    dropped: dict[str, str] | None = None,
) -> Path:
    """Record which contribution went into this bundle, and what did not make it.

    Without it, "which run is on the site right now?" can only be answered by
    re-deriving the bundle, and a bundle built from a records file that has since been
    replaced is indistinguishable from a current one. `dropped` is the other half of
    that: a model that scored but could not be published leaves no trace on the site,
    so the only record it ever existed is here.
    """
    path = run_dir / MANIFEST_NAME
    path.write_text(
        json.dumps(
            {
                "built_at": datetime.now(UTC).isoformat(),
                "dropped": dropped or {},
                "contributions": [
                    {
                        "benchmark_domain": c.benchmark.benchmark_domain,
                        "display_name": c.benchmark.display_name,
                        "source": c.source,
                        "generated_at": c.generated_at,
                        "n_records": len(c.records),
                    }
                    for c in contributions
                ],
                "missing": [p.benchmark_domain for p in missing],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _report(contributions: Sequence[RecordsFile], missing: Sequence[BenchmarkPair]):
    print("Records found:")
    for c in contributions:
        print(
            f"  {c.benchmark.benchmark_domain:<8} {c.generated_at or '?':<34} "
            f"{len(c.records):>4} models   {c.source or '-'}"
        )
    for pair in missing:
        print(f"  MISSING: {pair.benchmark_domain} ({pair.display_name}) -- no records")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="fab-leaderboard",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)
    publish = sub.add_parser("publish", help="build the leaderboard bundle")
    publish.add_argument(
        "--records-dir",
        action="append",
        default=None,
        required=True,
        metavar="DIR",
        help="directory holding leaderboard_records_*.json; repeat for each benchmark",
    )
    publish.add_argument(
        "--pairs",
        default=DEFAULT_PAIRS_FILE,
        help=f"pairs config defining the columns and their order "
        f"(default {DEFAULT_PAIRS_FILE})",
    )
    publish.add_argument(
        "--outdir",
        default="results/web",
        help="parent directory for the run folder (default results/web)",
    )
    publish.add_argument(
        "--cluster",
        default=None,
        help="override the cluster_name from the pairs config",
    )
    publish.add_argument(
        "--all-models",
        action="store_true",
        help="skip curation: pre-select every model with a result",
    )
    publish.add_argument("--max-models", type=int, default=DEFAULT_MODELS_MAX)
    publish.add_argument(
        "--include-all-frontier-models",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_MODELS_ADD_MISSING,
    )
    publish.add_argument(
        "--family-filter",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_MODELS_FILTER_BY_FAMILY,
    )
    args = parser.parse_args(argv)

    config = load_cluster_config(args.pairs)
    # Records first: with nothing to publish there is nothing worth downloading, and
    # this is the one failure a fresh checkout hits most often.
    paths = find_records_files(args.records_dir)
    if not paths:
        raise SystemExit(
            "ERROR: no leaderboard_records_*.json under "
            f"{', '.join(str(d) for d in args.records_dir)}\n"
            "       run each benchmark's results_for_website.py first"
        )
    try:
        contributions, missing = order_contributions(config, read_records_files(paths))
    except ValueError as e:
        raise SystemExit(f"ERROR: {e}") from e
    _report(contributions, missing)

    # Fetched here rather than left to the first open() inside the build: both are
    # synced from the shared-config bucket rather than committed, so on a fresh
    # checkout they are simply absent -- and a bundle built without them would render
    # every model with no provider, price or size instead of failing visibly.
    try:
        models_csv = ensure_shared_config(MODELS_CSV)
        providers_csv = ensure_shared_config(PROVIDERS_CSV)
    except Exception as e:
        raise SystemExit(f"ERROR: {e}") from e
    # The site's own providermeta, which every id in this bundle is resolved through.
    # Hard-failing on a fetch error rather than building without it: a bundle that
    # skipped the check is indistinguishable from one that passed, and the only reason
    # to build one is to upload it to that same Azure container anyway.
    try:
        provider_meta = fetch_providermeta()
    except Exception as e:
        raise SystemExit(
            f"ERROR: cannot read the leaderboard's providermeta.json -- {e}\n"
            f"       {AZURE_PROVIDERMETA_URL}"
        ) from e

    try:
        bundle = build_leaderboard_bundle(
            [r for c in contributions for r in c.records],
            benchmarks=[c.benchmark for c in contributions],
            models_csv=models_csv,
            providers_csv=providers_csv,
            provider_meta=provider_meta,
            curate_defaults=not args.all_models,
            max_models=args.max_models,
            add_missing_models=args.include_all_frontier_models,
            filter_by_family=args.family_filter,
        )
    except (ValueError, FileNotFoundError) as e:
        raise SystemExit(f"ERROR: {e}") from e

    run_dir = write_leaderboard_bundle(
        bundle,
        outdir=args.outdir,
        cluster_name=args.cluster or config.cluster_name,
    )
    manifest = write_manifest(run_dir, contributions, missing, bundle.dropped)
    print(f"  {manifest.name}  (what went into this bundle)")

    data_records = bundle.payloads["data"]
    print(
        f"\nDone. {len({r['providerId'] for r in data_records})} providers | "
        f"{len({r['modelId'] for r in data_records})} models | "
        f"{len(contributions)} pairs | {len(data_records)} data records"
    )
    if bundle.dropped:
        print(f"  {len(bundle.dropped)} model(s) dropped -- see {manifest.name}")
    for domain, meta in bundle.payloads["configmeta"]["benchmarks"].items():
        scored = sum(1 for r in data_records if r["benchmark_domain"] == domain)
        print(
            f"  {domain}: {len(meta['default_models'])}/{scored} models pre-selected"
            f", {len(meta['disabled_models'])} disabled"
        )


if __name__ == "__main__":  # pragma: no cover - module entry point
    main()

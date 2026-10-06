#!/usr/bin/env python3
"""
Publish this benchmark's scores to the foundational-literacy leaderboard.

Two things happen per run, and they are separate on purpose:

  1. RECORDS. Each (benchmark, language) pair's summary CSV becomes a records file,
     <outdir>/leaderboard_records_<domain>.json, holding LeaderboardRecords plus the
     pair it fills. This is this benchmark's contribution and the ONLY part that
     knows about judge-bench's summary CSV format.
  2. BUNDLE. Those records are assembled into the four leaderboard JSON files by
     fab_benchmark_publisher.leaderboard, shared with every other benchmark on this
     leaderboard. The site's fifth file, providermeta.json, is not built by anyone
     here -- it comes from Azure; `fab-leaderboard publish` reads it to check every
     published id resolves, and this script skips that check entirely.

The bundle is CLUSTER-WIDE: configmeta and data each span every pair, so one publish
carries both. SOURCES therefore lists only the pairs THIS benchmark owns (the LK
ones); the other columns come from their own benchmarks' records files.

WHICH MEANS THE BUNDLE THIS SCRIPT WRITES IS PARTIAL -- it covers the LK pairs and
nothing else, and is only useful for checking this benchmark's own contribution. The
publishable bundle is the one `fab-leaderboard publish` builds from every benchmark's
records file at once:

    uv run fab-leaderboard publish --records-dir benchmarks/linguistic_knowledge_benchmark/results/web --records-dir benchmarks/phonics_benchmark/results/web

  * default_models = the models the site pre-selects for that pair, curated by
    fab_benchmark_publisher.frontier in four stages:
      1. concave frontier-layer peeling (cost, and size for open models), capped by
         --max-models per frontier;
      2. --include-all-frontier-models re-adds any raw-Pareto vertex the concavity
         smoothed away. Leaving it ON is what guarantees every corner of a DRAWN
         frontier line is pre-selected; with it off, the build warns for each corner
         that is not, which renders as a line bending at an invisible model;
      3. --family-filter dedups per provider/family, keeping each family's most
         recent model unless an older one sits on a frontier;
      4. MANUAL_DEFAULT_MODELS are force-included last, so nothing above strips them.
    Defaults live in the shared DEFAULT_MODELS_* constants; --all-models bypasses all
    four. Curation only changes what is PRE-SELECTED - every model still ships in
    data, and stays togglable on the site.
  * disabled_models = models with a result somewhere in the cluster but none for
    this pair, so the site can grey them out rather than imply they scored 0.
  * cost/size frontier LINES are still computed (raw Pareto, matching the renderer's
    drawn-frontier behaviour) from models.csv price/size.

This script only WRITES local JSON files under --outdir - it does not upload anywhere.

Inputs:
  models.csv       model metadata (price, size, provider, open, release_date, ...)
  providers.csv    provider colors + SVG logos
  BENCHMARK_TYPES / LANGUAGES / SOURCES below - this benchmark's own pairs. Each
                            SOURCES row hardcodes its own judge-bench summary CSV, so
                            pairs do not have to share an experiment-id. Edit those
                            tables to publish a different experiment or a new pair -
                            no CLI override.

Outputs, under --outdir (default results/web/): one records file per published pair,
overwritten each run, plus one folder per run named for the run's timestamp so runs
never interleave and one can be uploaded, archived or deleted as a unit. Inside it the
four leaderboard files follow the website filename convention
<timestamp>_<cluster-name>_<file-name>.json, where <timestamp> is ISO 8601 UTC with
the time compacted (no colons - those are illegal in Windows filenames):

  results/web/
    leaderboard_records_LK-sw.json
    leaderboard_records_LK-en.json
    2026-08-27T142530Z/
      2026-08-27T142530Z_foundational-literacy_configmeta.json
      2026-08-27T142530Z_foundational-literacy_data.json
      2026-08-27T142530Z_foundational-literacy_data-costfrontier.json
      2026-08-27T142530Z_foundational-literacy_data-sizefrontier.json
      benchmark_default_models.json

benchmark_default_models.json is a record, NOT a fifth upload file: benchmark_domain ->
sorted model ids, mirroring the CDPK pipeline's file of the same name. Its name stays
outside the <timestamp>_<cluster>_*.json glob so a glob-based uploader ignores it. To
see what a curation change did, diff it against a previous run's folder.

Run from the repo root (no arguments needed - the CSVs to publish are hardcoded in
SOURCES). The --package flag is required: plain `python` has no pandas.

    uv run --package linguistic-knowledge-benchmark python benchmarks/linguistic_knowledge_benchmark/scripts/results_for_website.py

    # write somewhere else
    ... results_for_website.py --outdir /tmp/web

    # no curation at all - every model with a result is pre-selected
    ... results_for_website.py --all-models

    # looser/tighter curation; see the shared DEFAULT_MODELS_* constants
    ... results_for_website.py --no-family-filter
    ... results_for_website.py --no-family-filter --no-include-all-frontier-models
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from fab_benchmark_publisher import BenchmarkPair, LeaderboardRecord
from fab_benchmark_publisher.leaderboard import (
    DEFAULT_MODELS_ADD_MISSING,
    DEFAULT_MODELS_FILTER_BY_FAMILY,
    DEFAULT_MODELS_MAX,
    build_leaderboard_bundle,
    write_leaderboard_bundle,
    write_records_file,
)
from linguistic_knowledge_benchmark.shared_utils import MODELS_CSV, PROVIDERS_CSV

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS_DIR = ROOT / "results"

# Cluster this benchmark belongs to on the website; the middle segment of every output
# filename (<timestamp>_<cluster-name>_<file-name>.json).
CLUSTER_NAME = "foundational-literacy"

# Models always pre-selected, whatever the frontier/family logic decides. Each must be a
# model_id in models.csv AND have a result for at least one pair, or it is warned
# about and dropped. Keep this short (2-3 models).
MANUAL_DEFAULT_MODELS: list[str] = [
    # "claude-fable-5",
    # "inkling",
]

# Benchmark types THIS script publishes, used to validate SOURCES and to build each
# pair's benchmark_domain ("LK" + "sw" -> "LK-sw") and label ("LK Kiswahili"). The
# cluster holds other types -- DA, DTG -- but those are other benchmarks' to publish,
# and the full column list lives in the repo-root leaderboard_pairs.json rather than
# being restated here where the two could drift.
BENCHMARK_TYPES: dict[str, str] = {
    "LK": "Linguistic Knowledge",
}

# code is the ISO 639-1 language code and feeds the benchmark_domain id (LK-sw);
# display_name feeds the selector label (LK Kiswahili).
LANGUAGES: dict[str, dict[str, str]] = {
    "english": {"code": "en", "display_name": "English"},
    "kiswahili": {"code": "sw", "display_name": "Kiswahili"},
}

# One row per pair THIS benchmark owns. Pairs owned by another benchmark are not
# listed: each writes its own records file, and the full set of leaderboard columns
# lives in the repo-root leaderboard_pairs.json. Nothing here decides the column order
# on the website -- that comes from the pairs config when the bundle is built.
#
# summary_csv names the scores file to publish for that pair. Which experiment reaches
# the leaderboard is therefore a committed line of code rather than the newest file in
# results/, which holds several summaries at once (final-137-items,
# final-137-items_fallback, consistency-20x); there is no CLI override on purpose.
# Relative paths resolve under this benchmark's results/, absolute paths are used as
# given. summary_csv=None means "we own this pair but have no results yet" - the row is
# skipped with a note instead of failing the run.
#
# The file only has to carry these columns (anything else in it is ignored):
#   model                            model id, must match a model_id in models.csv
#                                    (rows that match nothing are dropped with a warning)
#   <accuracy_column>                the score, 0-100 - "accuracy" unless overridden below
#   scored_count, unscored_count     optional; feed the sample size n and the bad_format
#                                    share. Without them both are reported as 0.
#
# accuracy_column selects which score column of that CSV becomes the leaderboard value.
SOURCES: list[dict] = [
    {
        "benchmark": "LK",
        "language": "kiswahili",
        "accuracy_column": "accuracy",
        "summary_csv": "summary_linguistic-knowledge_kiswahili_final-135-items_fallback.csv",
    },
    {
        "benchmark": "LK",
        "language": "english",
        "accuracy_column": "accuracy",
        "summary_csv": "summary_linguistic-knowledge_english_final-137-items_fallback.csv",
    },
]


def resolve_sources() -> tuple[list[dict], list[str]]:
    """Expand SOURCES into (rows to publish, labels skipped for having no results yet).

    Each published row carries the BenchmarkPair it fills, the resolved CSV path and
    its accuracy column.
    """
    script = Path(__file__).name
    resolved: list[dict] = []
    skipped: list[str] = []
    for i, src in enumerate(SOURCES):
        bench, lang = src["benchmark"], src["language"]
        if bench not in BENCHMARK_TYPES:
            raise SystemExit(
                f"ERROR: SOURCES[{i}] benchmark '{bench}' is not in BENCHMARK_TYPES "
                f"({', '.join(BENCHMARK_TYPES)}) - add it in {script}"
            )
        if lang not in LANGUAGES:
            raise SystemExit(
                f"ERROR: SOURCES[{i}] language '{lang}' is not in LANGUAGES "
                f"({', '.join(LANGUAGES)}) - add it in {script}"
            )
        domain = f"{bench}-{LANGUAGES[lang]['code']}"
        label = f"{bench} {LANGUAGES[lang]['display_name']}"
        if not src["summary_csv"]:
            skipped.append(label)
            continue
        configured = Path(src["summary_csv"])
        path = configured if configured.is_absolute() else RESULTS_DIR / configured
        if not path.is_file():
            raise SystemExit(
                f"ERROR: summary CSV for {label} not found: {path}\n"
                f"       fix SOURCES[{i}]['summary_csv'] in {script}"
            )
        resolved.append(
            {
                "benchmark": BenchmarkPair(benchmark_domain=domain, display_name=label),
                "summary_csv": path,
                "accuracy_column": src["accuracy_column"],
            }
        )
    domains = [r["benchmark"].benchmark_domain for r in resolved]
    duplicates = sorted({d for d in domains if domains.count(d) > 1})
    if duplicates:
        raise SystemExit(
            f"ERROR: duplicate benchmark/language pairs in SOURCES: {', '.join(duplicates)}"
        )
    if not resolved:
        raise SystemExit("ERROR: no SOURCES row has a summary_csv - nothing to publish")
    return resolved, skipped


def records_for(
    summary_csv: Path, accuracy_col: str, benchmark_domain: str
) -> list[LeaderboardRecord]:
    """One judge-bench summary CSV -> LeaderboardRecords for one pair.

    This is the only place that knows judge-bench's summary format; everything
    downstream sees records. bad_format is derived from
    unscored_count / (scored_count + unscored_count) as a proxy for "answers that
    couldn't be judged" - this benchmark's summary CSV has no literal bad_format
    column, unlike the CDPK data the leaderboard's other inputs come from.
    """
    records: list[LeaderboardRecord] = []
    with open(summary_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            model = (row.get("model") or "").strip()
            acc = row.get(accuracy_col)
            if not model or acc in (None, ""):
                continue
            scored = int(float(row.get("scored_count") or 0))
            unscored = int(float(row.get("unscored_count") or 0))
            total = scored + unscored
            records.append(
                LeaderboardRecord(
                    benchmark_domain=benchmark_domain,
                    model=model,
                    accuracy=round(float(acc), 2),
                    n=total,
                    bad_format=round(100 * unscored / total, 4) if total else 0.0,
                )
            )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--cluster",
        default=CLUSTER_NAME,
        help="cluster-name segment of the output filenames "
        f"(<timestamp>_<cluster-name>_<file-name>.json); default {CLUSTER_NAME}",
    )
    parser.add_argument(
        "--outdir",
        default=str(RESULTS_DIR / "web"),
        help="parent directory for the records files and the run folders; each run "
        "writes <outdir>/<timestamp>/ holding the 5 JSON files plus the "
        "default-models record",
    )
    parser.add_argument("--models-csv", default=str(MODELS_CSV))
    parser.add_argument("--providers-csv", default=str(PROVIDERS_CSV))
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="skip curation: pre-select every model with a result, as before "
        "frontier selection existed. Useful for A/B-ing against a curated run.",
    )
    parser.add_argument(
        "--max-models",
        type=int,
        default=DEFAULT_MODELS_MAX,
        help="soft cap on models collected per frontier type per pair "
        f"(default {DEFAULT_MODELS_MAX})",
    )
    parser.add_argument(
        "--include-all-frontier-models",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_MODELS_ADD_MISSING,
        help="pre-select every model on the drawn cost/size frontier lines, "
        "including the ones the concave-hull peeling smoothed away "
        f"(default: {'on' if DEFAULT_MODELS_ADD_MISSING else 'off'}). Turn it off "
        "and a frontier line can bend at a model that is not shown by default.",
    )
    parser.add_argument(
        "--family-filter",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_MODELS_FILTER_BY_FAMILY,
        help="deduplicate models per provider/family via PROVIDER_FAMILY_FILTERS, "
        "keeping each family's most recent model unless an older one sits on a "
        f"frontier (default: {'on' if DEFAULT_MODELS_FILTER_BY_FAMILY else 'off'})",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sources, skipped = resolve_sources()
    if skipped:
        print(f"No results yet, not published: {', '.join(skipped)}")

    # --- 1. This benchmark's contribution: one records file per pair ---
    print("Reading summary CSVs:")
    records: list[LeaderboardRecord] = []
    for src in sources:
        pair = src["benchmark"]
        print(f"  {pair.display_name}: {src['summary_csv'].name}")
        pair_records = records_for(
            src["summary_csv"], src["accuracy_column"], pair.benchmark_domain
        )
        records.extend(pair_records)
        path = write_records_file(
            pair_records,
            benchmark=pair,
            outdir=outdir,
            source=src["summary_csv"].name,
        )
        print(f"      {path.name}  ({len(pair_records)} records)")

    # --- 2. The cluster bundle (this benchmark's pairs only, for now) ---
    try:
        bundle = build_leaderboard_bundle(
            records,
            benchmarks=[src["benchmark"] for src in sources],
            models_csv=Path(args.models_csv),
            providers_csv=Path(args.providers_csv),
            curate_defaults=not args.all_models,
            max_models=args.max_models,
            add_missing_models=args.include_all_frontier_models,
            filter_by_family=args.family_filter,
            manual_default_models=MANUAL_DEFAULT_MODELS,
        )
    except ValueError as e:
        raise SystemExit(f"ERROR: {e}") from e
    write_leaderboard_bundle(bundle, outdir=outdir, cluster_name=args.cluster)

    data_records = bundle.payloads["data"]
    unique_models = {row["modelId"] for row in data_records}
    n_providers = len({row["providerId"] for row in data_records})
    print(
        f"\nDone. {n_providers} providers | {len(unique_models)} models | "
        f"{len(sources)} benchmarks | {len(data_records)} data records"
    )
    # Pre-selected vs available, per pair: the number to watch when retuning.
    for domain, meta in bundle.payloads["configmeta"]["benchmarks"].items():
        scored = sum(1 for row in data_records if row["benchmark_domain"] == domain)
        print(
            f"  {domain}: {len(meta['default_models'])}/{scored} models pre-selected"
            f", {len(meta['disabled_models'])} disabled"
        )


if __name__ == "__main__":
    main()

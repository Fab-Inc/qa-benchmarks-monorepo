#!/usr/bin/env python3
"""
Generate the leaderboard JSON files for the foundational-literacy cluster's website.

One run writes one set of files covering every benchmark x language pair configured in
SOURCES - the cluster holds 3 benchmark types (LK / DA / DTG) x N languages, e.g.
"LK English", "DA English", "LK Kiswahili", "DA Kiswahili", "DTG Kiswahili".

  * default_models = every model that has a result row for that pair (no
    frontier-layering / provider-family dedup - the pool is already small enough to
    show in full).
  * cost/size frontier LINES are still computed (raw Pareto, matching the renderer's
    drawn-frontier behaviour) from data/models.csv price/size.

This script only WRITES local JSON files under --outdir - it does not upload anywhere.

Inputs:
  data/models.csv       model metadata (price, size, provider, open, release_date, ...)
  data/providers.csv    provider colors + SVG logos
  BENCHMARK_TYPES / LANGUAGES / SOURCES below - the benchmark x language matrix. Each
                            SOURCES row hardcodes its own judge-bench summary CSV, so
                            pairs do not have to share a experiment-id and may point at a
                            sibling benchmark's results/ directory. Edit those tables to
                            publish a different run or a new pair - no CLI override.

Outputs (--outdir, default results/web/) follow the website filename convention
<timestamp>_<cluster-name>_<file-name>.json, where <timestamp> is ISO 8601 UTC with the
time compacted (no colons - those are illegal in Windows filenames). The timestamp is
generated fresh on every run and shared by all five files of that run, e.g.:
  2026-08-27T142530Z_foundational-literacy_configmeta.json
  2026-08-27T142530Z_foundational-literacy_providermeta.json
  2026-08-27T142530Z_foundational-literacy_data.json
  2026-08-27T142530Z_foundational-literacy_data-costfrontier.json
  2026-08-27T142530Z_foundational-literacy_data-sizefrontier.json

Run (no arguments needed - the CSVs to publish are hardcoded in SOURCES):
    python benchmarks/linguistic_knowledge_benchmark/scripts/results_for_website.py
    python .../results_for_website.py --outdir /tmp/out
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

# Cluster this benchmark belongs to on the website; the middle segment of every output
# filename (<timestamp>_<cluster-name>_<file-name>.json).
CLUSTER_NAME = "foundational-literacy"

# The foundational-literacy cluster is a matrix of benchmark types x languages: the
# website shows one selector per (benchmark, language) pair, e.g. "LK English",
# "DA English", "LK Kiswahili", "DA Kiswahili", "DTG Kiswahili". Add a benchmark type or a
# language once below, then add one SOURCES row per pair you have results for.
BENCHMARK_TYPES: dict[str, str] = {
    "LK": "Linguistic Knowledge",
    "DA": "Decodability Assessment",
    "DTG": "Decodable Text Generation",
}

# code is the ISO 639-1 language code and feeds the benchmark_domain id (LK-sw);
# display_name feeds the selector label (LK Kiswahili).
LANGUAGES: dict[str, dict[str, str]] = {
    "english": {"code": "en", "display_name": "English"},
    "kiswahili": {"code": "sw", "display_name": "Kiswahili"},
}

# One row per (benchmark, language) pair to publish.
#
# summary_csv names the scores file to publish for that pair - every pair points at its
# own file, so each benchmark keeps its results wherever it produces them. Relative paths
# resolve under this benchmark's results/; absolute paths and ../ paths into a sibling
# benchmark both work. summary_csv=None means "pair exists but has no results yet" - the
# row is skipped with a note instead of failing the run.
#
# The file only has to carry these columns (anything else in it is ignored):
#   model                            model id, must match a model_id in data/models.csv
#                                    (rows that match nothing are dropped with a warning)
#   <accuracy_column>                the score, 0-100 - "accuracy" unless overridden below
#   scored_count, unscored_count     optional; feed the sample size n and the bad_format
#                                    share. Without them both are reported as 0.
#
# accuracy_column selects which score column of that CSV becomes the leaderboard value.
#
# The order here defines the order of the columns on the website leaderboard. Rows with no
# summary_csv still hold their slot, so filling one in later needs no reordering.
SOURCES: list[dict] = [
    {
        "benchmark": "LK",
        "language": "english",
        "accuracy_column": "accuracy",
        "summary_csv": "summary_linguistic-knowledge_english_final-137-items_fallback.csv",
    },
    {
        "benchmark": "DA",
        "language": "english",
        "accuracy_column": "accuracy",
        "summary_csv": None,
    },
    {
        "benchmark": "LK",
        "language": "kiswahili",
        "accuracy_column": "accuracy",
        "summary_csv": "summary_linguistic-knowledge_kiswahili_final-135-items_fallback.csv",
    },
    {
        "benchmark": "DA",
        "language": "kiswahili",
        "accuracy_column": "accuracy",
        "summary_csv": None,
    },
    {
        "benchmark": "DTG",
        "language": "kiswahili",
        "accuracy_column": "accuracy",
        "summary_csv": None,
    },
]


def resolve_sources() -> tuple[list[dict], list[str]]:
    """Expand SOURCES into (rows to publish, labels skipped for having no results yet).

    Each published row carries the derived benchmark_domain id ("LK-sw"), the selector
    label ("LK Kiswahili"), the resolved CSV path and its accuracy column.
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
                "benchmark_domain": domain,
                "display_name": label,
                "summary_csv": path,
                "accuracy_column": src["accuracy_column"],
            }
        )
    domains = [r["benchmark_domain"] for r in resolved]
    duplicates = sorted({d for d in domains if domains.count(d) > 1})
    if duplicates:
        raise SystemExit(
            f"ERROR: duplicate benchmark/language pairs in SOURCES: {', '.join(duplicates)}"
        )
    if not resolved:
        raise SystemExit("ERROR: no SOURCES row has a summary_csv - nothing to publish")
    return resolved, skipped


def _float_or_none(value: str | None) -> float | None:
    value = (value or "").strip()
    return float(value) if value else None


def load_providers(providers_csv: Path) -> dict[str, dict]:
    """provider name -> {color, svg}, in file order (so positional IDs are stable)."""
    providers_db: dict[str, dict] = {}
    with open(providers_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row["provider"].strip()
            providers_db[name] = {
                "color": row["color"].strip(),
                "svg": row["svg"].strip(),
            }
    return providers_db


def load_models(models_csv: Path) -> tuple[dict[str, dict], dict[str, int]]:
    """(model_id -> metadata, model_id -> 0-based row index in models.csv)."""
    models_db: dict[str, dict] = {}
    models_row_index: dict[str, int] = {}
    with open(models_csv, newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            mid = row["model_id"].strip()
            models_row_index[mid] = i
            inp = _float_or_none(row.get("input_cost"))
            out = _float_or_none(row.get("output_cost"))
            models_db[mid] = {
                "provider": row["provider"].strip(),
                "open": row["open"].strip().lower() == "true",
                "url": row.get("url", "").strip() or None,
                "release_date": row.get("release_date", "").strip() or None,
                "size": _float_or_none(row.get("size")),
                "input_cost": inp,
                "output_cost": out,
                # blended = 3:1 input:output weighted average, matching the CDPK leaderboard.
                "blended_price": round((inp * 3 + out) / 4, 6)
                if inp is not None and out is not None
                else None,
                "display_name": row.get("display_name", "").strip() or mid,
            }
    return models_db, models_row_index


def load_source_results(summary_csv: Path, accuracy_col: str) -> dict[str, dict]:
    """model_id -> {accuracy, n, bad_format} for one (benchmark, language) summary CSV.

    bad_format is derived from unscored_count / (scored_count + unscored_count) as a
    proxy for "answers that couldn't be judged" - this benchmark's summary CSV has no
    literal bad_format column, unlike the CDPK data format_data_web.py reads from.
    """
    out: dict[str, dict] = {}
    with open(summary_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            model = (row.get("model") or "").strip()
            acc = row.get(accuracy_col)
            if not model or acc in (None, ""):
                continue
            scored = int(float(row.get("scored_count") or 0))
            unscored = int(float(row.get("unscored_count") or 0))
            total = scored + unscored
            out[model] = {
                "accuracy": round(float(acc), 2),
                "n": total,
                "bad_format": round(100 * unscored / total, 4) if total else 0.0,
            }
    return out


def compute_frontier(df: pd.DataFrame, acc_col: str, sorting_col: str) -> pd.DataFrame:
    """Raw Pareto frontier (running-max accuracy vs sorting_col) - the rendered step
    function never sits below a real model. Ties on sorting_col keep the highest
    accuracy row."""
    d = df.sort_values(sorting_col, ascending=True)
    frontier, best = [], -1e9
    for _, row in d.iterrows():
        if row[acc_col] > best:
            frontier.append(row)
            best = row[acc_col]
    if not frontier:
        return pd.DataFrame(columns=df.columns)
    out = pd.DataFrame(frontier).sort_values(sorting_col)
    return out.drop_duplicates(sorting_col, keep="last").reset_index(drop=True)


def step_segments(frontier: list[dict]) -> list[dict]:
    """Convert frontier points to step-line segments for the web chart renderer.
    Before each new point, insert a horizontal segment at the previous y-value so the
    chart draws a staircase rather than diagonal lines."""
    segs: list[dict] = []
    for i, p in enumerate(frontier):
        if i > 0:
            segs.append(
                {
                    "x": p["x"],
                    "y": round(frontier[i - 1]["y"], 3),
                    "providerId": p["providerId"],
                }
            )
        segs.append({"x": p["x"], "y": round(p["y"], 3), "providerId": p["providerId"]})
    return segs


def frontier_line_segments(df_subset: pd.DataFrame, sorting_col: str) -> list[dict]:
    fr = compute_frontier(df_subset, acc_col="accuracy", sorting_col=sorting_col)
    pts = [
        {
            "x": float(row[sorting_col]),
            "y": float(row["accuracy"]),
            "providerId": row["providerId"],
        }
        for _, row in fr.iterrows()
    ]
    return step_segments(pts)


def build(
    models_csv: Path, providers_csv: Path, sources: list[dict]
) -> dict[str, object]:
    """Assemble all 5 leaderboard payloads from the current CSVs on disk.

    sources comes from resolve_sources(): one row per (benchmark, language) pair being
    published, each with its own summary CSV.
    """
    providers_db = load_providers(providers_csv)
    models_db, models_row_index = load_models(models_csv)

    # benchmark_domain ("LK-sw") -> {model_id: {accuracy, n, bad_format}}
    per_domain_results = {}
    for src in sources:
        print(f"  {src['display_name']}: {src['summary_csv'].name}")
        per_domain_results[src["benchmark_domain"]] = load_source_results(
            src["summary_csv"], src["accuracy_column"]
        )

    bench_model_names: set[str] = set()
    for results in per_domain_results.values():
        bench_model_names.update(results)

    unresolved = sorted(m for m in bench_model_names if m not in models_row_index)
    if unresolved:
        print("WARNING: no models.csv match - dropped:", ", ".join(unresolved))
    bench_model_names = sorted(m for m in bench_model_names if m in models_row_index)

    def get_provider(model_name: str) -> str:
        return models_db[model_name].get("provider") or "Unknown"

    # Positional IDs: row index in providers.csv / models.csv, so ids stay stable
    # across benchmarks that share these same config files.
    provider_id_map: dict[str, str] = {
        name: str(i) for i, name in enumerate(providers_db)
    }
    for j, provider in enumerate(
        sorted({get_provider(m) for m in bench_model_names} - set(provider_id_map))
    ):
        provider_id_map[provider] = str(len(providers_db) + j)
    all_provider_names = list(provider_id_map)
    model_id_map: dict[str, str] = {
        name: str(models_row_index[name]) for name in bench_model_names
    }

    # --- File 2: providermeta (every known provider, even with zero models here) ---
    provider_models: dict[str, dict] = {pid: {} for pid in provider_id_map.values()}
    for model_name in bench_model_names:
        pid = provider_id_map[get_provider(model_name)]
        mid = model_id_map[model_name]
        meta = models_db[model_name]
        provider_models[pid][mid] = {
            "id": mid,
            "name": model_name,
            "displayName": meta["display_name"],
            "input_price": meta["input_cost"],
            "output_price": meta["output_cost"],
            "blended_price": meta["blended_price"],
            "size": meta["size"],
            "open": meta["open"],
            "website": meta["url"],
            "release_date": meta["release_date"],
        }

    provider_meta: dict[str, dict] = {}
    for provider in all_provider_names:
        pid = provider_id_map[provider]
        cfg = providers_db.get(provider, {"color": "#888888", "svg": ""})
        provider_meta[pid] = {
            "metadata": {
                "id": pid,
                "name": provider,
                "displayName": provider,
                "color": cfg["color"],
                "logo": cfg["svg"],
            },
            "models": provider_models.get(pid, {}),
        }

    # --- File 3: data (one record per model x benchmark-language pair with a result) ---
    data: list[dict] = []
    for model_name in bench_model_names:
        mid = model_id_map[model_name]
        pid = provider_id_map[get_provider(model_name)]
        for src in sources:
            entry = per_domain_results[src["benchmark_domain"]].get(model_name)
            if entry is None:
                continue
            data.append(
                {
                    "benchmark_domain": src["benchmark_domain"],
                    "modelId": mid,
                    "providerId": pid,
                    "accuracy": entry["accuracy"],
                    "bad_format": entry["bad_format"],
                    "n": entry["n"],
                    "categories": [],
                }
            )

    # --- File 1: configmeta ---
    benchmarks_meta = {}
    for src in sources:
        results = per_domain_results[src["benchmark_domain"]]
        default_models = sorted(
            (m for m in bench_model_names if m in results),
            key=lambda m: results[m]["accuracy"],
            reverse=True,
        )
        benchmarks_meta[src["benchmark_domain"]] = {
            "displayName": src["display_name"],
            "categories": [],
            "default_models": [model_id_map[m] for m in default_models],
            "disabled_models": [],
        }
    config_meta = {
        "base_options": {
            "benchmarks": [
                {"id": src["benchmark_domain"], "displayName": src["display_name"]}
                for src in sources
            ],
            "axes": {"x": "blended_price", "y": "accuracy"},
        },
        "benchmarks": benchmarks_meta,
        "axes": {
            "x": {"blended_price": "log", "size": "log"},
            "y": {"accuracy": "linear"},
        },
    }

    # --- Files 4 & 5: cost / size frontier lines ---
    rows = []
    for model_name in bench_model_names:
        meta = models_db[model_name]
        pid = provider_id_map[get_provider(model_name)]
        for src in sources:
            entry = per_domain_results[src["benchmark_domain"]].get(model_name)
            if entry is None:
                continue
            rows.append(
                {
                    "benchmark": src["benchmark_domain"],
                    "accuracy": entry["accuracy"],
                    "cost": meta["blended_price"],
                    "size": meta["size"],
                    "open": meta["open"],
                    "providerId": pid,
                }
            )
    bench_df = pd.DataFrame(rows)

    cost_frontier = []
    size_frontier = []
    for src in sources:
        domain = src["benchmark_domain"]
        df = bench_df[bench_df["benchmark"] == domain]

        cost_df = df[df["cost"].notnull() & (df["cost"] > 0)]
        cost_frontier.append(
            {
                "benchmark_domain": domain,
                "lineSegments": frontier_line_segments(cost_df, "cost"),
                "categories": [],
            }
        )

        size_df = df[df["open"] & df["size"].notnull()]
        segs = frontier_line_segments(size_df, "size") if not size_df.empty else []
        if segs:
            size_frontier.append(
                {
                    "benchmark_domain": domain,
                    "lineSegments": segs,
                    "categories": [],
                }
            )

    # Keys are the <file-name> segment of the output filenames.
    return {
        "configmeta": config_meta,
        "providermeta": provider_meta,
        "data": data,
        "data-costfrontier": cost_frontier,
        "data-sizefrontier": size_frontier,
    }


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
        help="directory to write the 5 JSON files to",
    )
    parser.add_argument("--models-csv", default=str(DATA_DIR / "models.csv"))
    parser.add_argument("--providers-csv", default=str(DATA_DIR / "providers.csv"))
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sources, skipped = resolve_sources()
    if skipped:
        print(f"No results yet, not published: {', '.join(skipped)}")

    print("Reading summary CSVs:")
    payloads = build(Path(args.models_csv), Path(args.providers_csv), sources)

    # One timestamp per run, generated now, shared by all 5 files. UTC ISO 8601 with the
    # time compacted: 2026-08-27T142530Z - sorts lexicographically, unambiguous about
    # timezone, and colon-free so the filename stays valid on Windows.
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H%M%SZ")

    print("Writing:")
    written_paths = []
    for file_name, payload in payloads.items():
        out_path = outdir / f"{timestamp}_{args.cluster}_{file_name}.json"
        out_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        written_paths.append(out_path)
        size_kb = out_path.stat().st_size // 1024
        print(f"  {out_path}  ({size_kb} KB)")

    data_records = payloads["data"]
    unique_models = {row["modelId"] for row in data_records}
    n_providers = len(payloads["providermeta"])
    print(
        f"\nDone. {n_providers} providers | {len(unique_models)} models | "
        f"{len(sources)} benchmarks | {len(data_records)} data records"
    )


if __name__ == "__main__":
    main()

"""Build the leaderboard JSON bundle the foundational-literacy website reads.

Every benchmark publishing to this leaderboard turns its own results into
``LeaderboardRecord``s; everything downstream of that lives here, so the payload
shape, the provider/model id assignment and the filename convention are defined
once rather than per benchmark.

THE BUNDLE IS CLUSTER-WIDE, NOT PER-BENCHMARK. configmeta and data each span every
benchmark x language pair, so one publish carries all of them and a second benchmark
uploading its own bundle would replace the first rather than merge with it. Hence the
two-stage split:

  per benchmark   results -> [LeaderboardRecord] -> write_records_file()
                                                    results/web/leaderboard_records_<domain>.json
  cluster         read_records_files() -> build_leaderboard_bundle()
                                       -> write_leaderboard_bundle()

The records file is the seam. It keeps every benchmark's own results layout (phonics
is run-scoped, linguistic-knowledge is experiment-scoped) out of this module, and
lets each adapter be tested in the repo where its authors work.

The four payloads, keyed by the <file-name> segment of their upload filenames:

  configmeta          per-pair display name, pre-selected and disabled model ids
  data                one record per model x pair with a result
  data-costfrontier   the drawn accuracy-vs-cost frontier line, as step segments
  data-sizefrontier   the same against parameter count, open-weight models only

The site reads a fifth, providermeta, which resolves the ids these payloads carry into
names, prices, sizes and provider colours. It is NOT built here: it is shared across
every benchmark cluster on the site and published from an Azure container we do not
own. models.csv and providers.csv are still read, because the ids in `data` are row
positions in those exact files -- which is also what makes them line up with the ids in
Azure's providermeta, since it is generated from the same two files. See
leaderboard_cli.check_provider_meta for the check that they really do line up.

This module only builds and writes local files. Uploading is the caller's business;
see publisher.upload_file.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Sequence
from dataclasses import fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from fab_benchmark_publisher.contracts import (
    BenchmarkPair,
    LeaderboardBundle,
    LeaderboardRecord,
    RecordsFile,
)
from fab_benchmark_publisher.frontier import (
    compute_frontier,
    create_default_models_list,
)
from fab_benchmark_publisher.results import validate_records

# Default-model curation, matching the CDPK leaderboard's production settings. See
# frontier.create_default_models_list for what each does.
#
# DEFAULT_MODELS_ADD_MISSING keeps the upstream kwarg name (add_missing_models) in the
# vendored module so that file still diffs cleanly against CDPK. Leaving it True is
# what guarantees every vertex of a drawn frontier line is pre-selected -- with it off,
# build_leaderboard_bundle warns for each one that is not.
DEFAULT_MODELS_MAX = 10
DEFAULT_MODELS_ADD_MISSING = True
DEFAULT_MODELS_FILTER_BY_FAMILY = True

# Audit record of which models curation actually picked, written alongside the 5
# leaderboard files. Mirrors the CDPK pipeline's benchmark_default_models.json. Its
# name stays outside the <timestamp>_<cluster>_*.json glob so a glob-based uploader
# ignores it.
DEFAULT_MODELS_RECORD_NAME = "benchmark_default_models.json"

# Stable name for a benchmark's records file, so the cluster publisher can find every
# contribution by glob without knowing who wrote it.
RECORDS_FILE_GLOB = "leaderboard_records_*.json"

# Declaration order, not a set: it fixes the key order of every record written, so two
# runs of the same scores produce byte-identical files and a records file can be diffed.
_RECORD_FIELDS: tuple[str, ...] = tuple(
    f.name for f in fields(LeaderboardRecord) if f.name != "extra"
)


def records_file_name(benchmark_domain: str) -> str:
    """Filename a benchmark writes its records to, e.g. 'LK-sw' -> the LK-sw file."""
    return f"leaderboard_records_{benchmark_domain}.json"


# --------------------------------------------------------------------------- #
# The records file: one benchmark's contribution
# --------------------------------------------------------------------------- #
def write_records_file(
    records: Iterable[LeaderboardRecord],
    *,
    benchmark: BenchmarkPair,
    outdir: str | Path,
    source: str | None = None,
    generated_at: str | None = None,
) -> Path:
    """Write one benchmark's records to <outdir>/leaderboard_records_<domain>.json.

    `source` should name where the scores came from (a summary CSV, a run id) so a
    reader can tell which run is on the leaderboard without re-deriving it.
    """
    materialized = validate_records(records)
    foreign = sorted(
        {r.benchmark_domain for r in materialized} - {benchmark.benchmark_domain}
    )
    if foreign:
        raise ValueError(
            f"records for {benchmark.benchmark_domain} also carry benchmark_domain(s) "
            f"{', '.join(foreign)}; one records file holds exactly one pair"
        )
    payload = {
        "benchmark_domain": benchmark.benchmark_domain,
        "display_name": benchmark.display_name,
        "source": source,
        "generated_at": generated_at or datetime.now(UTC).isoformat(),
        "records": _records_to_payload(materialized),
    }
    path = Path(outdir) / records_file_name(benchmark.benchmark_domain)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return path


def _records_to_payload(records: list[LeaderboardRecord]) -> list[dict[str, Any]]:
    """Records as JSON dicts, dropping unset optionals and flattening `extra`.

    Deliberately not results.records_to_payload: that one validates again (these are
    already validated) and, more importantly, this round-trips -- read_records_file
    reconstructs the same records from what this writes.
    """
    payload = []
    for record in records:
        row = {
            name: getattr(record, name)
            for name in _RECORD_FIELDS
            if getattr(record, name) is not None
        }
        row.update(record.extra)
        payload.append(row)
    return payload


def read_records_file(path: str | Path) -> RecordsFile:
    """Read one records file back into records plus the pair it fills."""
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"{path} is not valid JSON: {e}") from e
    if not isinstance(payload, dict):
        raise ValueError(
            f"{path} holds a bare {type(payload).__name__}; expected an object with "
            "'benchmark_domain', 'display_name' and 'records'"
        )
    missing = [
        k for k in ("benchmark_domain", "display_name", "records") if k not in payload
    ]
    if missing:
        raise ValueError(f"{path} is missing required key(s): {', '.join(missing)}")
    records = [_record_from_dict(row, path) for row in payload["records"]]
    return RecordsFile(
        benchmark=BenchmarkPair(
            benchmark_domain=payload["benchmark_domain"],
            display_name=payload["display_name"],
        ),
        records=records,
        source=payload.get("source"),
        generated_at=payload.get("generated_at"),
    )


def _record_from_dict(row: Any, path: Path) -> LeaderboardRecord:
    if not isinstance(row, dict):
        raise ValueError(f"{path}: every entry of 'records' must be an object")
    known = {k: v for k, v in row.items() if k in _RECORD_FIELDS}
    extra = {k: v for k, v in row.items() if k not in _RECORD_FIELDS}
    try:
        return LeaderboardRecord(**known, extra=extra)
    except TypeError as e:
        raise ValueError(f"{path}: unusable record {row!r}: {e}") from e


def find_records_files(dirs: Iterable[str | Path]) -> list[Path]:
    """Every records file in `dirs`, sorted, so a listing order never leaks into the
    published column order (that comes from the pairs config).

    A directory that does not exist contributes nothing rather than raising: locally
    each benchmark keeps its records under its own tree, and not every benchmark has
    produced any yet.
    """
    found: list[Path] = []
    for directory in dirs:
        found.extend(sorted(Path(directory).glob(RECORDS_FILE_GLOB)))
    return found


def read_records_files(paths: Iterable[str | Path]) -> list[RecordsFile]:
    """Read several records files, rejecting two that claim the same pair.

    Order is preserved, so the caller decides the leaderboard's column order rather
    than inheriting whatever order the filesystem happened to list.
    """
    contributions = [read_records_file(p) for p in paths]
    seen: dict[str, Path] = {}
    for path, contribution in zip(paths, contributions, strict=True):
        domain = contribution.benchmark.benchmark_domain
        if domain in seen:
            raise ValueError(
                f"two records files claim benchmark_domain '{domain}': "
                f"{seen[domain]} and {path}"
            )
        seen[domain] = Path(path)
    return contributions


# --------------------------------------------------------------------------- #
# Model / provider metadata
# --------------------------------------------------------------------------- #
def _float_or_none(value: str | None) -> float | None:
    value = (value or "").strip()
    return float(value) if value else None


def load_providers(providers_csv: str | Path) -> dict[str, dict]:
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


def load_models(models_csv: str | Path) -> tuple[dict[str, dict], dict[str, int]]:
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
                # blended = 3:1 input:output weighted average, matching the CDPK
                # leaderboard.
                "blended_price": round((inp * 3 + out) / 4, 6)
                if inp is not None and out is not None
                else None,
                "display_name": row.get("display_name", "").strip() or mid,
            }
    return models_db, models_row_index


# --------------------------------------------------------------------------- #
# Frontier lines
# --------------------------------------------------------------------------- #
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
    """The drawn frontier line: raw Pareto (smooth=False), never the concave hull.

    The hull is only used to *select* default models; the rendered line must pass
    through real models, so it stays un-smoothed.
    """
    fr = compute_frontier(
        df_subset, acc_col="accuracy", sorting_col=sorting_col, smooth=False
    )
    pts = [
        {
            "x": float(row[sorting_col]),
            "y": float(row["accuracy"]),
            "providerId": row["providerId"],
        }
        for _, row in fr.iterrows()
    ]
    return step_segments(pts)


# --------------------------------------------------------------------------- #
# The bundle
# --------------------------------------------------------------------------- #
def _results_by_domain(
    records: Iterable[LeaderboardRecord], benchmarks: Sequence[BenchmarkPair]
) -> dict[str, dict[str, dict]]:
    """benchmark_domain -> {model_id: {accuracy, n, bad_format}}.

    A record naming a pair that is not being published is an error rather than a
    silent drop: it means the caller's benchmark list and its records disagree, and
    the symptom would otherwise be a column quietly missing from the website.
    """
    per_domain: dict[str, dict[str, dict]] = {
        pair.benchmark_domain: {} for pair in benchmarks
    }
    for record in validate_records(records):
        results = per_domain.get(record.benchmark_domain)
        if results is None:
            raise ValueError(
                f"record for model '{record.model}' names benchmark_domain "
                f"'{record.benchmark_domain}', which is not in the published pairs "
                f"({', '.join(per_domain) or 'none'})"
            )
        if record.model in results:
            raise ValueError(
                f"duplicate record for model '{record.model}' in "
                f"'{record.benchmark_domain}'"
            )
        results[record.model] = {
            "accuracy": round(record.accuracy, 2),
            "n": record.n if record.n is not None else 0,
            "bad_format": record.bad_format if record.bad_format is not None else 0.0,
        }
    return per_domain


def build_leaderboard_bundle(
    records: Iterable[LeaderboardRecord],
    *,
    benchmarks: Sequence[BenchmarkPair],
    models_csv: str | Path,
    providers_csv: str | Path,
    provider_meta: dict | None = None,
    curate_defaults: bool = True,
    max_models: int = DEFAULT_MODELS_MAX,
    add_missing_models: bool = DEFAULT_MODELS_ADD_MISSING,
    filter_by_family: bool = DEFAULT_MODELS_FILTER_BY_FAMILY,
    manual_default_models: Sequence[str] = (),
) -> LeaderboardBundle:
    """Assemble the 4 leaderboard payloads from records plus the current CSVs on disk.

    `benchmarks` is every (benchmark, language) pair being published, in the order
    their columns should appear on the site. A pair with no records still holds its
    slot -- every model then shows as disabled for it, which is what the site needs
    in order to grey them out rather than imply they scored 0.

    `provider_meta` is the site's providermeta.json, from Azure; pass it and any model
    it cannot resolve is dropped with a warning rather than published as an id that
    means nothing. Pass None to skip that gate.

    curate_defaults=False pre-selects every model with a result instead of running
    frontier curation; the other knobs are forwarded to create_default_models_list.
    """
    providers_db = load_providers(providers_csv)
    models_db, models_row_index = load_models(models_csv)
    per_domain_results = _results_by_domain(records, benchmarks)

    bench_model_names: set[str] = set()
    for results in per_domain_results.values():
        bench_model_names.update(results)

    # Two gates, both of which mean "the site could not render this model anyway".
    # A dropped model's score is simply not published, so each drop is warned about
    # here and recorded on the bundle for the manifest.
    dropped: dict[str, str] = {}

    unresolved = sorted(m for m in bench_model_names if m not in models_row_index)
    if unresolved:
        print("WARNING: no models.csv match - dropped:", ", ".join(unresolved))
        dropped.update(dict.fromkeys(unresolved, "not in models.csv"))
    bench_model_names = sorted(m for m in bench_model_names if m in models_row_index)

    # Gate 2: the site resolves a data row's ids through providermeta, which Azure
    # owns and regenerates on its own schedule. A model added to models.csv since its
    # last regeneration has an id that means nothing there, so publishing it would put
    # a nameless, priceless row on the page. Skipping the gate entirely (None) is for
    # callers with no network, and it publishes whatever models.csv allows.
    if provider_meta is not None:
        listed = {mid for p in provider_meta.values() for mid in p.get("models", {})}
        unlisted = [
            m for m in bench_model_names if str(models_row_index[m]) not in listed
        ]
        if unlisted:
            print(
                "WARNING: not in the leaderboard's providermeta - dropped:",
                ", ".join(unlisted),
            )
            print("         ask for a providermeta refresh to publish them")
            dropped.update(dict.fromkeys(unlisted, "not in providermeta"))
        bench_model_names = [m for m in bench_model_names if m not in set(unlisted)]

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
    model_id_map: dict[str, str] = {
        name: str(models_row_index[name]) for name in bench_model_names
    }

    # --- File 2: data (one record per model x pair with a result) ---
    data: list[dict] = []
    for model_name in bench_model_names:
        mid = model_id_map[model_name]
        pid = provider_id_map[get_provider(model_name)]
        for pair in benchmarks:
            entry = per_domain_results[pair.benchmark_domain].get(model_name)
            if entry is None:
                continue
            data.append(
                {
                    "benchmark_domain": pair.benchmark_domain,
                    "modelId": mid,
                    "providerId": pid,
                    "accuracy": entry["accuracy"],
                    "bad_format": entry["bad_format"],
                    "n": entry["n"],
                    "categories": [],
                }
            )

    # --- Per-model frame, shared by default-model selection and the frontier lines ---
    rows = []
    for model_name in bench_model_names:
        meta = models_db[model_name]
        pid = provider_id_map[get_provider(model_name)]
        for pair in benchmarks:
            entry = per_domain_results[pair.benchmark_domain].get(model_name)
            if entry is None:
                continue
            rows.append(
                {
                    "benchmark": pair.benchmark_domain,
                    "model": model_name,
                    "accuracy": entry["accuracy"],
                    "cost": meta["blended_price"],
                    "size": meta["size"],
                    "open": meta["open"],
                    "providerId": pid,
                }
            )
    # Columns named explicitly so an all-unresolved run still yields a frame the
    # per-pair slicing below can index, rather than a column KeyError.
    bench_df = pd.DataFrame(
        rows,
        columns=[
            "benchmark",
            "model",
            "accuracy",
            "cost",
            "size",
            "open",
            "providerId",
        ],
    )
    # Selection keys off `open` being a real bool; a string "True" would make every
    # open/closed subset empty and silently yield empty default lists.
    if (
        not bench_df.empty and bench_df["open"].dtype != bool
    ):  # pragma: no cover - tripwire: load_models always yields real bools
        raise ValueError(
            f"expected a boolean 'open' column, got {bench_df['open'].dtype}"
        )

    # --- Default models, per (benchmark, language) pair ---
    # Keyed by benchmark_domain straight from `benchmarks`, so a new pair flows through
    # with no change here or in the shared frontier module.
    if curate_defaults:
        benchmark_dfs = {
            pair.benchmark_domain: bench_df[
                bench_df["benchmark"] == pair.benchmark_domain
            ].copy()
            for pair in benchmarks
        }
        print("\nSelecting default models:")
        _selected, default_by_domain = create_default_models_list(
            benchmark_dfs,
            models_csv_path=models_csv,
            cost_var="cost",
            max_models=max_models,
            add_missing_models=add_missing_models,
            filter_by_provider_family=filter_by_family,
            manual_default_models=manual_default_models,
        )
    else:
        # bench_model_names, not per_domain_results: a model with no models.csv row was
        # warned about and dropped above, and pre-selecting it here would only trip the
        # "absent from models.csv" check below -- an uncurated run must not fail on a
        # model a curated one merely skips.
        default_by_domain = {
            pair.benchmark_domain: [
                m
                for m in bench_model_names
                if m in per_domain_results[pair.benchmark_domain]
            ]
            for pair in benchmarks
        }

    # --- File 1: configmeta ---
    benchmarks_meta = {}
    for pair in benchmarks:
        domain = pair.benchmark_domain
        results = per_domain_results[domain]
        # create_default_models_list returns model ids alphabetically; re-sort by
        # accuracy so the pre-selected order matches what the leaderboard shows.
        default_models = sorted(
            (m for m in default_by_domain.get(domain, []) if m in results),
            key=lambda m: results[m]["accuracy"],
            reverse=True,
        )
        # Scored somewhere in the cluster, but not for this pair.
        disabled_models = sorted(m for m in bench_model_names if m not in results)
        unknown = [m for m in default_models if m not in model_id_map]
        if unknown:  # pragma: no cover - tripwire: curation only ever returns models
            # that are already in bench_model_names, hence in models.csv
            raise ValueError(
                f"{domain} default models absent from models.csv: {', '.join(unknown)}"
            )
        benchmarks_meta[domain] = {
            "displayName": pair.display_name,
            "categories": [],
            "default_models": [model_id_map[m] for m in default_models],
            "disabled_models": [model_id_map[m] for m in disabled_models],
        }
    config_meta = {
        "base_options": {
            "benchmarks": [
                {"id": pair.benchmark_domain, "displayName": pair.display_name}
                for pair in benchmarks
            ],
            "axes": {"x": "blended_price", "y": "accuracy"},
        },
        "benchmarks": benchmarks_meta,
        "axes": {
            "x": {"blended_price": "log", "size": "log"},
            "y": {"accuracy": "linear"},
        },
    }

    # --- Files 4 & 5: cost / size frontier lines (from the same bench_df as above) ---
    cost_frontier = []
    size_frontier = []
    for pair in benchmarks:
        domain = pair.benchmark_domain
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

        # A drawn frontier vertex that is not pre-selected renders as a line with no
        # dot on it -- the one curation regression a visitor would actually see.
        chosen = set(default_by_domain.get(domain, []))
        for sorting_col, subset in (("cost", cost_df), ("size", size_df)):
            if subset.empty:
                continue
            vertices = compute_frontier(
                subset, acc_col="accuracy", sorting_col=sorting_col, smooth=False
            )["model"]
            hidden = [m for m in vertices if m not in chosen]
            if hidden:
                print(
                    f"WARNING: {domain} {sorting_col}-frontier vertices missing from "
                    f"default_models: {', '.join(hidden)}"
                )

    # Alphabetical, matching the CDPK benchmark_default_models.json this mirrors:
    # a record is easier to diff between runs when its order does not churn every
    # time a score moves. configmeta keeps its own accuracy-desc ordering.
    default_models_record = {
        pair.benchmark_domain: sorted(
            m
            for m in default_by_domain.get(pair.benchmark_domain, [])
            if m in per_domain_results[pair.benchmark_domain]
        )
        for pair in benchmarks
    }

    # Keys are the <file-name> segment of the output filenames.
    payloads = {
        "configmeta": config_meta,
        "data": data,
        "data-costfrontier": cost_frontier,
        "data-sizefrontier": size_frontier,
    }
    return LeaderboardBundle(
        payloads=payloads, default_models=default_models_record, dropped=dropped
    )


def write_leaderboard_bundle(
    bundle: LeaderboardBundle,
    *,
    outdir: str | Path,
    cluster_name: str,
    timestamp: str | None = None,
) -> Path:
    """Write one bundle to <outdir>/<timestamp>/ and return that folder.

    Everything one run produces lives in its own folder, so runs never interleave in
    a directory listing and a whole run can be uploaded, archived or deleted as one
    unit. Inside it the files keep the website's
    <timestamp>_<cluster-name>_<file-name>.json convention, where <timestamp> is ISO
    8601 UTC with the time compacted -- it sorts lexicographically, is unambiguous
    about timezone, and is colon-free so it stays a valid Windows path.

    TODO: uploading the folder is not done here. Once the cluster publisher exists,
    push the four <timestamp>_<cluster>_*.json files (NOT the default-models record)
    to the Azure container with publisher.upload_file.
    """
    timestamp = timestamp or datetime.now(UTC).strftime("%Y-%m-%dT%H%M%SZ")
    bundle_dir = Path(outdir) / timestamp
    bundle_dir.mkdir(parents=True, exist_ok=True)

    print(f"Writing to {bundle_dir}:")
    for file_name, payload in bundle.payloads.items():
        out_path = bundle_dir / f"{timestamp}_{cluster_name}_{file_name}.json"
        out_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        size_kb = out_path.stat().st_size // 1024
        print(f"  {out_path.name}  ({size_kb} KB)")

    record_path = bundle_dir / DEFAULT_MODELS_RECORD_NAME
    record_path.write_text(
        json.dumps(bundle.default_models, indent=4, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"  {record_path.name}  (default models per benchmark)")
    return bundle_dir

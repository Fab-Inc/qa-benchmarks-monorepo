"""Snapshot and score finished Kiswahili models without changing existing artifacts.

Run from the repository root (defaults select the two September cluster runs)::

    uv run --frozen --package phonics-benchmark python \
        benchmarks/phonics_benchmark/scripts/analyze_finished_kiswahili_models.py

Use repeated --run-source options for local run directories or GCS run prefixes.
--items-source accepts a local JSONL file or GCS URI; --items-sha256 pins its bytes.
--gold-labels-source optionally supplies corrected nondecodable_words from another
local JSONL file or GCS URI. Its item IDs, texts and student profiles must match
the original dataset. These labels are used directly, without decodability scoring;
the original dataset hash still validates the runs. Both inputs are snapshotted.
Local snapshots can be reanalysed with these options, into a NEW --output-dir.
No provider calls or cloud writes are made. Every output file is created exclusively.

Outputs: input snapshots and their generation/hash manifest, verified_items.jsonl
(the gold labels used for scoring),
coverage_by_model.csv (including unfinished models), excluded_responses.csv,
results_with_scoring.csv, summary_by_model.csv, and analysis.json. The metrics
use successful, parseable responses only. Finished means every dataset item has a
saved row, including recorded failures. Repeated words count once within each item.

summary_by_model.csv includes paired item-bootstrap percentile intervals for all
metrics. Defaults: --n-bootstrap 10000 --ci-level 0.95 --seed 0. Each replicate
samples the full dataset with replacement and shares that draw across models;
unscorable model/item pairs remain excluded. Point estimates use the original data.
Item-averaged F1 is the unweighted mean of defined per-item F1 scores. Empty gold
and prediction sets give undefined F1 and are omitted, matching the existing scorer;
false alarms on fully decodable items have F1=0 and are included. Per-item output
also includes the gold non-decodable ratio among unique words for diagnostic plots.
Pooled F2 is 5*sum(TP) / (5*sum(TP) + sum(FP) + 4*sum(FN)); it gives missed
non-decodable words four times the denominator penalty of false alarms. Finished
models with call or parsing failures are included, using their scorable responses.
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
from contextlib import redirect_stdout
from datetime import UTC, datetime
from hashlib import file_digest
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from fab_benchmark_publisher import parse_gs_uri
from phonics_benchmark.curation import add_labels_and_analysis
from phonics_benchmark.labeling import get_vocabulary_set
from phonics_benchmark.parsing import parse_word_list
from phonics_benchmark.scoring import add_per_item_confusion_matrix, summarize_by_model

DEFAULT_RUNS = (
    "gs://phonics-benchmark/phonics/runs/sw-tusome-clusters-20260905/",
    "gs://phonics-benchmark/phonics/runs/sw-tusome-clusters-20260906-remaining/",
)
DEFAULT_ITEMS = (
    "gs://phonics-benchmark/phonics/datasets/sw-tusome-clusters-20260905.jsonl"
)
DEFAULT_SHA256 = "379b5b4afa60bb86772b692258d5193eac8062c1ceaf8ea1091c8c7e0bf852b1"
DEFAULT_N_BOOTSTRAP = 10_000
DECODABILITY_CONFIG = {
    "scoring_methods": [
        "score_known_graphemes_kiswahili",
        "score_known_clusters_and_patterns_kiswahili",
        "score_whole_words_kiswahili",
    ],
    "aggregations": {
        "kenya_tusome_score": {"name": "kiswahili_kenya_tusome_aggregator"},
    },
    "score_column": "kenya_tusome_score",
}
METRIC_COLUMNS = [
    "pooled_f1",
    "perfect_match_accuracy",
    "pooled_tp",
    "pooled_fp",
    "pooled_fn",
    "item_averaged_f1",
    "n_items_with_f1",
    "n_items_f1_undefined",
    "pooled_f2",
]
SCORED_COLUMNS = [
    "run_id",
    "model",
    "benchmark_item_id",
    "response_parsed",
    "nondecodable_words",
    "tp_count",
    "fp_count",
    "fn_count",
    "tn_count",
    "perfect_match",
    "item_f1",
    "num_nondecodable_words",
    "num_total_words",
    "nondecodable_ratio",
]
EXCLUDED_COLUMNS = [
    "run_id",
    "model",
    "benchmark_item_id",
    "exclusion_reason",
    "error_type",
    "error_message",
]


def now() -> str:
    return datetime.now(UTC).isoformat()


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def read_jsonl(path: Path) -> pd.DataFrame:
    with path.open(encoding="utf-8") as handle:
        return pd.DataFrame(json.loads(line) for line in handle if line.strip())


def require(frame: pd.DataFrame, columns: list[str], description: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{description} lacks columns: {sorted(missing)}")


class Snapshotter:
    """Read-only GCS/local copies, pinning each GCS object to one generation."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.artifacts: list[dict[str, Any]] = []
        self._client: Any = None

    @property
    def client(self) -> Any:
        if self._client is None:
            from google.cloud.storage import Client

            self._client = Client()
        return self._client

    def copy(self, source: str, destination: Path, *, blob: Any = None) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        generation = None
        if source.startswith("gs://"):
            if blob is None:
                bucket, name = parse_gs_uri(source)
                blob = self.client.bucket(bucket).blob(name)
                blob.reload(timeout=60)
            generation = int(blob.generation)
            # If this generation disappears while a worker saves a checkpoint,
            # fail rather than silently downloading a different generation.
            with destination.open("xb") as handle:
                blob.download_to_file(
                    handle, if_generation_match=generation, timeout=60
                )
        else:
            with Path(source).expanduser().open("rb") as src:
                before = digest(Path(source).expanduser())
                with destination.open("xb") as dst:
                    shutil.copyfileobj(src, dst)
            if digest(destination) != before:
                raise ValueError(f"Local source changed while being copied: {source}")
        self.artifacts.append(
            {
                "source": source,
                "generation": generation,
                "local_path": str(destination.relative_to(self.root)),
                "sha256": digest(destination),
                "size_bytes": destination.stat().st_size,
                "copied_at": now(),
            }
        )
        return destination

    def run(self, source: str, index: int) -> Path:
        destination = self.root / "inputs" / f"run-{index}"
        if source.startswith("gs://"):
            bucket, prefix = parse_gs_uri(source.rstrip("/"))
            self.copy(f"{source.rstrip('/')}/config.json", destination / "config.json")
            response_prefix = f"{prefix}/responses/"
            blobs = sorted(
                self.client.list_blobs(bucket, prefix=response_prefix),
                key=lambda blob: blob.name,
            )
            for blob in blobs:
                relative = blob.name.removeprefix(response_prefix)
                if "/" not in relative and relative.endswith(".jsonl"):
                    print(f"Snapshotting {relative}", flush=True)
                    self.copy(
                        f"gs://{bucket}/{blob.name}",
                        destination / "responses" / relative,
                    )
        else:
            path = Path(source).expanduser().resolve()
            self.copy(str(path / "config.json"), destination / "config.json")
            for response in sorted((path / "responses").glob("*.jsonl")):
                self.copy(str(response), destination / "responses" / response.name)
        return destination


def use_saved_gold_labels(items: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Replace only gold labels, after checking that model inputs are unchanged."""
    gold = read_jsonl(path)
    identity_columns = [
        column
        for column in (
            "content",
            "student_knowledge",
            "language_iso_code",
            "text_id",
            "student_knowledge_id",
        )
        if column in items
    ]
    require(
        gold,
        ["benchmark_item_id", "nondecodable_words", *identity_columns],
        "Gold-label dataset",
    )
    ids = gold["benchmark_item_id"]
    if (
        ids.isna().any()
        or ids.duplicated().any()
        or set(ids) != set(items["benchmark_item_id"])
    ):
        raise ValueError(
            "Gold-label dataset must contain exactly the original item IDs"
        )
    # Preserve the original item order, including for paired bootstrap draws.
    gold = gold.set_index("benchmark_item_id").loc[items["benchmark_item_id"]]
    gold = gold.reset_index()
    for column in identity_columns:
        if not gold[column].equals(items[column]):
            raise ValueError(f"Gold-label dataset changes original {column}")
    for words in gold["nondecodable_words"]:
        if not isinstance(words, list) or not all(
            isinstance(word, str) and word.strip() for word in words
        ):
            raise ValueError("Gold-label nondecodable_words must contain word lists")
    labelled = items.copy()
    labelled["nondecodable_words"] = gold["nondecodable_words"].map(
        lambda words: sorted({word.lower() for word in words})
    )
    labelled["vocab_set"] = get_vocabulary_set(labelled, "sw")
    for row in labelled.itertuples():
        if not set(row.nondecodable_words).issubset(row.vocab_set):
            raise ValueError(
                f"Gold labels contain words outside {row.benchmark_item_id}"
            )
    labelled["num_nondecodable_words"] = labelled["nondecodable_words"].map(len)
    labelled["num_total_words"] = labelled["vocab_set"].map(len)
    if labelled["num_total_words"].eq(0).any():
        raise ValueError("Gold-label dataset contains items with empty vocabulary")
    labelled["nondecodable_ratio"] = (
        labelled["num_nondecodable_words"] / labelled["num_total_words"]
    )
    return labelled


def verify_items(
    path: Path,
    expected_sha256: str,
    output: Path,
    *,
    gold_labels_path: Path | None = None,
) -> pd.DataFrame:
    if digest(path) != expected_sha256:
        raise ValueError("Dataset SHA-256 does not match --items-sha256")
    items = read_jsonl(path)
    require(items, ["benchmark_item_id", "content", "student_knowledge"], "Dataset")
    if items.empty or items["benchmark_item_id"].isna().any():
        raise ValueError("Dataset must have non-null item IDs and at least one item")
    if items["benchmark_item_id"].duplicated().any():
        raise ValueError("Dataset contains duplicate benchmark_item_id values")
    if "language_iso_code" in items and not items["language_iso_code"].eq("sw").all():
        raise ValueError("Dataset contains a language other than sw")
    if gold_labels_path is not None:
        print(f"Using supplied gold labels for {len(items)} items", flush=True)
        labelled = use_saved_gold_labels(items, gold_labels_path)
    else:
        print(f"Recomputing Tusome labels for {len(items)} items", flush=True)
        labelled = add_labels_and_analysis(
            items, DECODABILITY_CONFIG, "sw", score_column="kenya_tusome_score"
        )
        if len(labelled) != len(items):
            raise ValueError("Decodability scoring dropped items with empty vocabulary")
        differences = []
        if "nondecodable_words" in items:
            for index in items.index:
                saved = items.at[index, "nondecodable_words"]
                computed = labelled.at[index, "nondecodable_words"]
                if not isinstance(saved, list) or not all(
                    isinstance(w, str) for w in saved
                ):
                    raise ValueError(
                        "Dataset nondecodable_words must contain word lists"
                    )
                if {w.lower() for w in saved} != set(computed):
                    differences.append(
                        {
                            "benchmark_item_id": items.at[index, "benchmark_item_id"],
                            "saved": saved,
                            "recomputed": computed,
                        }
                    )
        if differences:
            write_json(output / "label_differences.json", differences)
            raise ValueError(
                f"Tusome labels differ on {len(differences)} items; "
                "see label_differences.json. No model scores were produced."
            )
    with (output / "verified_items.jsonl").open("x", encoding="utf-8") as handle:
        labelled.to_json(handle, orient="records", lines=True, force_ascii=False)
    return labelled


def validate_rows(rows: pd.DataFrame, config: dict, expected_ids: set[str]) -> None:
    require(
        rows,
        [
            "run_id",
            "model",
            "benchmark_item_id",
            "result_id",
            "output",
            "call_status",
            "success",
        ],
        "Response file",
    )
    identity = ["run_id", "model", "benchmark_item_id", "result_id"]
    if rows[identity].isna().any().any():
        raise ValueError("Response identities must not be null")
    if not rows["run_id"].eq(config["run_id"]).all():
        raise ValueError("Response run_id differs from config snapshot")
    if rows["model"].nunique() != 1 or not rows["model"].isin(config["models"]).all():
        raise ValueError("Response file must contain one configured model")
    if (
        rows["benchmark_item_id"].duplicated().any()
        or rows["result_id"].duplicated().any()
    ):
        raise ValueError("Duplicate response identities; refusing to double-count")
    unexpected = set(rows["benchmark_item_id"]) - expected_ids
    if unexpected:
        raise ValueError(f"Responses contain unexpected item IDs: {sorted(unexpected)}")
    if not rows["call_status"].isin(["success", "error"]).all():
        raise ValueError("Unknown call_status in response file")
    expected_success = rows["call_status"].eq("success")
    if not rows["success"].map(lambda value: isinstance(value, bool)).all():
        raise ValueError("success must be a boolean")
    if not rows["success"].eq(expected_success).all():
        raise ValueError("success and call_status disagree")
    for column, expected in [
        ("language", "sw"),
        ("task", "decodability"),
        ("prompt_name", config["prompt_name"]),
    ]:
        if column in rows and not rows[column].eq(expected).all():
            raise ValueError(f"Response {column} differs from run configuration")


def parse_responses(rows: pd.DataFrame) -> pd.DataFrame:
    """Retain call failures separately; never interpret them as empty answers."""
    rows = rows.copy()
    parsed, reasons = [], []
    for row in rows.to_dict(orient="records"):
        if row["call_status"] != "success":
            parsed.append(None)
            reasons.append("call_failed")
        elif not isinstance(row["output"], str) or not row["output"].strip():
            parsed.append(None)
            reasons.append("parse_failed")
        else:
            # The existing parser prints prose warnings; keep terminal output
            # manageable without changing its parsing or scoring semantics.
            with redirect_stdout(io.StringIO()):
                result = parse_word_list(row["output"])
            parsed.append(result.words if result.parse_ok else None)
            reasons.append("" if result.parse_ok else "parse_failed")
    rows["response_parsed"] = parsed
    rows["exclusion_reason"] = reasons
    return rows


def validate_bootstrap_options(n_bootstrap: int, ci_level: float, seed: int) -> None:
    if n_bootstrap < 1:
        raise ValueError("--n-bootstrap must be positive")
    if not 0 < ci_level < 1:
        raise ValueError("--ci-level must be between 0 and 1 (exclusive)")
    if seed < 0:
        raise ValueError("--seed must be non-negative")


def bootstrap_intervals(
    scored: pd.DataFrame,
    item_ids: pd.Series,
    models: pd.DataFrame,
    *,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    ci_level: float = 0.95,
    seed: int = 0,
) -> pd.DataFrame:
    """Paired percentile bootstrap of profile/text items, retaining missing scores.

    ``models`` has one row per (run_id, model), including fully failed models.
    ``item_ids`` is the full dataset, including items excluded for some/all models.
    A validity count distinguishes exclusions from valid empty-list predictions.
    Zero-denominator replicates are undefined; report how many replicates contribute
    to each interval. Only one replicate's sampled item array is held at a time.
    """
    validate_bootstrap_options(n_bootstrap, ci_level, seed)
    items = pd.Index(item_ids)
    if items.empty or items.has_duplicates or items.hasnans:
        raise ValueError("Bootstrap requires non-empty, unique, non-null item IDs")
    keys = pd.MultiIndex.from_frame(models[["run_id", "model"]])
    if keys.has_duplicates:
        raise ValueError("Bootstrap requires unique (run_id, model) pairs")
    # Last dimension: TP, FP, FN, matches, scored count, item F1, defined-F1 count.
    values = np.zeros((len(items), len(keys), 7), dtype=np.float64)
    if not scored.empty:
        if scored.duplicated(["run_id", "model", "benchmark_item_id"]).any():
            raise ValueError("Bootstrap received duplicate model/item scores")
        item_positions = items.get_indexer(scored["benchmark_item_id"])
        model_positions = keys.get_indexer(
            pd.MultiIndex.from_frame(scored[["run_id", "model"]])
        )
        if (item_positions < 0).any() or (model_positions < 0).any():
            raise ValueError(
                "Scored responses fall outside the bootstrap item/model set"
            )
        values[item_positions, model_positions, :4] = scored[
            ["tp_count", "fp_count", "fn_count", "perfect_match"]
        ].to_numpy(dtype=np.int64)
        values[item_positions, model_positions, 4] = 1
        denominator = 2 * scored["tp_count"] + scored["fp_count"] + scored["fn_count"]
        item_f1 = (2 * scored["tp_count"] / denominator).where(denominator > 0)
        values[item_positions, model_positions, 5] = item_f1.fillna(0)
        values[item_positions, model_positions, 6] = item_f1.notna()

    result = models[["run_id", "model"]].reset_index(drop=True).copy()
    metric_names = [
        "pooled_f1",
        "perfect_match_accuracy",
        "item_averaged_f1",
        "pooled_f2",
    ]
    replicates = np.full((n_bootstrap, len(keys), len(metric_names)), np.nan)
    rng = np.random.default_rng(seed)
    if len(keys):
        print(f"Bootstrapping {n_bootstrap:,} shared item samples", flush=True)
        for replicate in range(n_bootstrap):
            indices = rng.integers(0, len(items), size=len(items))
            tp, fp, fn, matches, n_scored, f1_sum, n_f1 = values[indices].sum(axis=0).T
            denominator = 2 * tp + fp + fn
            np.divide(
                2 * tp,
                denominator,
                out=replicates[replicate, :, 0],
                where=denominator > 0,
            )
            np.divide(
                matches,
                n_scored,
                out=replicates[replicate, :, 1],
                where=n_scored > 0,
            )
            np.divide(
                f1_sum,
                n_f1,
                out=replicates[replicate, :, 2],
                where=n_f1 > 0,
            )
            f2_denominator = 5 * tp + fp + 4 * fn
            np.divide(
                5 * tp,
                f2_denominator,
                out=replicates[replicate, :, 3],
                where=f2_denominator > 0,
            )
    quantiles = [(1 - ci_level) / 2, (1 + ci_level) / 2]
    for metric_index, metric in enumerate(metric_names):
        bounds = np.full((len(keys), 2), np.nan)
        valid_counts = np.zeros(len(keys), dtype=int)
        for model_index in range(len(keys)):
            samples = replicates[:, model_index, metric_index]
            samples = samples[np.isfinite(samples)]
            valid_counts[model_index] = len(samples)
            if len(samples):
                bounds[model_index] = np.quantile(samples, quantiles)
        result[f"{metric}_ci_low"] = bounds[:, 0]
        result[f"{metric}_ci_high"] = bounds[:, 1]
        result[f"{metric}_n_bootstrap_valid"] = valid_counts
    return result


def score_runs(
    run_dirs: list[Path],
    items: pd.DataFrame,
    items_sha256: str,
    output: Path,
    *,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    ci_level: float = 0.95,
    seed: int = 0,
) -> pd.DataFrame:
    expected_ids = set(items["benchmark_item_id"])
    coverage, summaries, scored_frames, excluded_frames = [], [], [], []
    seen_runs: set[str] = set()
    common_prompt = None
    for run_dir in run_dirs:
        snapshot = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        config = snapshot["resolved_config"]
        run_id = config["run_id"]
        if run_id in seen_runs:
            raise ValueError(f"Repeated run_id: {run_id}")
        seen_runs.add(run_id)
        if config["language"] != "sw" or config["items_sha256"] != items_sha256:
            raise ValueError(f"Run {run_id} has a different language or dataset hash")
        if config.get("task", "decodability") != "decodability":
            raise ValueError(f"Run {run_id} is not a decodability run")
        if common_prompt is not None and common_prompt != config["prompt_name"]:
            raise ValueError("Runs used different prompts")
        common_prompt = config["prompt_name"]
        models = config["models"]
        if not models or len(models) != len(set(models)):
            raise ValueError(f"Invalid model selection in {run_id}")
        seen_models: set[str] = set()
        for path in sorted((run_dir / "responses").glob("*.jsonl")):
            rows = read_jsonl(path)
            if rows.empty:
                continue
            validate_rows(rows, config, expected_ids)
            model = rows["model"].iloc[0]
            if model in seen_models:
                raise ValueError(f"Multiple response files for {run_id}/{model}")
            seen_models.add(model)
            rows = parse_responses(rows)
            finished = set(rows["benchmark_item_id"]) == expected_ids
            scorable = rows["exclusion_reason"].eq("")
            counts = {
                "run_id": run_id,
                "model": model,
                "finished": finished,
                "n_expected": len(items),
                "n_recorded": len(rows),
                "n_missing": len(items) - len(rows),
                "n_call_failed": int(rows["exclusion_reason"].eq("call_failed").sum()),
                "n_parse_failed": int(
                    rows["exclusion_reason"].eq("parse_failed").sum()
                ),
                "n_scorable": int(scorable.sum()),
                "n_scored": int(scorable.sum()) if finished else 0,
            }
            coverage.append(counts)
            excluded_frames.append(
                rows.loc[~scorable].reindex(columns=EXCLUDED_COLUMNS)
            )
            if not finished:
                continue
            print(
                f"Scoring {model}: {counts['n_scored']}/{len(items)} items", flush=True
            )
            if not scorable.any():
                summaries.append(
                    {
                        **counts,
                        **dict.fromkeys(METRIC_COLUMNS, float("nan")),
                        "n_items_with_f1": 0,
                        "n_items_f1_undefined": 0,
                    }
                )
                continue
            # Join only identity and predictions, so saved response metadata cannot
            # shadow the selected gold labels or introduce ambiguous merge suffixes.
            joined = rows.loc[
                scorable,
                [
                    "run_id",
                    "model",
                    "benchmark_item_id",
                    "response_parsed",
                ],
            ].merge(items, on="benchmark_item_id", how="left", validate="many_to_one")
            scored = add_per_item_confusion_matrix(joined, "sw")
            scored["item_f1"] = scored["f1"]
            metrics = summarize_by_model(scored).iloc[0]
            f2_denominator = (
                5 * metrics["pooled_tp"]
                + metrics["pooled_fp"]
                + 4 * metrics["pooled_fn"]
            )
            summaries.append(
                {
                    **counts,
                    "pooled_f1": metrics["pooled_f1"],
                    "perfect_match_accuracy": metrics["accuracy"],
                    "pooled_tp": int(metrics["pooled_tp"]),
                    "pooled_fp": int(metrics["pooled_fp"]),
                    "pooled_fn": int(metrics["pooled_fn"]),
                    "item_averaged_f1": scored["item_f1"].mean(),
                    "n_items_with_f1": int(scored["item_f1"].notna().sum()),
                    "n_items_f1_undefined": int(scored["item_f1"].isna().sum()),
                    "pooled_f2": (
                        5 * metrics["pooled_tp"] / f2_denominator
                        if f2_denominator > 0
                        else float("nan")
                    ),
                }
            )
            scored_frames.append(scored[SCORED_COLUMNS])
        for model in models:
            if model not in seen_models:
                coverage.append(
                    {
                        "run_id": run_id,
                        "model": model,
                        "finished": False,
                        "n_expected": len(items),
                        "n_recorded": 0,
                        "n_missing": len(items),
                        "n_call_failed": 0,
                        "n_parse_failed": 0,
                        "n_scorable": 0,
                        "n_scored": 0,
                    }
                )
    coverage_df = pd.DataFrame(coverage).sort_values(["run_id", "model"])
    coverage_df.to_csv(output / "coverage_by_model.csv", index=False, mode="x")
    summary = pd.DataFrame(summaries, columns=[*coverage_df.columns, *METRIC_COLUMNS])
    summary = summary.sort_values(
        ["pooled_f1", "perfect_match_accuracy", "model"], ascending=[False, False, True]
    )
    scored_results = (
        pd.concat(scored_frames, ignore_index=True)
        if scored_frames
        else pd.DataFrame(columns=SCORED_COLUMNS)
    )
    intervals = bootstrap_intervals(
        scored_results,
        items["benchmark_item_id"],
        summary,
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
        seed=seed,
    )
    summary = summary.merge(intervals, on=["run_id", "model"], validate="one_to_one")
    summary.to_csv(output / "summary_by_model.csv", index=False, mode="x")
    for frames, columns, filename in [
        (scored_frames, SCORED_COLUMNS, "results_with_scoring.csv"),
        (excluded_frames, EXCLUDED_COLUMNS, "excluded_responses.csv"),
    ]:
        frame = (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=columns)
        )
        frame.to_csv(output / filename, index=False, mode="x")
    return summary


def analyze(
    run_sources: list[str],
    items_source: str,
    items_sha256: str,
    output: Path,
    *,
    gold_labels_source: str | None = None,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    ci_level: float = 0.95,
    seed: int = 0,
) -> pd.DataFrame:
    validate_bootstrap_options(n_bootstrap, ci_level, seed)
    # Claim the directory before any network access; even empty existing folders
    # are rejected. A failed analysis is retained and never reused implicitly.
    output.mkdir(parents=True, exist_ok=False)
    started_at = now()
    print(f"Analysis directory: {output.resolve()}", flush=True)
    snapshotter = Snapshotter(output)
    items_path = snapshotter.copy(
        items_source, output / "inputs" / "benchmark_items.jsonl"
    )
    gold_labels_path = (
        snapshotter.copy(gold_labels_source, output / "inputs" / "gold_labels.jsonl")
        if gold_labels_source is not None
        else None
    )
    items = verify_items(
        items_path, items_sha256, output, gold_labels_path=gold_labels_path
    )
    run_dirs = [snapshotter.run(source, i) for i, source in enumerate(run_sources, 1)]
    write_json(
        output / "snapshot_manifest.json",
        {
            "started_at": started_at,
            "completed_at": now(),
            "note": "Each object is pinned separately, at its download time.",
            "artifacts": snapshotter.artifacts,
        },
    )
    summary = score_runs(
        run_dirs,
        items,
        items_sha256,
        output,
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
        seed=seed,
    )
    repo = Path(__file__).resolve().parents[3]
    write_json(
        output / "analysis.json",
        {
            "started_at": started_at,
            "completed_at": now(),
            "run_sources": run_sources,
            "items_source": items_source,
            "items_sha256": items_sha256,
            "gold_labels_source": gold_labels_source,
            "gold_labels_sha256": (
                digest(gold_labels_path) if gold_labels_path is not None else None
            ),
            "n_expected": len(items),
            "n_finished_models": len(summary),
            "model_inclusion": (
                "Every dataset item has a saved row, including recorded call failures. "
                "Models with errors are included; failed or unparseable responses "
                "are excluded from metric calculations and counted separately."
            ),
            "bootstrap": {
                "method": "paired nonparametric percentile bootstrap",
                "resampling_unit": "benchmark_item_id (profile/text combination)",
                "shared_draws_across_models": True,
                "sample_size": len(items),
                "n_bootstrap": n_bootstrap,
                "ci_level": ci_level,
                "seed": seed,
                "undefined_replicates": (
                    "Excluded from interval quantiles; valid replicate counts "
                    "are reported separately for each metric."
                ),
            },
            "decodability_config": (
                DECODABILITY_CONFIG if gold_labels_path is None else None
            ),
            "script_sha256": digest(Path(__file__)),
            "uv_lock_sha256": digest(repo / "uv.lock"),
            "label_verification": (
                "Supplied gold labels used directly, without decodability scoring. "
                "Item IDs, texts and student profiles matched the original dataset; "
                "gold words were checked against each item's vocabulary."
                if gold_labels_path is not None
                else "Saved labels matched recomputed labels, if present."
            ),
            "metric_denominator": "Successful, parseable items (n_scored).",
            "perfect_match_accuracy": "Mean equality of lowercased word sets per item.",
            "pooled_f1": (
                "2 * sum(TP) / (2 * sum(TP) + sum(FP) + sum(FN)); "
                "undefined for a zero denominator."
            ),
            "pooled_f2": (
                "5 * sum(TP) / (5 * sum(TP) + sum(FP) + 4 * sum(FN)); "
                "undefined for a zero denominator. Paired bootstrap draws are shared "
                "with the other metrics. No item-averaged F2 is calculated."
            ),
            "item_averaged_f1": (
                "Unweighted mean of per-item 2*TP / (2*TP + FP + FN), over "
                "n_items_with_f1 successful, parseable items. Empty gold and "
                "prediction sets have undefined F1 and are omitted from the mean "
                "and counted in n_items_f1_undefined. False alarms on fully "
                "decodable items have F1=0 and are included. Bootstrap replicates "
                "recompute the mean over sampled items with defined F1."
            ),
            "nondecodable_ratio": (
                "Gold non-decodable unique word count / total unique word count "
                "per item, using the gold labels in verified_items.jsonl."
            ),
        },
    )
    print(
        summary[
            [
                "model",
                "pooled_f1",
                "pooled_f1_ci_low",
                "pooled_f1_ci_high",
                "perfect_match_accuracy",
                "perfect_match_accuracy_ci_low",
                "perfect_match_accuracy_ci_high",
                "item_averaged_f1",
                "item_averaged_f1_ci_low",
                "item_averaged_f1_ci_high",
                "pooled_f2",
                "pooled_f2_ci_low",
                "pooled_f2_ci_high",
                "n_items_with_f1",
                "n_scored",
                "n_call_failed",
                "n_parse_failed",
            ]
        ].to_string(index=False)
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--run-source",
        action="append",
        help="Repeat for each GCS prefix or local run directory.",
    )
    parser.add_argument("--items-source", default=DEFAULT_ITEMS)
    parser.add_argument("--items-sha256", default=DEFAULT_SHA256)
    parser.add_argument(
        "--gold-labels-source",
        help="Use saved nondecodable_words from this local JSONL file or GCS URI.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=DEFAULT_N_BOOTSTRAP,
        help="Number of paired item resamples (default: 10000).",
    )
    parser.add_argument(
        "--ci-level",
        type=float,
        default=0.95,
        help="Confidence level between 0 and 1 (default: 0.95).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Non-negative random seed for reproducible intervals (default: 0).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Must not exist; defaults to a unique outputs/ directory.",
    )
    args = parser.parse_args()
    output = args.output_dir or (
        Path(__file__).resolve().parents[1]
        / "outputs"
        / f"kiswahili-analysis-{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    )
    try:
        analyze(
            args.run_source or list(DEFAULT_RUNS),
            args.items_source,
            args.items_sha256,
            output,
            gold_labels_source=args.gold_labels_source,
            n_bootstrap=args.n_bootstrap,
            ci_level=args.ci_level,
            seed=args.seed,
        )
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        parser.exit(1, f"Analysis stopped: {error}\n")


if __name__ == "__main__":
    main()

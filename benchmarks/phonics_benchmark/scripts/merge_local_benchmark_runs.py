"""Build analysis-only runs from a downloaded snapshot, without network access.

Run from the worktree root. Original responses and configs are kept under sources/.
The generated configs point at local scoring labels, not the original run labels.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import tempfile
from collections import Counter, defaultdict
from hashlib import sha256
from pathlib import Path

import pandas as pd
from phonics_benchmark.labeling import get_vocabulary_set
from phonics_benchmark.parsing import parse_word_list
from phonics_benchmark.prompts import build_decodability_prompt

DEFAULT_ROOT = Path("outputs/phonics-merged-20260921")
DEFAULT_MANIFEST = Path(__file__).with_name("merged_sources_20260921.json")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def unique_items(rows: list[dict]) -> tuple[dict, list[dict]]:
    """Collapse exact duplicate items; refuse conflicting ID collisions."""
    items, duplicates = {}, []
    for line, row in enumerate(rows, 1):
        item_id = row["benchmark_item_id"]
        if item_id in items:
            if row != items[item_id]:
                raise ValueError(f"Conflicting item rows for {item_id}")
            duplicates.append({"benchmark_item_id": item_id, "duplicate_line": line})
        else:
            items[item_id] = row
    return items, duplicates


def validate_same_input(left: dict, right: dict) -> None:
    """Gold labels may change; the model's text and knowledge must not."""
    for field in ("benchmark_item_id", "content", "student_knowledge"):
        if left[field] != right[field]:
            raise ValueError(f"Changed {field} for {left['benchmark_item_id']}")


def replace_gold_labels(
    original: dict, updated: dict, language: str
) -> tuple[dict, list]:
    """Replace scoring labels only after validating item identity and vocabulary."""
    if original.keys() != updated.keys():
        raise ValueError("Updated gold must contain exactly the original item IDs")
    gold_fields = {"nondecodable_words", "num_nondecodable_words", "nondecodable_ratio"}
    frame = pd.DataFrame(list(original.values())).set_index(
        "benchmark_item_id", drop=False
    )
    vocabularies = get_vocabulary_set(frame, language)
    target, changes = {}, []
    for item_id, source in original.items():
        replacement = updated[item_id]
        validate_same_input(source, replacement)
        for field in (source.keys() | replacement.keys()) - gold_fields:
            if source.get(field) != replacement.get(field):
                raise ValueError(f"Changed non-gold field {field} for {item_id}")
        words = sorted({word.lower() for word in replacement["nondecodable_words"]})
        vocab = vocabularies.loc[item_id]
        if not set(words) <= vocab:
            raise ValueError(f"Gold words outside vocabulary for {item_id}")
        target[item_id] = {
            **source,
            "nondecodable_words": words,
            "num_nondecodable_words": len(words),
            "nondecodable_ratio": len(words) / len(vocab) if vocab else 0.0,
        }
        old_words = {word.lower() for word in source["nondecodable_words"]}
        if old_words != set(words):
            changes.append(
                {
                    "benchmark_item_id": item_id,
                    "old_nondecodable_words": sorted(old_words),
                    "new_nondecodable_words": words,
                    "removed_words": sorted(old_words - set(words)),
                    "added_words": sorted(set(words) - old_words),
                }
            )
    return target, changes


def deduplicate_calls(entries: list[tuple[dict, dict]]) -> tuple[list, list]:
    """Prefer success, then newest attempt; record every duplicate candidate.

    Parsing and scoring never affect selection. Original result/run IDs survive.
    The source path and line break exact timestamp ties deterministically.
    """
    by_call = defaultdict(list)
    for row, source in entries:
        if not row.get("model_call_id"):
            raise ValueError(f"Missing model_call_id at {source}")
        by_call[row["model_call_id"]].append((row, source))
    selected, audit = [], []
    for call_id, candidates in sorted(by_call.items()):
        identities = {
            (r["language"], r["model"], r["benchmark_item_id"], r["prompt"])
            for r, _ in candidates
        }
        if len(identities) != 1:
            raise ValueError(f"Conflicting call identity: {call_id}")
        ranked = sorted(
            candidates,
            key=lambda pair: (
                pair[0]["call_status"] == "success",
                pair[0].get("timestamp_utc", ""),
                pair[1]["file"],
                pair[1]["line"],
            ),
        )
        winner, winner_source = ranked[-1]
        selected.append((winner, winner_source, [s for _, s in ranked]))
        if len(ranked) > 1:
            success_outputs = {
                r.get("output") for r, _ in ranked if r["call_status"] == "success"
            }
            audit.append(
                {
                    "model_call_id": call_id,
                    "model": winner["model"],
                    "benchmark_item_id": winner["benchmark_item_id"],
                    "conflicting_success_outputs": len(success_outputs) > 1,
                    "selected_source": winner_source,
                    "candidates": [
                        {
                            **source,
                            "call_status": row["call_status"],
                            "timestamp_utc": row.get("timestamp_utc"),
                            "output_sha256": sha256(
                                (row.get("output") or "").encode()
                            ).hexdigest(),
                        }
                        for row, source in ranked
                    ],
                }
            )
    return selected, audit


def source_dataset_uri(snapshot: dict) -> str:
    config = snapshot["resolved_config"]
    uri = config.get("metadata", {}).get("items_source_uri") or snapshot.get(
        "raw_config", {}
    ).get("items_path")
    if not uri or not uri.startswith("gs://"):
        raise ValueError(f"Missing GCS dataset URI for {config['run_id']}")
    return uri


LANGUAGE_NAMES = {"en": "english", "sw": "kiswahili"}


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def manifest_runs(manifest: dict):
    """Read explicit languages; only recognize known legacy group names."""
    languages = {
        "en": "en",
        "sw_original": "sw",
        "sw_additional": "sw",
        **manifest.get("group_languages", {}),
    }
    seen = set()
    for group, runs in manifest["groups"].items():
        language = languages.get(group)
        if language not in LANGUAGE_NAMES:
            raise ValueError(f"Declare group_languages for {group}")
        for run in runs:
            if not run or Path(run).name != run or run in {".", ".."}:
                raise ValueError(f"Invalid run folder: {run}")
            if run in seen:
                raise ValueError(f"Repeated source run: {run}")
            seen.add(run)
            yield group, language, run


def normalize_items(items: dict, language: str) -> dict:
    """Derive scoring counts from the scorer's actual unique-word vocabulary."""
    if not items:
        return {}
    frame = pd.DataFrame(list(items.values())).set_index(
        "benchmark_item_id", drop=False
    )
    vocabularies = get_vocabulary_set(frame, language)
    result = {}
    for item_id, item in sorted(items.items()):
        if item.get("language_iso_code", language) != language:
            raise ValueError(f"Wrong item language: {item_id}")
        words = sorted({word.lower() for word in item["nondecodable_words"]})
        vocab = vocabularies.loc[item_id]
        if not vocab or not set(words) <= vocab:
            raise ValueError(f"Empty vocabulary or gold outside vocabulary: {item_id}")
        result[item_id] = {
            **item,
            "nondecodable_words": words,
            "num_total_words": len(vocab),
            "num_nondecodable_words": len(words),
            "nondecodable_ratio": len(words) / len(vocab),
        }
    return result


def ratio_band(item: dict) -> int:
    gold, vocab = item["num_nondecodable_words"], item["num_total_words"]
    return 0 if gold == 0 else (10 * gold - 1) // vocab + 1


def write_analysis_run(
    run_dir: Path,
    published_dir: Path,
    language: str,
    items: dict,
    selected: list,
    models: set,
    item_provenance: dict,
    metadata: dict,
) -> dict:
    """Write a full or selected run with counts derived from its own items."""
    if not items:
        raise ValueError(f"Empty analysis dataset: {language}")
    selected = sorted(
        selected, key=lambda v: (v[0]["model"], v[0]["benchmark_item_id"])
    )
    pairs = [(r["model"], r["benchmark_item_id"]) for r, _, _ in selected]
    if len(pairs) != len(set(pairs)):
        raise ValueError(f"Multiple distinct calls per model/item in {language}")
    if any(item_id not in items for _, item_id in pairs):
        raise ValueError("Responses outside the analysis dataset")
    models = sorted(set(models) | {model for model, _ in pairs})
    write_jsonl(run_dir / "benchmark_items.jsonl", (items[k] for k in sorted(items)))
    write_jsonl(run_dir / "responses/merged.jsonl", (r for r, _, _ in selected))
    write_jsonl(
        run_dir / "provenance.jsonl",
        (
            {
                "model_call_id": r["model_call_id"],
                "selected_source": s,
                "all_sources": origins,
            }
            for r, s, origins in selected
        ),
    )
    write_jsonl(
        run_dir / "item_provenance.jsonl", (item_provenance[k] for k in sorted(items))
    )
    statuses = Counter(item_provenance[k]["gold_status"] for k in items)
    write_json(
        run_dir / "config.json",
        {
            "analysis_only": True,
            "resolved_config": {
                "run_id": LANGUAGE_NAMES[language],
                "language": language,
                "items_path": str(published_dir / "benchmark_items.jsonl"),
                "items_sha256": digest(run_dir / "benchmark_items.jsonl"),
                "models": models,
                "metadata": {
                    **metadata,
                    "expected_items": len(items),
                    "gold_status_counts": dict(statuses),
                },
            },
        },
    )
    counts, band_counts, excluded = defaultdict(Counter), defaultdict(Counter), []
    with contextlib.redirect_stdout(io.StringIO()):
        for row, _, _ in selected:
            model, item_id = row["model"], row["benchmark_item_id"]
            reason = ""
            if row["call_status"] != "success":
                reason = "call_failed"
            elif not parse_word_list(row["output"]).parse_ok:
                reason = "parse_failed"
            for count in (
                counts[model],
                band_counts[(model, ratio_band(items[item_id]))],
            ):
                count["response_rows"] += 1
                count[reason or "scorable_items"] += 1
            if reason:
                excluded.append(
                    {
                        "model": model,
                        "benchmark_item_id": item_id,
                        "model_call_id": row["model_call_id"],
                        "exclusion_reason": reason,
                    }
                )

    def coverage_row(model, count, size):
        return {
            "model": model,
            "expected_items": size,
            "response_rows": count["response_rows"],
            "scorable_items": count["scorable_items"],
            "missing_responses": size - count["response_rows"],
            "call_failed": count["call_failed"],
            "parse_failed": count["parse_failed"],
            "scorable_fraction": count["scorable_items"] / size,
            "complete_scorable": count["scorable_items"] == size,
            "passes_90_percent_gate": count["scorable_items"] * 10 > size * 9,
            # Retain the historical flag for older analysis consumers.
            "passes_80_percent_gate": count["scorable_items"] * 5 > size * 4,
        }

    coverage = [coverage_row(model, counts[model], len(items)) for model in models]
    bands = Counter(ratio_band(item) for item in items.values())
    write_csv(run_dir / "coverage_by_model.csv", coverage)
    write_csv(
        run_dir / "coverage_by_model_and_ratio_band.csv",
        [
            {
                "ratio_band": band,
                **coverage_row(model, band_counts[(model, band)], size),
            }
            for model in models
            for band, size in sorted(bands.items())
        ],
    )
    write_jsonl(run_dir / "excluded_responses.jsonl", excluded)
    return {
        "items": len(items),
        "retained_rows": len(selected),
        "scorable_rows": sum(c["scorable_items"] for c in coverage),
        "expected_models": len(models),
        "models_with_responses": sum(c["response_rows"] > 0 for c in coverage),
        "models_with_scores": sum(c["scorable_items"] > 0 for c in coverage),
        "models_complete_scorable": sum(c["complete_scorable"] for c in coverage),
        "models_over90": sum(c["passes_90_percent_gate"] for c in coverage),
        "models_over80": sum(c["passes_80_percent_gate"] for c in coverage),
        "gold_status_counts": dict(statuses),
        "ratio_band_counts": dict(sorted(bands.items())),
    }


def seal_artifacts(directory: Path, details: dict) -> None:
    files = {
        str(p.relative_to(directory)): digest(p)
        for p in sorted(directory.rglob("*"))
        if p.is_file() and p.name != "artifact_manifest.json"
    }
    write_json(directory / "artifact_manifest.json", {**details, "files": files})


def verify_artifacts(directory: Path) -> dict:
    manifest = json.loads((directory / "artifact_manifest.json").read_text())
    for rel, expected in manifest["files"].items():
        path = (directory / rel).resolve()
        if not path.is_relative_to(directory.resolve()) or digest(path) != expected:
            raise ValueError(f"Merged artifact changed: {rel}")
    return manifest


def merge_snapshot(root: Path, manifest_path: Path) -> dict:
    """Always materialize the complete source-item union at root/full first."""
    root, manifest_path = root.resolve(), manifest_path.resolve()
    output = root / "full"
    if output.exists():
        raise FileExistsError(f"Preserving existing full dataset: {output}")
    manifest = json.loads(manifest_path.read_text())
    runs = list(manifest_runs(manifest))
    downloads = json.loads((root / "download_manifest.json").read_text())
    verified_paths = set()
    for entry in downloads["files"]:
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to(root) or digest(path) != entry["sha256"]:
            raise ValueError(f"Downloaded snapshot changed: {path}")
        verified_paths.add(path)
    failures = {d["source"]: d for d in downloads["downloads"] if d["returncode"]}
    for failure in failures.values():
        if "matched no objects or files" not in failure.get("error", ""):
            raise ValueError(f"Unresolved download failure: {failure}")

    def require_verified(path):
        if path.resolve() not in verified_paths:
            raise ValueError(f"File absent from download manifest: {path}")

    datasets, dataset_uris, duplicate_items = {}, {}, []

    def dataset(uri):
        if not uri.startswith("gs://"):
            raise ValueError(f"Expected GCS dataset URI: {uri}")
        name = uri.rsplit("/", 1)[-1]
        if name in dataset_uris and dataset_uris[name] != uri:
            raise ValueError(f"Dataset filename collision: {uri}")
        dataset_uris[name] = uri
        path = root / "sources/datasets" / name
        require_verified(path)
        if uri not in datasets:
            values, duplicates = unique_items(read_jsonl(path))
            datasets[uri] = values
            duplicate_items.extend({"dataset_uri": uri, **d} for d in duplicates)
        return datasets[uri], path

    pools = {language: {} for language in LANGUAGE_NAMES}
    origins = {language: {} for language in LANGUAGE_NAMES}
    entries = {language: [] for language in LANGUAGE_NAMES}
    models = {language: set() for language in LANGUAGE_NAMES}
    source_gold_conflicts = {language: set() for language in LANGUAGE_NAMES}
    seen_datasets, source_audit = set(), []
    for group, language, run in runs:
        run_dir = root / "sources/runs" / run
        uri = manifest["run_prefix"] + run
        config_path = run_dir / "config.json"
        if not config_path.exists():
            if uri not in failures:
                raise ValueError(f"Missing config without download record: {run}")
            source_audit.append(
                {
                    "destination": uri,
                    "group": group,
                    "language": language,
                    "status": "no_objects",
                    "dataset_uri": "",
                    "response_rows": 0,
                }
            )
            continue
        require_verified(config_path)
        snapshot = json.loads(config_path.read_text())
        config = snapshot["resolved_config"]
        if config["language"] != language:
            raise ValueError(f"Wrong language for {run}")
        dataset_uri = source_dataset_uri(snapshot)
        raw_items, dataset_path = dataset(dataset_uri)
        if digest(dataset_path) != config["items_sha256"]:
            raise ValueError(f"Dataset does not match destination config: {run}")
        if (language, dataset_uri) not in seen_datasets:
            seen_datasets.add((language, dataset_uri))
            status = manifest.get("source_gold_status", {}).get(dataset_uri, "original")
            if status not in {"original", "provisional", "corrected"}:
                raise ValueError(f"Unknown gold status: {status}")
            for item_id, item in normalize_items(raw_items, language).items():
                if item_id in pools[language]:
                    validate_same_input(pools[language][item_id], item)
                    if (
                        pools[language][item_id]["nondecodable_words"]
                        != item["nondecodable_words"]
                    ):
                        source_gold_conflicts[language].add(item_id)
                    origins[language][item_id]["source_datasets"].append(dataset_uri)
                else:
                    pools[language][item_id] = item
                    origins[language][item_id] = {
                        "benchmark_item_id": item_id,
                        "source_datasets": [dataset_uri],
                        "gold_sources": [dataset_uri],
                        "gold_status": status,
                    }
        models[language].update(config["models"])
        expected_prompt = (
            "decodability_evaluation_schwa_rule"
            if language == "en"
            else "decodability_evaluation_clusters"
        )
        if config["prompt_name"] != expected_prompt:
            raise ValueError(f"Unexpected prompt for {run}")
        prompts = {
            key: build_decodability_prompt(expected_prompt, item, language)
            for key, item in raw_items.items()
        }
        count = 0
        for path in sorted((run_dir / "responses").glob("*.jsonl")):
            require_verified(path)
            for line, row in enumerate(read_jsonl(path), 1):
                item_id = row["benchmark_item_id"]
                if (
                    item_id not in prompts
                    or row["language"] != language
                    or row["prompt"] != prompts[item_id]
                ):
                    raise ValueError(
                        f"Response inputs differ from config: {path}:{line}"
                    )
                if row["call_status"] not in {"success", "error"}:
                    raise ValueError(f"Unknown call status: {path}:{line}")
                entries[language].append(
                    (
                        row,
                        {
                            "destination": uri,
                            "dataset_uri": dataset_uri,
                            "file": str(path.relative_to(root)),
                            "line": line,
                        },
                    )
                )
                count += 1
        source_audit.append(
            {
                "destination": uri,
                "group": group,
                "language": language,
                "status": "downloaded",
                "dataset_uri": dataset_uri,
                "response_rows": count,
            }
        )

    overlays = manifest.get("gold_overlays")
    if overlays is None:
        # Older manifests remain readable, but their absent original-SW overlay
        # is explicitly labelled original gold in the full pool.
        overlays = {
            "en": [manifest["english_gold"]] if manifest.get("english_gold") else [],
            "sw": [
                uri
                for uri in manifest.get("updated_gold", [])
                if "syllabic_nasals" in uri and "/kiswahili" in uri
            ],
        }
    if set(overlays) - set(LANGUAGE_NAMES):
        raise ValueError("Unknown gold-overlay language")
    label_audit, overlay_audit = [], []
    for language in LANGUAGE_NAMES:
        uris = overlays.get(language, [])
        corrected = {}
        for uri in uris:
            replacements, _ = dataset(uri)
            matched = replacements.keys() & pools[language].keys()
            for item_id in sorted(matched):
                previous, updated = pools[language][item_id], replacements[item_id]
                validate_same_input(previous, updated)
                words = sorted({w.lower() for w in updated["nondecodable_words"]})
                if item_id in corrected and corrected[item_id] != words:
                    raise ValueError(f"Conflicting gold overlays: {item_id}")
                corrected[item_id] = words
                if previous["nondecodable_words"] != words:
                    label_audit.append(
                        {
                            "language": language,
                            "benchmark_item_id": item_id,
                            "gold_source": uri,
                            "old_nondecodable_words": previous["nondecodable_words"],
                            "new_nondecodable_words": words,
                        }
                    )
                pools[language][item_id] = {**previous, "nondecodable_words": words}
                origin = origins[language][item_id]
                if origin["gold_status"] != "corrected":
                    origin["gold_sources"] = []
                if uri not in origin["gold_sources"]:
                    origin["gold_sources"].append(uri)
                origin["gold_status"] = "corrected"
            overlay_audit.append(
                {
                    "language": language,
                    "dataset_uri": uri,
                    "matched_items": len(matched),
                    "items_outside_source_pool": sorted(replacements.keys() - matched),
                }
            )
        unresolved = source_gold_conflicts[language] - corrected.keys()
        if unresolved:
            raise ValueError(
                f"Conflicting source gold without overlay: {sorted(unresolved)}"
            )
    for language in LANGUAGE_NAMES:
        pools[language] = normalize_items(pools[language], language)

    audit = {
        "snapshot_started_at": downloads["started_at"],
        "source_runs": len(source_audit),
        "unavailable_runs": [
            s["destination"] for s in source_audit if s["status"] == "no_objects"
        ],
        "deduplication_policy": (
            "model_call_id; prefer success, then newest timestamp; "
            "never parse/score quality"
        ),
        "selection": (
            "all unique items in generation datasets; gold overlays do not add items"
        ),
        "languages": {},
    }
    with tempfile.TemporaryDirectory(prefix=".full-merge-", dir=root) as temporary:
        staging = Path(temporary) / "full"
        staging.mkdir()
        write_json(staging / "source_manifest.json", manifest)
        write_csv(staging / "audit/sources.csv", source_audit)
        write_csv(staging / "audit/duplicate_items.csv", duplicate_items)
        write_jsonl(staging / "audit/gold_label_changes.jsonl", label_audit)
        write_json(staging / "audit/gold_overlays.json", overlay_audit)
        for language, items in pools.items():
            if not items:
                continue
            selected, duplicates = deduplicate_calls(entries[language])
            name = LANGUAGE_NAMES[language]
            stats = write_analysis_run(
                staging / name,
                output / name,
                language,
                items,
                selected,
                models[language],
                origins[language],
                {
                    "dataset_kind": "full",
                    "source_manifest_sha256": digest(manifest_path),
                    "snapshot_started_at": downloads["started_at"],
                },
            )
            write_jsonl(staging / f"audit/{language}_duplicate_calls.jsonl", duplicates)
            audit["languages"][language] = {
                **stats,
                "raw_rows": len(entries[language]),
                "duplicate_rows_removed": len(entries[language]) - len(selected),
                "responses_outside_target": 0,
                "conflicting_success_calls": sum(
                    d["conflicting_success_outputs"] for d in duplicates
                ),
            }
        write_json(staging / "audit/merge_summary.json", audit)
        write_json(
            staging / "README.json",
            {
                "description": (
                    "Complete source-item union; "
                    "optional subsets are separate datasets."
                ),
                "languages": audit["languages"],
                "gold_note": (
                    "English extension labels remain provisional; "
                    "see per-item provenance."
                ),
            },
        )
        seal_artifacts(
            staging,
            {
                "kind": "full",
                "source_snapshot": str(root),
                "download_manifest_sha256": digest(root / "download_manifest.json"),
                "merger_sha256": digest(Path(__file__)),
                "source_manifest_sha256": digest(manifest_path),
            },
        )
        staging.rename(output)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--selection",
        type=Path,
        help="Optionally select a separate subset after the full merge",
    )
    args = parser.parse_args()
    print(json.dumps(merge_snapshot(args.root, args.manifest), indent=2))
    if args.selection:
        from select_local_benchmark_subset import select_subset

        print(json.dumps(select_subset(args.root / "full", args.selection), indent=2))


if __name__ == "__main__":
    main()

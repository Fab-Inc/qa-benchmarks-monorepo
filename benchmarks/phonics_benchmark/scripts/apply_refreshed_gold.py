"""Publish a corrected full pool, then retain the previous fixed analysis IDs.

Uses the sibling merger/selector scripts. No GCS writes or model calls.
The latest English gold file is authoritative for its exact selected-item IDs;
gold on the rest of the parent full pool is left unchanged.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path


def main(args):
    import merge_local_benchmark_runs as merge
    from select_local_benchmark_subset import select_subset

    parent_full = args.parent_full.resolve()
    parent_selection = args.parent_selection.resolve()
    root = args.output_root.resolve()
    selection_name = args.selection_name or parent_selection.name
    if (
        not selection_name
        or selection_name in {".", ".."}
        or Path(selection_name).name != selection_name
    ):
        raise ValueError("Selection name must be a single directory name")
    if root.exists():
        raise FileExistsError(root)
    full_manifest = merge.verify_artifacts(parent_full)
    selection_manifest = merge.verify_artifacts(parent_selection)
    if full_manifest["kind"] != "full":
        raise ValueError("Gold updates must start from a full merged pool")
    previous, duplicates = merge.unique_items(
        merge.read_jsonl(parent_selection / "english/benchmark_items.jsonl")
    )
    updated, new_duplicates = merge.unique_items(merge.read_jsonl(args.gold))
    if duplicates or new_duplicates:
        raise ValueError("Gold and target must have unique item IDs")
    corrected, changes = merge.replace_gold_labels(previous, updated, "en")
    old_origin = {
        r["benchmark_item_id"]: r
        for r in merge.read_jsonl(parent_selection / "english/item_provenance.jsonl")
    }
    for change in changes:
        change["previous_gold_status"] = old_origin[change["benchmark_item_id"]][
            "gold_status"
        ]
        change["previous_gold_sources"] = old_origin[change["benchmark_item_id"]][
            "gold_sources"
        ]
    root.mkdir(parents=True)
    source = root / "sources/datasets" / args.gold.name
    source.parent.mkdir(parents=True)
    shutil.copy2(args.gold, source)
    update_record = {
        "source_uri": args.gold_uri,
        "source_sha256": merge.digest(source),
        "parent_full_root": str(parent_full),
        "parent_full_manifest_sha256": merge.digest(
            parent_full / "artifact_manifest.json"
        ),
        "parent_selection_root": str(parent_selection),
        "parent_selection_manifest_sha256": merge.digest(
            parent_selection / "artifact_manifest.json"
        ),
        "matched_items": len(corrected),
        "items_with_changed_gold": len(changes),
        "changed_items_by_previous_status": dict(
            Counter(r["previous_gold_status"] for r in changes)
        ),
        "gold_words_removed": sum(len(r["removed_words"]) for r in changes),
        "gold_words_added": sum(len(r["added_words"]) for r in changes),
        "model_inputs_unchanged": True,
        "response_snapshot_refreshed": True,
        "selection_policy": "Keep the exact previous English and Kiswahili "
        "item IDs, irrespective of updated ratio bands.",
    }
    merge.write_json(root / "gold_update.json", update_record)
    with tempfile.TemporaryDirectory(prefix=".full-gold-update-", dir=root) as tmp:
        staging = Path(tmp) / "full"
        staging.mkdir()
        stats = {}
        for language, name in (("en", "english"), ("sw", "kiswahili")):
            run = parent_full / name
            items, duplicate_items = merge.unique_items(
                merge.read_jsonl(run / "benchmark_items.jsonl")
            )
            if duplicate_items:
                raise ValueError("Duplicate full-pool items")
            origins = {
                row["benchmark_item_id"]: row
                for row in merge.read_jsonl(run / "item_provenance.jsonl")
            }
            if language == "en":
                if corrected.keys() - items.keys():
                    raise ValueError(
                        "Updated gold contains items outside the parent full pool"
                    )
                for item_id, replacement in corrected.items():
                    merge.validate_same_input(items[item_id], replacement)
                    items[item_id] = {
                        **items[item_id],
                        "nondecodable_words": replacement["nondecodable_words"],
                    }
                    origins[item_id] = {
                        **origins[item_id],
                        "gold_status": "corrected",
                        "gold_sources": [args.gold_uri],
                    }
                items = merge.normalize_items(items, language)
            response_origins = {
                row["model_call_id"]: row
                for row in merge.read_jsonl(run / "provenance.jsonl")
            }
            responses = [
                (
                    row,
                    response_origins[row["model_call_id"]]["selected_source"],
                    response_origins[row["model_call_id"]]["all_sources"],
                )
                for row in merge.read_jsonl(run / "responses/merged.jsonl")
            ]
            config = json.loads((run / "config.json").read_text())["resolved_config"]
            stats[name] = merge.write_analysis_run(
                staging / name,
                root / "full" / name,
                language,
                items,
                responses,
                set(config["models"]),
                origins,
                {
                    **config["metadata"],
                    "dataset_kind": "full",
                    "gold_update": update_record,
                },
            )
            for unchanged in (
                "responses/merged.jsonl",
                "provenance.jsonl",
                "coverage_by_model.csv",
                "excluded_responses.jsonl",
            ):
                if merge.digest(staging / name / unchanged) != merge.digest(
                    run / unchanged
                ):
                    raise ValueError(f"Unexpected change to {name}/{unchanged}")
            if language == "sw" and merge.digest(
                staging / name / "benchmark_items.jsonl"
            ) != merge.digest(run / "benchmark_items.jsonl"):
                raise ValueError("Kiswahili items changed")
        merge.write_json(staging / "audit/gold_update.json", update_record)
        merge.write_jsonl(staging / "audit/gold_label_changes.jsonl", changes)
        merge.write_json(staging / "audit/full_summary.json", stats)
        if merge.verify_artifacts(parent_full) != full_manifest:
            raise ValueError("Parent full artifacts changed")
        merge.seal_artifacts(
            staging,
            {
                "kind": "full",
                "parent_full_root": str(parent_full),
                "parent_full_manifest_sha256": update_record[
                    "parent_full_manifest_sha256"
                ],
                "gold_update": update_record,
            },
        )
        staging.rename(root / "full")

    spec = {
        "name": selection_name,
        "languages": {},
        "notes": [
            update_record["selection_policy"],
            "This is a fixed-ID gold update, not a fresh Beta-mixture selection.",
        ],
    }
    for language, directory in (("en", "english"), ("sw", "kiswahili")):
        spec["languages"][language] = {
            "item_ids": [
                r["benchmark_item_id"]
                for r in merge.read_jsonl(
                    parent_selection / directory / "benchmark_items.jsonl"
                )
            ]
        }
    spec_path = root / "fixed_selection.json"
    merge.write_json(spec_path, spec)
    selection_summary = select_subset(root / "full", spec_path)
    selected = root / "selections" / selection_name
    for language in ("english", "kiswahili"):
        target_ids = {
            row["benchmark_item_id"]
            for row in merge.read_jsonl(
                parent_selection / language / "benchmark_items.jsonl"
            )
        }
        fresh = {
            row["model_call_id"]: row
            for row in merge.read_jsonl(
                parent_full / language / "responses/merged.jsonl"
            )
            if row["benchmark_item_id"] in target_ids
        }
        actual = {
            row["model_call_id"]: row
            for row in merge.read_jsonl(selected / language / "responses/merged.jsonl")
        }
        if actual != fresh:
            raise ValueError(f"Selection changed latest responses: {language}")
    if merge.verify_artifacts(parent_selection) != selection_manifest:
        raise ValueError("Parent selected artifacts changed")
    reproduction = root / "reproduction"
    reproduction.mkdir()
    shutil.copy2(Path(__file__), reproduction / Path(__file__).name)
    for script in ("merge_local_benchmark_runs.py", "select_local_benchmark_subset.py"):
        shutil.copy2(Path(__file__).resolve().parent / script, reproduction / script)
    print(
        json.dumps(
            {
                "gold_update": update_record,
                "full": stats,
                "selection": selection_summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-full", type=Path, required=True)
    parser.add_argument("--parent-selection", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--gold-uri", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--selection-name", help="Defaults to the previous selection directory name"
    )
    main(parser.parse_args())

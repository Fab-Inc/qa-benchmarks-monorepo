"""Select a separate analysis dataset from an immutable full local merge.

Selections use gold labels and item IDs only, never model responses. Nothing in
the full dataset is rewritten. No GCS access or model calls are made.
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from hashlib import sha256
from pathlib import Path

from merge_local_benchmark_runs import (
    LANGUAGE_NAMES,
    digest,
    ratio_band,
    read_jsonl,
    seal_artifacts,
    unique_items,
    validate_same_input,
    verify_artifacts,
    write_analysis_run,
    write_json,
)


def mixture_quotas(spec: dict) -> list[int]:
    """Largest-remainder allocation for zero + Beta(1,b) + uniform."""
    size = spec.get("size", 1000)
    beta = spec.get("beta", 8)
    zero, uniform = spec.get("zero_share", 0.1), spec.get("uniform_share", 0.2)
    if (
        not isinstance(size, int)
        or isinstance(size, bool)
        or size < 1
        or not all(math.isfinite(v) for v in (beta, zero, uniform))
        or beta <= 0
        or min(zero, uniform) < 0
        or zero + uniform > 1
    ):
        raise ValueError("Invalid mixture size, beta or shares")
    beta_share = 1 - zero - uniform
    probabilities = [zero] + [
        beta_share * ((1 - i / 10) ** beta - (1 - (i + 1) / 10) ** beta) + uniform / 10
        for i in range(10)
    ]
    expected = [size * p for p in probabilities]
    quotas = [math.floor(n) for n in expected]
    order = sorted(range(11), key=lambda i: (-(expected[i] - quotas[i]), i))
    for i in order[: size - sum(quotas)]:
        quotas[i] += 1
    assert sum(quotas) == size
    return quotas


def choose_items(items: dict, rule: dict, language: str, spec_dir: Path):
    """Choose IDs, retaining the full dataset's labels and reporting shortages."""
    selectors = set(rule) & {"mixture", "items_file", "item_ids"}
    if len(selectors) != 1:
        raise ValueError("Choose exactly one of mixture, items_file or item_ids")
    if "mixture" in rule:
        spec = rule["mixture"]
        quotas = mixture_quotas(spec)
        policy = spec.get("shortfall", "error")
        if policy not in {"error", "keep_available"}:
            raise ValueError("shortfall must be error or keep_available")
        seed = spec.get("seed", 42)
        if not isinstance(seed, int):
            raise ValueError("Mixture seed must be an integer")
        bins = [[] for _ in quotas]
        for item_id, item in items.items():
            bins[ratio_band(item)].append(item_id)
        selected, allocation = [], []
        for band, (available, quota) in enumerate(zip(bins, quotas, strict=True)):
            ordered = sorted(
                available,
                key=lambda key: (
                    sha256(f"{seed}:{language}:{key}".encode()).hexdigest(),
                    key,
                ),
            )
            picked = ordered[:quota]
            selected.extend(picked)
            allocation.append(
                {
                    "ratio_band": band,
                    "requested": quota,
                    "available": len(available),
                    "selected": len(picked),
                    "shortfall": max(0, quota - len(available)),
                }
            )
        shortages = [row for row in allocation if row["shortfall"]]
        if shortages and policy == "error":
            raise ValueError(f"Insufficient items for mixture: {shortages}")
        details = {
            "method": "ratio_mixture",
            "specification": spec,
            "requested_items": sum(quotas),
            "selected_items": len(selected),
            "shortfall_items": sum(row["shortfall"] for row in allocation),
            "allocation": allocation,
            "selection_order": "SHA-256 of seed:language:item_id within each bin",
        }
    else:
        details = {"method": "explicit_ids"}
        if "items_file" in rule:
            source = Path(rule["items_file"])
            source = (
                (spec_dir / source).resolve()
                if not source.is_absolute()
                else source.resolve()
            )
            # A complete JSONL dataset is accepted as an ID selection, but its
            # labels cannot silently replace the full dataset's scoring gold.
            records = read_jsonl(source)
            selected = [record["benchmark_item_id"] for record in records]
            for record in records:
                item_id = record["benchmark_item_id"]
                if item_id in items:
                    for field in ("content", "student_knowledge"):
                        if field in record and record[field] != items[item_id][field]:
                            raise ValueError(f"Changed {field} for {item_id}")
                    if all(key in record for key in ("content", "student_knowledge")):
                        validate_same_input(items[item_id], record)
            details.update(
                {"items_file": str(source), "items_file_sha256": digest(source)}
            )
        else:
            selected = rule["item_ids"]
            if not isinstance(selected, list) or not all(
                isinstance(i, str) for i in selected
            ):
                raise ValueError("item_ids must be a list of strings")
        if len(selected) != len(set(selected)):
            raise ValueError("Duplicate selected item IDs")
        missing = sorted(set(selected) - items.keys())
        if missing:
            raise ValueError(f"Selected IDs absent from full dataset: {missing}")
        details.update(
            {
                "requested_items": len(selected),
                "selected_items": len(selected),
                "shortfall_items": 0,
            }
        )
    if not selected:
        raise ValueError("Selection has no items")
    return {key: items[key] for key in sorted(selected)}, details


def select_subset(
    full_root: Path, selection_path: Path, output_root: Path | None = None
):
    full_root, selection_path = full_root.resolve(), selection_path.resolve()
    parent_manifest = verify_artifacts(full_root)
    if parent_manifest.get("kind") != "full":
        raise ValueError("Selection requires a full merged dataset")
    spec = json.loads(selection_path.read_text())
    name = spec["name"]
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError("Selection name must be a single folder name")
    output = (output_root or full_root.parent / "selections" / name).resolve()
    if (
        output == full_root
        or output.is_relative_to(full_root)
        or full_root.is_relative_to(output)
    ):
        raise ValueError("Subset output must be separate from the full dataset")
    if output.exists():
        raise FileExistsError(f"Preserving existing subset: {output}")
    rules = spec["languages"]
    if not rules or set(rules) - set(LANGUAGE_NAMES):
        raise ValueError("Selection languages must be en and/or sw")
    prepared = {}
    for language, rule in rules.items():
        run = full_root / LANGUAGE_NAMES[language]
        items, duplicates = unique_items(read_jsonl(run / "benchmark_items.jsonl"))
        if duplicates:
            raise ValueError("Full dataset contains duplicate items")
        selected, details = choose_items(items, rule, language, selection_path.parent)
        origins = {
            row["benchmark_item_id"]: row
            for row in read_jsonl(run / "item_provenance.jsonl")
        }
        provenance = {
            row["model_call_id"]: row for row in read_jsonl(run / "provenance.jsonl")
        }
        responses = []
        for row in read_jsonl(run / "responses/merged.jsonl"):
            if row["benchmark_item_id"] in selected:
                origin = provenance[row["model_call_id"]]
                responses.append(
                    (row, origin["selected_source"], origin["all_sources"])
                )
        config = json.loads((run / "config.json").read_text())["resolved_config"]
        prepared[language] = (selected, details, origins, responses, config)
    audit = {
        "name": name,
        "full_root": str(full_root),
        "full_manifest_sha256": digest(full_root / "artifact_manifest.json"),
        "selection_spec_sha256": digest(selection_path),
        "languages": {},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".subset-", dir=output.parent) as temporary:
        staging = Path(temporary) / "selected"
        staging.mkdir()
        write_json(staging / "selection_spec.json", spec)
        for language, (items, details, origins, responses, config) in prepared.items():
            name = LANGUAGE_NAMES[language]
            stats = write_analysis_run(
                staging / name,
                output / name,
                language,
                items,
                responses,
                set(config["models"]),
                origins,
                {
                    **config["metadata"],
                    "dataset_kind": "subset",
                    "parent_full_root": str(full_root),
                    "selection_name": spec["name"],
                    "selection_spec_sha256": digest(selection_path),
                },
            )
            audit["languages"][language] = {**details, **stats}
        # Detect concurrent edits to full inputs before publishing any subset.
        verify_artifacts(full_root)
        write_json(staging / "audit/selection_summary.json", audit)
        seal_artifacts(
            staging,
            {
                "kind": "subset",
                "parent_full_root": str(full_root),
                "parent_manifest_sha256": audit["full_manifest_sha256"],
                "selector_sha256": digest(Path(__file__)),
            },
        )
        staging.rename(output)
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            select_subset(args.full_root, args.selection, args.output_root), indent=2
        )
    )

"""Reconcile the archived September 22 merge and September 23 Astra additions.

Read-only, offline audit of historical files, independent of the merger's helpers.
Writes small review tables and a compressed row-level ledger; never rebuilds runs.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

LANGUAGES = {"english": "en", "kiswahili": "sw"}
DROPPED = {"qwen-3-14b", "qwen3-8b-alibaba"}
DISPOSITIONS = (
    "identical_copy",
    "superseded_response",
    "outside_selection",
    "dropped_model",
    "retained",
    "added",
    "later_not_incorporated",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    with path.open(encoding="utf-8") as stream:
        for line, text in enumerate(stream, 1):
            if text.strip():
                yield line, json.loads(text)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def content_digest(value):
    """Compare JSON values exactly, allowing integral float/int representation."""

    def normalize(obj):
        if isinstance(obj, float):
            require(math.isfinite(obj), "Non-finite JSON number")
            return int(obj) if obj.is_integer() else obj
        if isinstance(obj, list):
            return [normalize(v) for v in obj]
        if isinstance(obj, dict):
            return {k: normalize(v) for k, v in obj.items()}
        return obj

    encoded = json.dumps(normalize(value), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


@dataclass(frozen=True)
class Response:
    call: str
    model: str
    item: str
    status: str
    timestamp: str
    sha256: str
    output_sha256: str
    input_sha256: str

    @classmethod
    def from_row(cls, row):
        return cls(
            row["model_call_id"],
            row["model"],
            row["benchmark_item_id"],
            row["call_status"],
            row.get("timestamp_utc", ""),
            content_digest(row),
            content_digest(row.get("output")),
            content_digest(
                {
                    key: row[key]
                    for key in ("language", "model", "benchmark_item_id", "prompt")
                }
            ),
        )


def response_index(paths):
    result, pairs = {}, set()
    for path in paths:
        for _, row in rows(path):
            response = Response.from_row(row)
            pair = (response.model, response.item)
            require(response.call not in result, f"Duplicate call in {path}")
            require(pair not in pairs, f"Duplicate model/item in {path}: {pair}")
            pairs.add(pair)
            result[response.call] = response
    return result


def item_index(path):
    result = {}
    for _, row in rows(path):
        key = row["benchmark_item_id"]
        require(key not in result, f"Duplicate item in {path}: {key}")
        result[key] = row
    return result


def location(source):
    return source["file"], source["line"]


def verify_response_set(expected, actual, label):
    require(set(expected) == set(actual), f"{label}: missing or unexpected call IDs")
    require(expected == actual, f"{label}: response values changed")


def verify_winner(occurrences, provenance, merged):
    """Check the recorded decision directly against all source occurrences."""
    require(
        len({row.input_sha256 for row in occurrences.values()}) == 1,
        f"Conflicting source call identity: {merged.call}",
    )
    winner = max(
        occurrences,
        key=lambda key: (
            occurrences[key].status == "success",
            occurrences[key].timestamp,
            *key,
        ),
    )
    require(
        location(provenance["selected_source"]) == winner,
        f"Unexpected duplicate winner: {merged.call}",
    )
    require(
        Counter(map(location, provenance["all_sources"]))
        == Counter(occurrences.keys()),
        f"Incomplete source occurrences: {merged.call}",
    )
    require(merged == occurrences[winner], f"Merged response changed: {merged.call}")
    return winner


def provenance_index(path):
    result = {}
    for _, row in rows(path):
        call = row["model_call_id"]
        require(call not in result, f"Duplicate provenance: {path}")
        result[call] = row
    return result


class Audit:
    def __init__(self, outputs, ledger):
        self.outputs = outputs.resolve()
        self.ledger = ledger
        self.hashes = {}
        self.by_source = defaultdict(Counter)
        self.checked = set()

    def label(self, path):
        return str(path.resolve().relative_to(self.outputs))

    def relocated(self, recorded):
        # Historical manifests contain absolute paths from the original checkout.
        suffix = str(recorded).split("/outputs/", 1)[-1]
        path = (self.outputs / suffix).resolve()
        require(path.is_relative_to(self.outputs), f"Path outside outputs: {recorded}")
        return path

    def check_hash(self, path, expected=None):
        name = self.label(path)
        actual = digest(path)
        require(expected is None or actual == expected, f"Hash mismatch: {name}")
        require(
            name not in self.hashes or self.hashes[name] == actual,
            f"Input changed during audit: {name}",
        )
        self.hashes[name] = actual
        return actual

    def sealed(self, root, expected=None):
        manifest_path = root / "artifact_manifest.json"
        self.check_hash(manifest_path, expected)
        manifest = read_json(manifest_path)
        if root not in self.checked:
            for relative, checksum in manifest["files"].items():
                path = (root / relative).resolve()
                require(path.is_relative_to(root.resolve()), "Manifest escapes root")
                self.check_hash(path, checksum)
            self.checked.add(root)
        return manifest

    def record(self, language, snapshot, source, response, disposition, winner):
        source_file, line = source
        run = Path(source_file).parts[2]
        key = (language, self.label(snapshot), run, response.model)
        self.by_source[key][disposition] += 1
        self.ledger.writerow(
            {
                "language": language,
                "snapshot": self.label(snapshot),
                "source_file": source_file,
                "source_line": line,
                "model": response.model,
                "benchmark_item_id": response.item,
                "model_call_id": response.call,
                "disposition": disposition,
                "selected_file": winner[0],
                "selected_line": winner[1],
                "response_sha256": response.sha256,
            }
        )

    def snapshot(self, selection, language, classify):
        """Check raw sources -> full merge -> corrected full -> fixed selection."""
        selected_manifest = self.sealed(selection)
        corrected = self.relocated(selected_manifest["parent_full_root"])
        corrected_manifest = self.sealed(
            corrected, selected_manifest["parent_manifest_sha256"]
        )
        full = self.relocated(corrected_manifest["parent_full_root"])
        full_manifest = self.sealed(
            full, corrected_manifest["parent_full_manifest_sha256"]
        )
        snapshot = full.parent
        self.check_hash(
            snapshot / "download_manifest.json",
            full_manifest["download_manifest_sha256"],
        )
        self.check_hash(
            full / "source_manifest.json", full_manifest["source_manifest_sha256"]
        )
        downloads = read_json(snapshot / "download_manifest.json")
        files = {entry["path"]: entry["sha256"] for entry in downloads["files"]}
        manifest = read_json(full / "source_manifest.json")
        candidates = defaultdict(dict)
        source_items = {}
        source_metadata = {}
        for group, runs in manifest["groups"].items():
            if manifest["group_languages"][group] != LANGUAGES[language]:
                continue
            for run in runs:
                directory = snapshot / "sources/runs" / run
                config_path = directory / "config.json"
                self.check_hash(
                    config_path, files[str(config_path.relative_to(snapshot))]
                )
                config = read_json(config_path)["resolved_config"]
                require(
                    config["language"] == LANGUAGES[language], "Wrong source language"
                )
                uri = config["metadata"]["items_source_uri"]
                dataset = snapshot / "sources/datasets" / uri.rsplit("/", 1)[-1]
                self.check_hash(dataset, config["items_sha256"])
                require(
                    files[str(dataset.relative_to(snapshot))] == config["items_sha256"],
                    f"Dataset not sealed: {dataset}",
                )
                for item_id, item in item_index(dataset).items():
                    inputs = (item["content"], item["student_knowledge"])
                    require(
                        item_id not in source_items or source_items[item_id] == inputs,
                        f"Source item inputs disagree: {item_id}",
                    )
                    source_items[item_id] = inputs
                actual_files = {
                    str(p.relative_to(snapshot))
                    for p in (directory / "responses").glob("*.jsonl")
                }
                declared = {
                    p
                    for p in files
                    if p.startswith(f"sources/runs/{run}/responses/")
                    and p.endswith(".jsonl")
                }
                require(actual_files == declared, f"Response files differ: {run}")
                for relative in sorted(actual_files):
                    path = snapshot / relative
                    source_metadata[relative] = {
                        "dataset_uri": uri,
                        "destination": manifest["run_prefix"] + run,
                    }
                    self.check_hash(path, files[relative])
                    for line, row in rows(path):
                        require(
                            row["language"] == LANGUAGES[language],
                            "Wrong response language",
                        )
                        response = Response.from_row(row)
                        candidates[response.call][(relative, line)] = response

        merged_items = item_index(full / language / "benchmark_items.jsonl")
        require(set(merged_items) == set(source_items), "Full item union differs")
        for item_id, row in merged_items.items():
            require(
                (row["content"], row["student_knowledge"]) == source_items[item_id],
                f"Merged item input changed: {item_id}",
            )
        merged = response_index([full / language / "responses/merged.jsonl"])
        require(
            set(merged) == set(candidates), "Full merged call IDs differ from sources"
        )
        provenance = provenance_index(full / language / "provenance.jsonl")
        require(set(provenance) == set(merged), "Incomplete full provenance")
        selected = response_index([selection / language / "responses/merged.jsonl"])
        selected_items = item_index(selection / language / "benchmark_items.jsonl")
        spec = read_json(selection / "selection_spec.json")
        require(
            set(selected_items)
            == set(spec["languages"][LANGUAGES[language]]["item_ids"]),
            "Selection differs from declared item IDs",
        )
        expected = {
            key: row for key, row in merged.items() if row.item in selected_items
        }
        verify_response_set(expected, selected, "Selection")
        self.check_hash(
            corrected / language / "responses/merged.jsonl",
            digest(full / language / "responses/merged.jsonl"),
        )
        corrected_items = item_index(corrected / language / "benchmark_items.jsonl")
        require(
            set(corrected_items) == set(merged_items), "Gold refresh changed item IDs"
        )
        for item_id, row in corrected_items.items():
            require(
                (row["content"], row["student_knowledge"]) == source_items[item_id],
                f"Gold refresh changed model input: {item_id}",
            )
        require(
            selected_items == {key: corrected_items[key] for key in selected_items},
            "Selected gold differs from corrected full dataset",
        )
        selected_provenance = provenance_index(
            selection / language / "provenance.jsonl"
        )
        require(
            selected_provenance == {key: provenance[key] for key in selected},
            "Selection changed response provenance",
        )

        stats = Counter(
            raw_rows=sum(map(len, candidates.values())),
            full_rows=len(merged),
            full_items=len(merged_items),
            selected_rows=len(selected),
            selected_items=len(selected_items),
        )
        for call, occurrences in candidates.items():
            entry = provenance[call]
            winner = verify_winner(occurrences, entry, merged[call])
            for source in [entry["selected_source"], *entry["all_sources"]]:
                relative, line = location(source)
                require(
                    source
                    == {"file": relative, "line": line, **source_metadata[relative]},
                    f"Source location metadata changed: {call}",
                )
            successes = {
                r.output_sha256 for r in occurrences.values() if r.status == "success"
            }
            stats["conflicting_success_calls"] += len(successes) > 1
            for source, response in occurrences.items():
                if source != winner:
                    disposition = (
                        "identical_copy"
                        if response.sha256 == merged[call].sha256
                        else "superseded_response"
                    )
                elif response.item not in selected_items:
                    disposition = "outside_selection"
                elif response.model in DROPPED:
                    disposition = "dropped_model"
                else:
                    disposition = "retained"
                stats[disposition] += 1
                if classify:
                    self.record(
                        language, snapshot, source, response, disposition, winner
                    )
        return selected, selected_items, selected_provenance, snapshot, stats


def write_csv(path, records, columns=None):
    records = list(records)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=columns or list(records[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(records)


def markdown_table(headers, records):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(str(v) for v in row) + " |" for row in records),
        ]
    )


def run(outputs, bundle, report, ledger_path):
    require(not report.exists(), f"Preserving existing report: {report}")
    require(not ledger_path.exists(), f"Preserving existing ledger: {ledger_path}")
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "language",
        "snapshot",
        "source_file",
        "source_line",
        "model",
        "benchmark_item_id",
        "model_call_id",
        "disposition",
        "selected_file",
        "selected_line",
        "response_sha256",
    ]
    stats_by_language = {}
    partial_ledger = ledger_path.with_suffix(".partial.gz")
    with gzip.open(partial_ledger, "wt", encoding="utf-8", newline="") as stream:
        ledger = csv.DictWriter(stream, fieldnames=columns)
        ledger.writeheader()
        audit = Audit(outputs, ledger)
        audit.sealed(bundle)
        info = read_json(bundle / "bundle_info.json")
        require(
            set(info["dropped_models"]) == DROPPED, "Historical dropped models changed"
        )
        selection = audit.relocated(info["source_selection"])
        audit.sealed(selection, info["source_selection_manifest_sha256"])
        require(len(info["additions"]) == 1, "Expected one historical Astra addition")
        addition = info["additions"][0]
        require(
            addition["models"] == ["gpt-6-astra-medium"], "Unexpected addition models"
        )
        later = audit.relocated(addition["selection"])
        audit.sealed(later, addition["selection_manifest_sha256"])
        for language in LANGUAGES:
            print(f"Auditing {language} source records", flush=True)
            selected, items, provenance, snapshot, stats = audit.snapshot(
                selection, language, True
            )
            offered, later_items, later_provenance, later_snapshot, _ = audit.snapshot(
                later, language, False
            )
            require(items == later_items, "Item values changed in later selection")
            expected = {
                key: row for key, row in selected.items() if row.model not in DROPPED
            }
            expected_provenance = {key: provenance[key] for key in expected}
            for call, row in offered.items():
                if call in selected:
                    require(
                        row == selected[call],
                        f"Later selection changed existing response: {call}",
                    )
                    continue
                source = location(later_provenance[call]["selected_source"])
                disposition = "later_not_incorporated"
                if row.model in addition["models"]:
                    expected[call] = row
                    expected_provenance[call] = later_provenance[call]
                    disposition = "added"
                stats[disposition] += 1
                audit.record(language, later_snapshot, source, row, disposition, source)
            require(
                set(selected) <= set(offered), "Later selection lost historical calls"
            )
            run_dir = bundle / language
            actual = response_index(sorted((run_dir / "responses").glob("*.jsonl")))
            verify_response_set(expected, actual, f"Unified {language}")
            require(
                item_index(run_dir / "benchmark_items.jsonl") == items,
                "Unified items differ from corrected selection",
            )
            final_provenance = provenance_index(run_dir / "provenance.jsonl")
            require(
                final_provenance == expected_provenance, "Unified provenance changed"
            )
            require(
                item_index(run_dir / "item_provenance.jsonl")
                == item_index(selection / language / "item_provenance.jsonl"),
                "Unified item provenance changed",
            )
            stats["unified_rows"] = len(actual)
            require(
                len(actual) == info["runs"][language]["response_rows"],
                "Bundle count disagrees",
            )
            exclusions = list(
                row for _, row in rows(run_dir / "excluded_responses.jsonl")
            )
            excluded_ids = [row["model_call_id"] for row in exclusions]
            require(
                len(excluded_ids) == len(set(excluded_ids)),
                "Duplicate scoring exclusion",
            )
            require(set(excluded_ids) <= actual.keys(), "Unknown scoring exclusion")
            for row in exclusions:
                response = actual[row["model_call_id"]]
                require(
                    (row["model"], row["benchmark_item_id"])
                    == (response.model, response.item),
                    "Scoring exclusion identity changed",
                )
                reason = row["exclusion_reason"]
                require(
                    reason in {"call_failed", "parse_failed"},
                    "Unknown scoring exclusion",
                )
                require(
                    (reason == "call_failed") == (response.status != "success"),
                    "Failure status differs from exclusion",
                )
                stats[reason] += 1
            require(
                {key for key, row in actual.items() if row.status != "success"}
                == {
                    r["model_call_id"]
                    for r in exclusions
                    if r["exclusion_reason"] == "call_failed"
                },
                "Incomplete failure exclusions",
            )
            stats["scorable_rows"] = len(actual) - len(exclusions)
            stats_by_language[language] = dict(stats)
        # Verify all recorded bytes again, detecting concurrent input changes.
        for relative, expected in list(audit.hashes.items()):
            audit.check_hash(audit.outputs / relative, expected)

    partial_ledger.rename(ledger_path)
    report.mkdir(parents=True)
    records = [
        {
            "language": k[0],
            "snapshot": k[1],
            "source_run": k[2],
            "model": k[3],
            **{name: counts[name] for name in DISPOSITIONS},
        }
        for k, counts in sorted(audit.by_source.items())
    ]
    write_csv(report / "responses_by_source_and_model.csv", records)
    grouped = defaultdict(Counter)
    for row in records:
        key = (row["language"], row["snapshot"], row["source_run"])
        grouped[key].update({name: row[name] for name in DISPOSITIONS})
    source_rows = [
        {
            "language": k[0],
            "snapshot": k[1],
            "source_run": k[2],
            **{name: counts[name] for name in DISPOSITIONS},
        }
        for k, counts in sorted(grouped.items())
    ]
    write_csv(report / "responses_by_source.csv", source_rows)
    write_csv(
        report / "input_hashes.csv",
        ({"path": k, "sha256": v} for k, v in sorted(audit.hashes.items())),
    )
    stages = [
        ("Downloaded September 22 rows", "raw_rows", 1),
        ("Identical copies removed", "identical_copy", -1),
        ("Other duplicate responses removed", "superseded_response", -1),
        ("Outside selected items", "outside_selection", -1),
        ("Two Qwen models removed", "dropped_model", -1),
        ("September 23 Astra additions", "added", 1),
        ("Final unified responses", "unified_rows", 1),
    ]
    stage_table = markdown_table(
        ["Transformation", "English", "Kiswahili"],
        [
            (
                label,
                *(
                    f"{sign * stats_by_language[lang].get(key, 0):,}"
                    for lang in LANGUAGES
                ),
            )
            for label, key, sign in stages
        ],
    )
    source_table = markdown_table(
        [
            "Language",
            "Snapshot",
            "Source run",
            "Copies",
            "Other duplicates",
            "Outside subset",
            "Qwen removed",
            "Kept",
            "Added",
            "Not incorporated",
        ],
        [
            (
                r["language"],
                r["snapshot"].replace("phonics-sources-", ""),
                r["source_run"],
                *(r[k] for k in DISPOSITIONS),
            )
            for r in source_rows
        ],
    )
    status_table = markdown_table(
        ["Final data", "English", "Kiswahili"],
        [
            (label, *(f"{stats_by_language[lang].get(key, 0):,}" for lang in LANGUAGES))
            for label, key in [
                ("Items", "selected_items"),
                ("Scorable responses", "scorable_rows"),
                ("Retained failed calls", "call_failed"),
                ("Retained parsing failures", "parse_failed"),
                ("Later responses not incorporated", "later_not_incorporated"),
                (
                    "Source call groups with differing successful outputs",
                    "conflicting_success_calls",
                ),
            ]
        ],
    )
    verification = {
        "status": "passed",
        "bundle": audit.label(bundle),
        "languages": stats_by_language,
        "verified_files": len(audit.hashes),
        "audit_script_sha256": digest(Path(__file__)),
        "ledger": audit.label(ledger_path),
        "ledger_sha256": digest(ledger_path),
    }
    (report / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    (report / "README.md").write_text(
        f"""# Preservation of the September unified phonics runs

The September 22 source snapshot was merged, restricted to 991 English and 1,000
Kiswahili items, and packaged as unified runs. Two Qwen models were removed because
their token limits did not match the unified runner configuration. The September 23
Astra additions were subsequently incorporated. All retained response values and
their recorded source locations match the archived source files.

## Actual response accounting

{stage_table}

{status_table}

Failed and unparseable responses remain in the unified files. They are excluded
from scoring, not from preservation. Parsing exclusions above are the sealed
historical classifications; this audit does not redefine or rerun the parser.
The later unincorporated rows are outside the September 22 starting population.
See the model breakdown for their identities and counts.

## Sources and dispositions

Each September 22 source occurrence appears exactly once below. The disposition
order is: duplicate, outside selected items, excluded model, retained. September 23
rows appear only when absent from the September 22 selection, avoiding counting
the same historical responses twice. Source runs are download destinations; a row's
original run ID can differ when earlier responses were copied there.

{source_table}

[Source table](responses_by_source.csv) ·
[Source and model table](responses_by_source_and_model.csv) ·
[Input hashes](input_hashes.csv) · [Verification](verification.json)

## Checks and reproduction

The audit reads the archived JSONL files independently of the merging helpers. It
checks file hashes, the complete source item union, exact response-ID sets and
model/item uniqueness, every provenance occurrence, and the selected duplicate
winner (success, then latest timestamp, then source path and line). It compares
all retained response fields; only equivalent integral JSON numbers such as
30000 and 30000.0 are normalized. Corrected labels are checked against the sealed
corrected dataset; their linguistic correctness is outside this preservation check.

The full response files survive the gold refresh unchanged, the selection contains
exactly its declared items, and the final item values match that corrected selection.
The Astra additions preserve every existing response. All inputs are hash-checked
again after the audit. No cloud access or model calls are involved.

Run from the repository root with the archived folders under `outputs/`:

```sh
python benchmarks/phonics_benchmark/scripts/audit_unified_preservation.py \\
  --report-dir outputs/merge-audit-review \\
  --ledger outputs/merge-audit-review-rows.csv.gz
```

Use `--outputs-root` to relocate the archives. Existing reports and ledgers are
refused. The compressed ledger records every source row's location, call ID,
content hash, disposition and selected occurrence. It remains under ignored
`outputs/`; the small tables and verification record are kept with this document.
The archived source files are required to rerun this check; normal CI runs the
small metric examples and audit rejection tests without downloading the archives.
""",
        encoding="utf-8",
    )
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--bundle", default="phonics-unified-leaderboard-20260922")
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("benchmarks/phonics_benchmark/docs/merge-audit-20260922"),
    )
    parser.add_argument(
        "--ledger", type=Path, default=Path("outputs/merge-audit-20260922-rows.csv.gz")
    )
    args = parser.parse_args()
    run(
        args.outputs_root,
        (args.outputs_root / args.bundle).resolve(),
        args.report_dir,
        args.ledger,
    )

"""Run-level extraction: turn a run's raw responses into pairs and a status report."""

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from reading_comprehension_question_generation.parsing import parse_qa_pairs
from reading_comprehension_question_generation.prompts import NUM_PAIRS

PAIRS_FILENAME = "pairs.jsonl"
RESPONSE_STATUS_FILENAME = "response_status.jsonl"
SUMMARY_FILENAME = "extraction_summary.json"

# Status of a response whose model call failed. Those rows have no output to parse.
CALL_ERROR = "call_error"


class ExtractionError(ValueError):
    """Raised when a run cannot be read; the message names the problem."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file, naming the file and line of any malformed line."""
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ExtractionError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return rows


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    """Write dict rows as JSONL, keeping non-ASCII text readable."""
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _row_order(row: dict[str, Any]) -> tuple[str, str]:
    """Sort key: model, then item id."""
    return str(row["model"]), str(row["benchmark_item_id"])


def _items_hash_check(run_dir: Path, items_path: Path | None) -> dict[str, Any]:
    """Compare the items file's SHA-256 with the one the run recorded.

    ``match`` is None when the check cannot be made (no recorded hash, or the items
    file is not a readable local file). It is a report only and never fails the run.
    """
    config_path = run_dir / "config.json"
    resolved: dict[str, Any] = {}
    if config_path.exists():
        resolved = json.loads(config_path.read_text(encoding="utf-8")).get(
            "resolved_config", {}
        )
    recorded = resolved.get("items_sha256")
    path = items_path or (
        Path(resolved["items_path"]) if resolved.get("items_path") else None
    )
    current = None
    if path is not None and path.is_file():
        current = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "items_path": str(path) if path else None,
        "recorded_sha256": recorded,
        "current_sha256": current,
        "match": (recorded == current) if recorded and current else None,
    }


def extract_run(run_dir: Path, items_path: Path | None = None) -> dict[str, Any]:
    """Extract pairs from every response in ``run_dir`` and write the three outputs.

    Reads ``run_dir/responses/*.jsonl`` and writes ``pairs.jsonl``,
    ``response_status.jsonl`` and ``extraction_summary.json`` into ``run_dir``,
    overwriting earlier copies. Parse problems are recorded, not raised. Returns the
    summary.
    """
    response_files = sorted((run_dir / "responses").glob("*.jsonl"))
    if not response_files:
        raise ExtractionError(f"No response files found in {run_dir / 'responses'}")

    pair_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    for response_file in response_files:
        for row in _read_jsonl(response_file):
            item_id = row.get("benchmark_item_id")
            call_status = row.get("call_status")
            if call_status == "error":
                # The call failed, so there is nothing to parse.
                pairs, status = [], CALL_ERROR
            else:
                pairs, status = parse_qa_pairs(row.get("output"))
            base = {
                "run_id": row.get("run_id"),
                "model": row.get("model"),
                "benchmark_item_id": item_id,
                "text_id": item_id,  # the item id is the text_id (join key to texts)
            }
            status_rows.append(
                {
                    **base,
                    "call_status": call_status,
                    "parse_status": status,
                    "requested_pairs": NUM_PAIRS,
                    "extracted_pairs": len(pairs),
                }
            )
            pair_rows.extend(
                {
                    **base,
                    "pair_index": index,
                    "question": pair.question,
                    "answer": pair.answer,
                    "parse_status": status,
                    "requested_pairs": NUM_PAIRS,
                    "extracted_pairs": len(pairs),
                }
                for index, pair in enumerate(pairs, start=1)
            )

    # A stable order, so reruns write identical files.
    status_rows.sort(key=_row_order)
    pair_rows.sort(key=lambda r: (*_row_order(r), r["pair_index"]))

    models: dict[str, dict[str, Any]] = {}
    for model in sorted({str(r["model"]) for r in status_rows}):
        model_rows = [r for r in status_rows if str(r["model"]) == model]
        models[model] = {
            "responses": len(model_rows),
            "total_pairs": sum(r["extracted_pairs"] for r in model_rows),
            "status_counts": dict(Counter(r["parse_status"] for r in model_rows)),
        }
    summary = {
        "run_id": status_rows[0]["run_id"] if status_rows else None,
        "requested_pairs": NUM_PAIRS,
        "items_file": _items_hash_check(run_dir, items_path),
        "models": models,
    }

    _write_jsonl(pair_rows, run_dir / PAIRS_FILENAME)
    _write_jsonl(status_rows, run_dir / RESPONSE_STATUS_FILENAME)
    (run_dir / SUMMARY_FILENAME).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary

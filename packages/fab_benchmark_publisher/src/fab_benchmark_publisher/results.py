import json
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fab_benchmark_publisher.contracts import LeaderboardRecord


def validate_records(records: Iterable[LeaderboardRecord]) -> list[LeaderboardRecord]:
    """Materialize records and perform lightweight result checks."""

    materialized = list(records)
    if not materialized:
        raise ValueError("No leaderboard records were provided")
    for record in materialized:
        if record.accuracy < 0 or record.accuracy > 100:
            raise ValueError("accuracy must be a percentage between 0 and 100")
        if record.n is not None and record.n < 0:
            raise ValueError("n must be greater than or equal to 0")
    return materialized


def records_to_payload(records: Iterable[LeaderboardRecord]) -> list[dict[str, Any]]:
    """Convert validated records to JSON-serializable dictionaries."""

    payload = []
    for record in validate_records(records):
        row = {
            key: value
            for key, value in asdict(record).items()
            if value is not None and key != "extra"
        }
        row.update(record.extra)
        payload.append(row)
    return payload


def results_to_json(records: Iterable[LeaderboardRecord]) -> str:
    """Serialize records as pretty-printed JSON for website upload."""

    return json.dumps(records_to_payload(records), ensure_ascii=False, indent=2)


def write_results_json(
    records: Iterable[LeaderboardRecord],
    output_path: str | Path,
) -> Path:
    """Write a website result payload to disk and return the path."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(results_to_json(records), encoding="utf-8")
    return path

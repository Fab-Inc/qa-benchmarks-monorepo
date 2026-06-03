from pathlib import Path

import pytest
from fab_benchmark_publisher import (
    LeaderboardRecord,
    records_to_payload,
    results_blob_name,
    results_to_json,
    validate_records,
    write_results_json,
)


def test_records_to_payload_keeps_core_fields() -> None:
    payload = records_to_payload(
        [
            LeaderboardRecord(
                benchmark_domain="PHONICS",
                model="example-model",
                provider="Example",
                accuracy=88.0,
                subject="Decoding",
            )
        ]
    )

    assert payload == [
        {
            "benchmark_domain": "PHONICS",
            "model": "example-model",
            "accuracy": 88.0,
            "provider": "Example",
            "subject": "Decoding",
        }
    ]


def test_records_to_payload_merges_extra_fields() -> None:
    payload = records_to_payload(
        [
            LeaderboardRecord(
                benchmark_domain="PHONICS",
                model="example-model",
                accuracy=88.0,
                extra={"modelId": "123", "custom_metric": 0.42},
            )
        ]
    )

    assert payload[0]["modelId"] == "123"
    assert payload[0]["custom_metric"] == 0.42


def test_records_to_payload_rejects_invalid_accuracy() -> None:
    with pytest.raises(ValueError, match="accuracy"):
        records_to_payload(
            [
                LeaderboardRecord(
                    benchmark_domain="PHONICS",
                    model="example-model",
                    accuracy=101.0,
                )
            ]
        )


def test_validate_records_rejects_empty_records() -> None:
    with pytest.raises(ValueError, match="No leaderboard records"):
        validate_records([])


def test_records_to_payload_rejects_negative_n() -> None:
    with pytest.raises(ValueError, match="n"):
        records_to_payload(
            [
                LeaderboardRecord(
                    benchmark_domain="PHONICS",
                    model="example-model",
                    accuracy=88.0,
                    n=-1,
                )
            ]
        )


def test_results_to_json_serializes_payload() -> None:
    payload = results_to_json(
        [
            LeaderboardRecord(
                benchmark_domain="PHONICS",
                model="example-model",
                accuracy=88.0,
            )
        ]
    )

    assert '"benchmark_domain": "PHONICS"' in payload
    assert '"model": "example-model"' in payload


def test_write_results_json_creates_parent_directories(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "leaderboard.json"

    path = write_results_json(
        [
            LeaderboardRecord(
                benchmark_domain="PHONICS",
                model="example-model",
                accuracy=88.0,
            )
        ],
        output_path,
    )

    assert path == output_path
    assert '"accuracy": 88.0' in output_path.read_text(encoding="utf-8")


def test_results_blob_name_uses_file_prefix() -> None:
    assert (
        results_blob_name(
            benchmark_domain="PHONICS",
            artifact_name="leaderboard_data.json",
            file_prefix="dev",
        )
        == "dev/PHONICS/leaderboard_data.json"
    )

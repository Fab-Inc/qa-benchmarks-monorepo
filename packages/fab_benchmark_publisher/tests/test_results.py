import pytest
from fab_benchmark_publisher import (
    LeaderboardRecord,
    records_to_payload,
    results_blob_name,
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


def test_results_blob_name_uses_file_prefix() -> None:
    assert (
        results_blob_name(
            benchmark_domain="PHONICS",
            artifact_name="leaderboard_data.json",
            file_prefix="dev",
        )
        == "dev/PHONICS/leaderboard_data.json"
    )

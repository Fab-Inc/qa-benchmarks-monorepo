import json

import pytest
from phonics_benchmark.config import ConfigError
from reading_comprehension_question_generation.items import load_items

GOOD = json.dumps(
    {
        "benchmark_item_id": "A",
        "text_content": "Passage.",
        "language": "English",
        "grade": 2,
    }
)
OTHER = json.dumps(
    {
        "benchmark_item_id": "B",
        "text_content": "More.",
        "language": "English",
        "grade": 2,
    }
)


@pytest.mark.parametrize(
    ("lines", "expected"),
    [
        ([GOOD, "{not json"], r"items\.jsonl:2: invalid JSON"),
        ([GOOD, OTHER, GOOD], r"items\.jsonl:3: duplicate benchmark_item_id 'A'"),
        ([GOOD, "", OTHER], r"items\.jsonl:2: blank JSONL lines are not allowed"),
    ],
    ids=["malformed", "duplicate", "blank"],
)
def test_bad_lines_name_the_file_and_line(tmp_path, lines, expected) -> None:
    path = tmp_path / "items.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(ConfigError, match=expected):
        load_items(path)


def test_item_without_a_language_is_rejected_with_its_line(tmp_path) -> None:
    path = tmp_path / "items.jsonl"
    no_language = json.dumps(
        {"benchmark_item_id": "B", "text_content": "More.", "grade": 2}
    )
    path.write_text(f"{GOOD}\n{no_language}\n", encoding="utf-8")

    with pytest.raises(ConfigError, match=r"items\.jsonl:2: language must be"):
        load_items(path)


def test_item_without_a_grade_is_rejected_with_its_line(tmp_path) -> None:
    path = tmp_path / "items.jsonl"
    no_grade = json.dumps(
        {"benchmark_item_id": "B", "text_content": "More.", "language": "English"}
    )
    path.write_text(f"{GOOD}\n{no_grade}\n", encoding="utf-8")

    with pytest.raises(ConfigError, match=r"items\.jsonl:2: grade has the wrong type"):
        load_items(path)

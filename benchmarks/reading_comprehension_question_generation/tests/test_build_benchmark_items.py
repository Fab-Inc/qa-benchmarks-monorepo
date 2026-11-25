import json
from pathlib import Path
from types import ModuleType

import pytest
import typer
from typer.testing import CliRunner


def _cli(script: ModuleType) -> typer.Typer:
    """Wrap the script's command in an app so CliRunner can invoke it."""
    app = typer.Typer()
    app.command()(script.build_benchmark_items)
    return app


def _write_texts(
    data_root: Path, records: list[dict], language: str = "english"
) -> Path:
    path = data_root / language / "texts.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_builds_one_four_field_item_per_record_in_order(
    tmp_path, build_script, text_record
) -> None:
    records = [
        text_record(
            text_id="A-1",
            text_content="First passage.",
            language_iso_code="en",
            grade=2,
        ),
        text_record(
            text_id="B-2",
            text_content="Pili aliona paka.",
            language_iso_code="sw",
            grade=5,
            source="Some Book",
        ),
    ]
    data_root = tmp_path / "data"
    texts_path = _write_texts(data_root, records)
    before = texts_path.read_bytes()

    result = CliRunner().invoke(
        _cli(build_script), ["--language", "english", "--data-root", str(data_root)]
    )

    assert result.exit_code == 0, result.output
    assert "Built 2 benchmark item(s)" in result.output
    items = _read_jsonl(data_root / "english" / "benchmark_items.jsonl")
    assert items == [
        {
            "benchmark_item_id": "A-1",
            "text_content": "First passage.",
            "language": "English",
            "grade": 2,
        },
        {
            "benchmark_item_id": "B-2",
            "text_content": "Pili aliona paka.",
            "language": "Kiswahili",
            "grade": 5,
        },
    ]
    assert texts_path.read_bytes() == before
    assert all("benchmark_item_id" not in r for r in _read_jsonl(texts_path))


@pytest.mark.parametrize(
    ("code", "name"),
    [("EN", "English"), (" sw ", "Kiswahili"), ("fr", "fr")],
)
def test_language_code_maps_to_a_name_or_is_used_as_typed(
    build_script, code, name
) -> None:
    assert build_script.language_name(code) == name


def test_malformed_text_record_names_the_file_and_line(
    tmp_path, build_script, text_record
) -> None:
    broken = text_record(text_id="B")
    del broken["text_content"]
    data_root = tmp_path / "data"
    _write_texts(data_root, [text_record(text_id="A"), broken])

    with pytest.raises(build_script.ItemBuildError, match=r"texts\.jsonl:2:"):
        build_script.build_items_for_language("english", data_root)


def test_duplicate_text_id_names_the_file_and_line(
    tmp_path, build_script, text_record
) -> None:
    records = [
        text_record(text_id="A"),
        text_record(text_id="B"),
        text_record(text_id="A"),
    ]
    data_root = tmp_path / "data"
    _write_texts(data_root, records)

    with pytest.raises(
        build_script.ItemBuildError, match=r"texts\.jsonl:3: duplicate text_id 'A'"
    ):
        build_script.build_items_for_language("english", data_root)

    assert not (data_root / "english" / "benchmark_items.jsonl").exists()


def test_missing_texts_file_is_reported_without_a_traceback(
    tmp_path, build_script
) -> None:
    data_root = tmp_path / "data"
    (data_root / "english").mkdir(parents=True)

    result = CliRunner().invoke(
        _cli(build_script), ["--language", "english", "--data-root", str(data_root)]
    )

    assert result.exit_code == 1
    assert "ERROR:" in result.output
    assert "run ingest_texts.py first" in result.output
    assert "Traceback" not in result.output
    assert not (data_root / "english" / "benchmark_items.jsonl").exists()


def test_unsafe_language_argument_is_rejected(tmp_path, build_script) -> None:
    """The --language rule matches ingestion: a single folder name only."""
    with pytest.raises(build_script.ItemBuildError, match="Invalid --language"):
        build_script.build_items_for_language("a/b", tmp_path)

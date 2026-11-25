r"""Build benchmark items from ingested text records.

Reads ``data/<language>/texts.jsonl`` (written by ``ingest_texts.py``) and writes
``data/<language>/benchmark_items.jsonl`` next to it: one item per text, in the same
order, with exactly four keys:

- ``benchmark_item_id``: the text's ``text_id``. The shared run harness needs this key,
  which ``texts.jsonl`` does not have.
- ``text_content``: the passage.
- ``language``: the language name used in the prompt, mapped from the text's
  ``language_iso_code`` (``en`` to English, ``sw`` to Kiswahili; any other code is
  written as typed).
- ``grade``: the text's grade.

No other descriptive metadata is copied onto items; analysis joins back to
``texts.jsonl`` by ``text_id``. Every text becomes an item: this script filters nothing.
It is also where any future filtering would be added. The prompt is not built here; it
is rendered at run time.

Run from the benchmark folder with (as an example):

    cd benchmarks/reading_comprehension_question_generation
    uv run python scripts/build_benchmark_items.py --language english
"""

import json
from pathlib import Path

import typer
from phonics_benchmark.data_io import write_jsonl
from pydantic import ValidationError
from reading_comprehension_question_generation.models import (
    BenchmarkItem,
    TextRecord,
)

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[1] / "data"

TEXTS_FILENAME = "texts.jsonl"
ITEMS_FILENAME = "benchmark_items.jsonl"

# Language names for the prompt, keyed by lower-case ISO code. Extend it when a
# language is added; a code that is not listed is used as typed, so an unlisted code
# still builds.
LANGUAGE_NAMES = {"en": "English", "sw": "Kiswahili"}


class ItemBuildError(ValueError):
    """Raised when items cannot be built; the message names the problem."""


def validate_language_folder(language: str) -> str:
    """Return ``language`` if it is safe to use as a single data-folder name.

    The argument only selects ``data/<language>/``. It is not looked up in a list and
    is not compared with the texts' language code.
    """
    if (
        not language.strip()
        or language in {".", ".."}
        or language != Path(language).name
        or "\\" in language
    ):
        raise ItemBuildError(
            f"Invalid --language '{language}': use a single folder name such as "
            "'english', without path separators."
        )
    return language


def language_name(code: str) -> str:
    """Map a language code to the name used in the prompt, or return it as typed."""
    code = code.strip()
    return LANGUAGE_NAMES.get(code.casefold(), code)


def read_text_records(texts_path: Path) -> list[TextRecord]:
    """Read ``texts.jsonl`` into text records, naming the file and line on errors."""
    try:
        lines = texts_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ItemBuildError(
            f"Could not read {texts_path}; run ingest_texts.py first: {exc}"
        ) from exc

    records: list[TextRecord] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        try:
            record = TextRecord.model_validate(json.loads(line))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ItemBuildError(
                f"{texts_path}:{line_number}: invalid text record: {exc}"
            ) from exc
        if record.text_id in seen_ids:
            raise ItemBuildError(
                f"{texts_path}:{line_number}: duplicate text_id '{record.text_id}'"
            )
        seen_ids.add(record.text_id)
        records.append(record)

    if not records:
        raise ItemBuildError(f"No text records were found in {texts_path}")
    return records


def build_items(records: list[TextRecord]) -> list[BenchmarkItem]:
    """Project every text record onto a benchmark item; nothing is excluded."""
    return [
        BenchmarkItem(
            benchmark_item_id=record.text_id,
            text_content=record.text_content,
            language=language_name(record.language_iso_code),
            grade=record.grade,
        )
        for record in records
    ]


def build_items_for_language(language: str, data_root: Path) -> tuple[Path, int]:
    """Build ``data_root/<language>/benchmark_items.jsonl`` from ``texts.jsonl``.

    Returns the written path and the number of items.
    """
    data_dir = data_root / validate_language_folder(language)
    items = build_items(read_text_records(data_dir / TEXTS_FILENAME))
    output_path = data_dir / ITEMS_FILENAME
    write_jsonl(items, output_path)
    return output_path, len(items)


def build_benchmark_items(
    language: str = typer.Option(
        ...,
        "--language",
        "-l",
        help="Data folder name under the data root, for example 'english'.",
    ),
    data_root: Path = typer.Option(
        DEFAULT_DATA_ROOT,
        "--data-root",
        file_okay=False,
        help="Root data directory. Defaults to the package's data/ folder.",
    ),
) -> None:
    """Write one benchmark item per ingested text for one language."""
    try:
        output_path, count = build_items_for_language(language, data_root)
    except ItemBuildError as exc:
        # Print input problems without a traceback; they are not bugs.
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(1) from exc

    typer.echo(f"Built {count} benchmark item(s) in {output_path}")
    typer.echo(
        "Next: point a run config's items_path at this file and run "
        "'reading-comprehension-question-generation run-generation'."
    )


def main() -> None:
    """Entry point."""
    typer.run(build_benchmark_items)


if __name__ == "__main__":
    main()

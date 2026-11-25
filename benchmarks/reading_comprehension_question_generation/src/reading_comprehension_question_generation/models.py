"""Data models for the reading comprehension question generation benchmark."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

NonEmptyStr = Annotated[str, Field(min_length=1)]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]


class TextRecord(BaseModel):
    """One passage and its metadata, as entered in the workbook's ``Texts`` sheet.

    Only ``text_id``, ``text_content``, ``language_iso_code`` and ``grade`` drive the
    pipeline, so only they are required. ``grade`` is a positive whole number (ingestion
    parses it from the workbook cell). The other eleven fields are descriptive metadata
    kept for later analysis. They are stored exactly as typed (trimmed strings, or
    ``None`` when blank, including the count-like ``word_count`` and ``page_count``) and
    no processing step reads them. Field order follows the workbook columns.

    Any other workbook column is ignored. That includes the template's
    ``existing_comprehension_questions`` and ``notes`` columns, which are not carried
    for now because they may need preprocessing.
    """

    model_config = ConfigDict(extra="forbid")

    text_id: NonEmptyStr
    text_title: str | None = None
    text_content: NonEmptyStr
    text_type: str | None = None
    country_code: str | None = None
    language_iso_code: NonEmptyStr
    grade: PositiveInt
    source: str | None = None
    source_url: str | None = None
    word_count: str | None = None
    page_count: str | None = None
    source_grade_label: str | None = None
    is_adaptation: str | None = None
    adaptation_source: str | None = None
    has_images: str | None = None


class BenchmarkItem(BaseModel):
    """One benchmark item, the minimal input a run needs.

    An item is a four-field projection of a text record: ``benchmark_item_id`` (the
    text's ``text_id``), ``text_content``, ``language`` (the language name that goes
    into the prompt) and ``grade`` (the record's whole-number grade, unchanged). The
    shared run harness needs the ``benchmark_item_id`` key, which text records do not
    have. Other descriptive metadata is not copied onto items; it stays in
    ``texts.jsonl`` and is joined back by ``text_id`` for analysis. The prompt is
    rendered from the item at run time, so items hold no prompt.
    """

    model_config = ConfigDict(extra="forbid")

    benchmark_item_id: NonEmptyStr
    text_content: NonEmptyStr
    language: NonEmptyStr
    grade: PositiveInt


class QAPair(BaseModel):
    """One question and its answer, as extracted from a model response."""

    model_config = ConfigDict(extra="forbid")

    question: NonEmptyStr
    answer: NonEmptyStr

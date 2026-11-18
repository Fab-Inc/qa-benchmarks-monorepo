"""Data models for the reading comprehension question generation benchmark."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

NonEmptyStr = Annotated[str, Field(min_length=1)]


class TextRecord(BaseModel):
    """One passage and its metadata, as entered in the workbook's ``Texts`` sheet.

    Only ``text_id``, ``text_content`` and ``language_iso_code`` drive the pipeline,
    so only they are required. The other eleven fields are descriptive metadata
    kept for later analysis. They are stored exactly as typed (trimmed strings, or
    ``None`` when blank) and no processing step reads them. Field order follows the
    workbook columns.

    The workbook's ``existing_comprehension_questions`` and ``notes`` columns are not
    carried for now, because they may need preprocessing; ingestion ignores them like
    any other extra column.
    """

    model_config = ConfigDict(extra="forbid")

    text_id: NonEmptyStr
    text_title: str | None = None
    text_content: NonEmptyStr
    text_type: str | None = None
    country_code: str | None = None
    language_iso_code: NonEmptyStr
    grade: str | None = None
    source: str | None = None
    location: str | None = None
    source_grade_label: str | None = None
    is_target: str | None = None
    is_adaptation: str | None = None
    adaptation_source: str | None = None
    has_images: str | None = None


class BenchmarkItem(BaseModel):
    """One benchmark item, the minimal input a run needs.

    An item is a three-field projection of a text record: ``benchmark_item_id`` (the
    text's ``text_id``), ``text_content``, and ``language`` (the language name that
    goes into the prompt). The shared run harness needs the ``benchmark_item_id`` key,
    which text records do not have. Descriptive metadata is not copied onto items; it
    stays in ``texts.jsonl`` and is joined back by ``text_id`` for analysis. The prompt
    is rendered from the item at run time, so items hold no prompt.
    """

    model_config = ConfigDict(extra="forbid")

    benchmark_item_id: NonEmptyStr
    text_content: NonEmptyStr
    language: NonEmptyStr


class QAPair(BaseModel):
    """One question and its answer, as extracted from a model response."""

    model_config = ConfigDict(extra="forbid")

    question: NonEmptyStr
    answer: NonEmptyStr

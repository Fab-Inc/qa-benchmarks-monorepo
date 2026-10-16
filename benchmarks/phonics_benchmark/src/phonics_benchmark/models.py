"""Data models for phonics_benchmark."""

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

NonEmptyStr = Annotated[str, Field(min_length=1)]


class StudentKnowledge(BaseModel):
    model_config = ConfigDict(extra="forbid")
    student_knowledge_id: NonEmptyStr
    language_iso_code: NonEmptyStr
    jagged: bool
    student_knowledge: dict[str, Any]


class DecodableText(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text_id: NonEmptyStr
    language_iso_code: NonEmptyStr
    content: NonEmptyStr


class BenchmarkItem(BaseModel):
    """A text paired with a student profile — the pure pairing, no content."""

    model_config = ConfigDict(extra="forbid")
    benchmark_item_id: NonEmptyStr
    text_id: NonEmptyStr
    student_knowledge_id: NonEmptyStr


class LabelledBenchmarkItem(BaseModel):
    """A benchmark item with its text, knowledge and decodability analysis inlined."""

    model_config = ConfigDict(extra="forbid")

    benchmark_item_id: NonEmptyStr
    text_id: NonEmptyStr
    student_knowledge_id: NonEmptyStr
    language_iso_code: NonEmptyStr

    content: NonEmptyStr
    student_knowledge: dict[str, Any]

    nondecodable_words: list[str]
    num_nondecodable_words: int = Field(ge=0)
    num_total_words: int = Field(gt=0)
    nondecodable_ratio: float = Field(ge=0.0, le=1.0)

    jagged: bool | None = None

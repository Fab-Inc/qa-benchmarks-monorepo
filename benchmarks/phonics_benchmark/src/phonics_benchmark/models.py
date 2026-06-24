"""Data models for phonics_benchmark."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

NonEmptyStr = Annotated[str, Field(min_length=1)]


class StudentKnowledgeProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")
    student_knowledge_id: NonEmptyStr
    language_iso_code: NonEmptyStr
    # Schema is intentionally open — taught_knowledge is still evolving.
    taught_knowledge: dict[str, list[str] | int]


class DecodableText(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text_id: NonEmptyStr
    language_iso_code: NonEmptyStr
    content: NonEmptyStr


class BenchmarkItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    benchmark_item_id: NonEmptyStr
    text_id: NonEmptyStr
    student_knowledge_id: NonEmptyStr

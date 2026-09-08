"""Data models for phonics_benchmark."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

NonEmptyStr = Annotated[str, Field(min_length=1)]

# Student profiles are bucketed by how much of the grapheme inventory they have
# been taught. The benchmark reports scores per bucket because the amount of
# writable vocabulary varies by more than an order of magnitude across them.
DifficultyBucket = Literal["very-early", "early", "mid", "late"]


class StudentKnowledge(BaseModel):
    model_config = ConfigDict(extra="forbid")
    student_knowledge_id: NonEmptyStr
    language_iso_code: NonEmptyStr
    student_knowledge: dict[str, Any]


class GenerationBenchmarkItem(BaseModel):
    """One (scenario x student profile) pairing the model must write text for.

    Unlike the old discrimination benchmark there is no reference answer: the
    deterministic `decodability` checker grades whatever the model writes.
    """

    model_config = ConfigDict(extra="forbid")

    benchmark_item_id: NonEmptyStr
    scenario_id: NonEmptyStr
    scenario: NonEmptyStr
    student_knowledge_id: NonEmptyStr
    language_iso_code: NonEmptyStr
    student_knowledge: dict[str, Any]
    difficulty_bucket: DifficultyBucket
    n_graphemes_known: int = Field(ge=0)
    n_clusters_known: int = Field(ge=0)
    n_cluster_patterns_known: int = Field(ge=0)

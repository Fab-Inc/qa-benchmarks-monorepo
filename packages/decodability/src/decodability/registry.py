"""Per-language decodability object registry.

Each language subpackage exposes the same surface (a student-knowledge model,
scoring methods, aggregations, and a word extractor). This module bundles those
behind a single lookup keyed by ISO 639 code, so callers can stay
language-agnostic — adding a language is a one-line entry here once its
subpackage exists.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import cache

from pydantic import BaseModel

from decodability.annotate import Aggregation, Measure


@dataclass(frozen=True)
class DecodabilityObjects:
    """The decodability tooling for one language."""

    name: str  # human-readable, e.g. "Kiswahili"
    student_knowledge_model: type[BaseModel]
    scoring_methods: dict[str, Measure]
    aggregations: dict[str, Aggregation]
    extract_words: Callable[[str], list[str]]


@cache
def _registry() -> dict[str, DecodabilityObjects]:
    """Build the language registry on first use.

    The kiswahili subpackage imports back from the top-level ``decodability``
    package, so it must not be imported while ``decodability`` is still
    initializing. Importing it lazily here defers that dependency to first call
    and avoids a circular import.
    """
    from decodability.kiswahili import (
        AGGREGATIONS,
        SCORING_METHODS,
        KiswahiliStudentKnowledge,
        extract_words_kiswahili,
    )

    return {
        "sw": DecodabilityObjects(
            name="Kiswahili",
            student_knowledge_model=KiswahiliStudentKnowledge,
            scoring_methods=SCORING_METHODS,
            aggregations=AGGREGATIONS,
            extract_words=extract_words_kiswahili,
        ),
    }


def get_decodability_objects(language: str) -> DecodabilityObjects:
    """Return the decodability objects for an ISO 639 language code.

    Raises KeyError listing the supported languages if unknown.
    """
    registry = _registry()
    try:
        return registry[language]
    except KeyError:
        supported = ", ".join(sorted(registry)) or "(none)"
        raise KeyError(
            f"No decodability objects registered for language '{language}'. "
            f"Supported languages: {supported}"
        ) from None

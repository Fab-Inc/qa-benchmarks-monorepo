"""Shared rendering of student-knowledge profiles into prompt text.

A profile renders in one of two notations:

- ``"quoted"``: every item in double quotes, GPCs as ``"a" → "æ"``. Every phonics
  (decodability assessment) benchmark run used this notation, so it is frozen:
  changing it changes those prompts and their ``model_call_id`` hashes.
- ``"plain"``: unquoted items, GPCs as ``a -> /æ/``. Used by the decodable text
  generation benchmark for English.

Every language supports both formats; each caller chooses one explicitly.

``str()`` of a profile uses ``"quoted"``. Code that builds prompts calls ``render()``
with an explicit format rather than relying on that default.
"""

from collections.abc import Iterable
from typing import Literal, get_args

from pydantic import BaseModel

KnowledgeFormat = Literal["quoted", "plain"]
KNOWLEDGE_FORMATS: tuple[str, ...] = get_args(KnowledgeFormat)
DEFAULT_KNOWLEDGE_FORMAT: KnowledgeFormat = "quoted"


def check_knowledge_format(fmt: str) -> None:
    """Raise ValueError unless ``fmt`` is a known knowledge format."""
    if fmt not in KNOWLEDGE_FORMATS:
        raise ValueError(
            f"Unknown knowledge format {fmt!r}; expected one of {KNOWLEDGE_FORMATS}"
        )


def format_items(items: Iterable[str], fmt: KnowledgeFormat) -> str:
    """Render items sorted and comma-separated, or ``None`` when there are none."""
    check_knowledge_format(fmt)
    ordered = sorted(items)
    if fmt == "quoted":
        ordered = [f'"{item}"' for item in ordered]
    return ", ".join(ordered) or "None"


def format_pairs(pairs: Iterable[tuple[str, str]], fmt: KnowledgeFormat) -> str:
    """Render grapheme-phoneme pairs in their given order, or ``None`` if empty."""
    check_knowledge_format(fmt)
    if fmt == "quoted":
        rendered = "; ".join(f'"{grapheme}" → "{ipa}"' for grapheme, ipa in pairs)
    else:
        rendered = "; ".join(f"{grapheme} -> /{ipa}/" for grapheme, ipa in pairs)
    return rendered or "None"


class StudentKnowledgeBase(BaseModel):
    """Base for the language-specific student-knowledge models.

    Subclasses implement ``_render``; callers use ``render`` so every language
    accepts the same formats.
    """

    def render(self, fmt: KnowledgeFormat) -> str:
        """Render the profile as the STUDENT KNOWLEDGE block of a prompt."""
        check_knowledge_format(fmt)
        return self._render(fmt)

    def _render(self, fmt: KnowledgeFormat) -> str:
        raise NotImplementedError

    def __str__(self) -> str:
        """Render in the default ``"quoted"`` notation."""
        return self.render(DEFAULT_KNOWLEDGE_FORMAT)

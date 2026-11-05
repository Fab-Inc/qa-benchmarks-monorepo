from pydantic import field_validator

from decodability.english.definitions import (
    UFLI_GPC_PAIRS_FLAT,
    normalize_gpc_grapheme,
)
from decodability.student_knowledge import (
    KnowledgeFormat,
    StudentKnowledgeBase,
    format_items,
    format_pairs,
)


class EnglishStudentKnowledge(StudentKnowledgeBase):
    gpcs: list[tuple[str, str]] = []
    sight_words: set[str] = set()
    # Marker-bearing units taught as chunks: affixes ("-ed", "-ies", "un-")
    # and positional graphemes ("_ce", "-ve", "-le"). The workbook writes the
    # marker at the edge of the grapheme; ingestion preserves it here while
    # the same pair also lands in ``gpcs`` stripped, as letter knowledge.
    affixes: set[str] = set()

    @field_validator("gpcs")
    @classmethod
    def check_known_gpcs(
        cls,
        gpcs: list[tuple[str, str]],
    ) -> list[tuple[str, str]]:
        """Check that every GPC is taught in the UFLI inventory.

        Graphemes are compared in canonical form, so a pair carries the same
        meaning whether the profile writes the affix marker (``-ed``,
        ``pre-``, ``_ce``) or not (``ed``, ``pre``, ``ce``).
        """
        invalid = [
            g
            for g in gpcs
            if (normalize_gpc_grapheme(g[0]), g[1]) not in UFLI_GPC_PAIRS_FLAT
        ]
        if invalid:
            raise ValueError(
                f"Not taught GPCs: {invalid}. "
                "Only GPCs from UFLI GPC inventory are allowed."
            )
        return gpcs

    @field_validator("sight_words")
    @classmethod
    def lowercase_sight_words(cls, sight_words: set[str]) -> set[str]:
        """Lowercase all sight words to ensure consistency."""
        return {word.lower() for word in sight_words}

    @field_validator("affixes")
    @classmethod
    def lowercase_affixes(cls, affixes: set[str]) -> set[str]:
        """Lowercase all affixes to ensure consistency."""
        return {affix.lower() for affix in affixes}

    def _render(self, fmt: KnowledgeFormat) -> str:
        """Render known GPCs and sight words; ``affixes`` are not rendered."""
        return "\n".join(
            [
                f"- Known GPCs: {format_pairs(self.gpcs, fmt)}",
                f"- Known sight words: {format_items(self.sight_words, fmt)}",
            ]
        )

from pydantic import BaseModel, field_validator

from decodability.kiswahili.config import VALID_GRAPHEMES


class KiswahiliStudentKnowledge(BaseModel):
    """Model for Kiswahili student knowledge representation."""

    graphemes: set[str] = set()
    whole_words: set[str] = set()
    syllables: set[str] = set()
    max_word_length: int = 5
    syllable_patterns: set[str] = set()

    @field_validator("graphemes")
    def check_known_graphemes(cls, graphemes: set[str]) -> set[str]:  # noqa: N805
        """Check that every grapheme is a valid Kiswahili grapheme."""
        invalid_graphemes = [
            grapheme
            for grapheme in graphemes
            if grapheme.lower() not in VALID_GRAPHEMES
        ]

        if invalid_graphemes:
            raise ValueError(
                f"Invalid graphemes: {invalid_graphemes}. "
                "Only valid Kiswahili graphemes are allowed."
            )

        return graphemes

    @field_validator("syllable_patterns")
    def check_cv_pattern(cls, patterns: set[str]) -> bool | set[str]:  # noqa: N805
        """Check if a given pattern is a valid CV pattern."""
        invalid_patterns = []
        for pattern in patterns:
            if not all(char in {"C", "V"} for char in pattern):
                invalid_patterns.append(pattern)

        if invalid_patterns:
            raise ValueError(
                f"Invalid CV patterns: {invalid_patterns}. "
                "Only 'C' and 'V' are allowed."
            )

        return patterns

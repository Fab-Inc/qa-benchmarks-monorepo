from pydantic import BaseModel, field_validator

from decodability.kiswahili.config import VALID_GRAPHEMES


class KiswahiliStudentKnowledge(BaseModel):
    """Model for Kiswahili student knowledge representation."""

    graphemes: set[str] = set()
    whole_words: set[str] = set()

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

    @field_validator("whole_words")
    def lowercase_whole_words(cls, whole_words: set[str]) -> set[str]:  # noqa: N805
        """Lowercase all whole words to ensure consistency."""
        return {word.lower() for word in whole_words}

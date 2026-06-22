from pydantic import BaseModel, field_validator


class KiswahiliStudentKnowledge(BaseModel):
    """Model for Kiswahili student knowledge representation."""

    known_syllables: set[str] = set()
    max_word_length: int = 1
    known_syllable_patterns: set[str] = set()

    @field_validator("known_syllable_patterns")
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

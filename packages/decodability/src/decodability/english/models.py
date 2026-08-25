from pydantic import BaseModel, field_validator

from decodability.english.definitions import UFLI_GPC_PAIRS_FLAT


class EnglishStudentKnowledge(BaseModel):
    gpcs: list[tuple[str, str]] = []
    sight_words: set[str] = set()

    @field_validator("gpcs")
    @classmethod
    def check_known_gpcs(
        cls,
        gpcs: list[tuple[str, str]],
    ) -> list[tuple[str, str]]:
        """Check that every GPC is taught in the UFLI inventory."""
        invalid = [g for g in gpcs if tuple(g) not in UFLI_GPC_PAIRS_FLAT]
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

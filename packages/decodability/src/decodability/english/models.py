from pydantic import BaseModel, field_validator

from decodability.english.definitions import UPL_GPC_PAIRS


class EnglishStudentKnowledge(BaseModel):
    gpcs: list[tuple[str, tuple[str, ...]]] = []
    whole_words: set[str] = set()

    @field_validator("gpcs")
    @classmethod
    def check_known_gpcs(
        cls,
        gpcs: list[tuple[str, tuple[str, ...]]],
    ) -> list[tuple[str, tuple[str, ...]]]:
        """Check that every GPC is taught in the UPL inventory."""
        invalid = [g for g in gpcs if tuple(g) not in UPL_GPC_PAIRS]
        if invalid:
            raise ValueError(
                f"Not taught GPCs: {invalid}. "
                "Only GPCs from UPL_GPC_INVENTORY are allowed."
            )
        return gpcs

    @field_validator("whole_words")
    @classmethod
    def lowercase_whole_words(cls, whole_words: set[str]) -> set[str]:
        """Lowercase all whole words to ensure consistency."""
        return {word.lower() for word in whole_words}

    def __str__(self) -> str:
        """Return a string representation of the EnglishStudentKnowledge instance."""
        gpcs = ", ".join(f'"{g}"' for g in sorted(self.gpcs))
        whole_words = ", ".join(f'"{w}"' for w in sorted(self.whole_words))

        if not self.gpcs:
            gpcs = "None"

        if not self.whole_words:
            whole_words = "None"

        return f"- Known GPCs: {gpcs}\n- Known whole words: {whole_words}"

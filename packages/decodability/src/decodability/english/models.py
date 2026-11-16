from typing import Literal, Self, get_args

from pydantic import (
    BaseModel,
    PrivateAttr,
    ValidationInfo,
    field_validator,
    model_validator,
)

from decodability.english.definitions import (
    GPC,
    UFLI_GPC_INVENTORY,
    gpc_pairs,
    normalize_gpc_grapheme,
)

EnglishKnowledgeFormat = Literal["quoted", "plain"]


def _context_inventory(info: ValidationInfo) -> tuple[GPC, ...]:
    """The inventory a caller passed as ``context={"gpc_inventory": ...}``."""
    return (info.context or {}).get("gpc_inventory", UFLI_GPC_INVENTORY)


class EnglishStudentKnowledge(BaseModel):
    gpcs: list[tuple[str, str]] = []
    sight_words: set[str] = set()
    # Marker-bearing units taught as chunks: affixes ("-ed", "-ies", "un-")
    # and positional graphemes ("_ce", "-ve", "-le"). The workbook writes the
    # marker at the edge of the grapheme; ingestion preserves it here while
    # the same pair also lands in ``gpcs`` stripped, as letter knowledge.
    affixes: set[str] = set()
    # The inventory ``gpcs`` were checked against and scorers parse words with:
    # UFLI_GPC_INVENTORY unless the caller validates with
    # ``context={"gpc_inventory": ...}``, such as a benchmark's copy of an
    # earlier inventory. Private, so it is not part of the profile or its dumps.
    _gpc_inventory: tuple[GPC, ...] = PrivateAttr(default=UFLI_GPC_INVENTORY)

    @field_validator("gpcs")
    @classmethod
    def check_known_gpcs(
        cls,
        gpcs: list[tuple[str, str]],
        info: ValidationInfo,
    ) -> list[tuple[str, str]]:
        """Check that every GPC is taught in the inventory.

        Graphemes are compared in canonical form, so a pair carries the same
        meaning whether the profile writes the affix marker (``-ed``,
        ``pre-``, ``_ce``) or not (``ed``, ``pre``, ``ce``).
        """
        taught = gpc_pairs(_context_inventory(info))
        invalid = [
            g for g in gpcs if (normalize_gpc_grapheme(g[0]), g[1]) not in taught
        ]
        if invalid:
            raise ValueError(
                f"Not taught GPCs: {invalid}. "
                "Only GPCs from UFLI GPC inventory are allowed."
            )
        return gpcs

    @model_validator(mode="after")
    def keep_gpc_inventory(self, info: ValidationInfo) -> Self:
        self._gpc_inventory = _context_inventory(info)
        return self

    @property
    def gpc_inventory(self) -> tuple[GPC, ...]:
        """The inventory ``gpcs`` were checked against; scorers parse with it."""
        return self._gpc_inventory

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

    def render(self, fmt: EnglishKnowledgeFormat) -> str:
        """Render known GPCs and sight words as a prompt's STUDENT KNOWLEDGE block.

        - ``"quoted"``: ``"a" → "æ"``, ``"the"``
        - ``"plain"``: ``a -> /æ/``, ``the``

        GPCs keep their order, sight words are sorted, and ``affixes`` are not
        rendered.
        """
        if fmt == "quoted":
            gpcs = "; ".join(f'"{grapheme}" → "{ipa}"' for grapheme, ipa in self.gpcs)
            sight_words = ", ".join(f'"{w}"' for w in sorted(self.sight_words))
        elif fmt == "plain":
            gpcs = "; ".join(f"{grapheme} -> /{ipa}/" for grapheme, ipa in self.gpcs)
            sight_words = ", ".join(sorted(self.sight_words))
        else:
            raise ValueError(
                f"Unknown knowledge format {fmt!r}; English supports "
                f"{get_args(EnglishKnowledgeFormat)}"
            )

        return "\n".join(
            [
                f"- Known GPCs: {gpcs or 'None'}",
                f"- Known sight words: {sight_words or 'None'}",
            ]
        )

    def __str__(self) -> str:
        return self.render("quoted")

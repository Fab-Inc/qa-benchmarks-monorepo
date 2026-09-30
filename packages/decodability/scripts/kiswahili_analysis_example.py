"""Small runnable Kiswahili example for per-word decodability analysis.

Shows the analysis API: each word gets a record of what blocks it, plus a verdict.
Run from the repository root with:

    uv run --frozen python examples/kiswahili_analysis_example.py
"""

from __future__ import annotations

from decodability.kiswahili import (
    KiswahiliStudentKnowledge,
    KiswahiliWordAnalysis,
    analyse_word_kiswahili,
    assess_decodability_kenya_tusome,
    extract_words_kiswahili,
)

TEXT = """Tulisikia mayowe tukiwa sebuleni. Sote tukatoka.
Kutazama tukaona moto ukiwaka sana. Waya wa stima uliwaka moto.
Sote tukawa na wasiwasi sana. Wazima moto wakatokea.
Mioyo yetu ikatulia, tukaelekea sebuleni tena."""

STUDENT_KNOWLEDGE = KiswahiliStudentKnowledge.model_validate(
    dict(
        graphemes={
            "a",
            "u",
            "i",
            "o",
            "e",
            "m",
            "k",
            "t",
            "n",
            "s",
            "b",
            "y",
            "z",
            "g",
            "d",
            "j",
            "r",
            "w",
        },
        clusters={"kw", "st"},
        cluster_patterns={"prenasalised"},
        whole_words={"mama", "ama", "anaona", "na", "paka"},
    )
)


def get_nondecodable_word_analyses() -> list[KiswahiliWordAnalysis]:
    """Return the analyses of all non-decodable words in the text."""
    words = extract_words_kiswahili(TEXT)
    analyses = {word: analyse_word_kiswahili(word, STUDENT_KNOWLEDGE) for word in words}
    return [
        analyses[word]
        for word in analyses
        if not assess_decodability_kenya_tusome(analyses[word])
    ]


def main() -> None:
    nondecodable_word_analyses = get_nondecodable_word_analyses()
    for analysis in nondecodable_word_analyses:
        print(analysis.model_dump_json(indent=2), "\n")


if __name__ == "__main__":
    main()

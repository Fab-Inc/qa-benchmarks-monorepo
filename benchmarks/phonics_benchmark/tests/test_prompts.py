"""Tests for loading and rendering packaged prompts."""

import pytest
from decodability import DECODABILITY_REGISTRY
from decodability.english import EnglishStudentKnowledge
from decodability.kiswahili import KiswahiliStudentKnowledge
from phonics_benchmark.config import DEFAULT_PROMPT_NAME, ConfigError
from phonics_benchmark.prompts import (
    KNOWLEDGE_FORMAT,
    PROMPT_FIELD_BUILDERS,
    _load_prompts,
    build_decodability_prompt,
    get_prompt,
)

ITEM = {
    "benchmark_item_id": "item-1",
    "content": "Mama anaona kuku.",
    "student_knowledge": {"graphemes": ["a", "m", "n"], "whole_words": ["mama"]},
}

EN_ITEM = {
    "benchmark_item_id": "item-en",
    "content": "The cat sat.",
    "student_knowledge": {
        "gpcs": [["c", "k"], ["a", "æ"], ["t", "t"], ["s", "s"]],
        "sight_words": ["the"],
    },
}

SW_CLUSTER_ITEM = {
    "benchmark_item_id": "item-sw-clusters",
    "content": "Mbuzi.",
    "student_knowledge": {
        "graphemes": ["m", "b", "d"],
        "clusters": ["mb", "md"],
        "cluster_patterns": ["palatalised_glide"],
    },
}


@pytest.mark.parametrize(
    ("prompt_name", "item", "language", "expected_block"),
    [
        (
            "decodability_evaluation_schwa_rule",
            EN_ITEM,
            "en",
            '- Known GPCs: "c" → "k"; "a" → "æ"; "t" → "t"; "s" → "s"\n'
            '- Known sight words: "the"',
        ),
        (
            "decodability_evaluation_clusters",
            SW_CLUSTER_ITEM,
            "sw",
            '- Known graphemes: "b", "d", "m"\n'
            '- Known clusters: "mb", "md"\n'
            '- Known cluster patterns: "consonant + "y""\n'
            "- Known whole words: None",
        ),
    ],
)
def test_leaderboard_prompts_keep_the_frozen_quoted_notation(
    prompt_name: str, item: dict, language: str, expected_block: str
) -> None:
    """Every phonics run used this notation; changing it splits results."""
    prompt = build_decodability_prompt(prompt_name, item, language)

    assert expected_block in prompt


def test_leaderboard_prompts_render_profiles_under_the_frozen_inventory() -> None:
    """Stored l128 profiles spell uni- as /juni/, which the latest inventory rejects."""
    item = {
        "benchmark_item_id": "item-en-l128",
        "content": "A unit.",
        "student_knowledge": {"gpcs": [["a", "æ"], ["uni", "juni"], ["t", "t"]]},
    }

    prompt = build_decodability_prompt("decodability_evaluation_schwa_rule", item, "en")

    assert '- Known GPCs: "a" → "æ"; "uni" → "juni"; "t" → "t"' in prompt


def test_phonics_prompts_do_not_depend_on_str(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prompts render KNOWLEDGE_FORMAT explicitly, so str() changes can't reach them."""

    def fail(self: object) -> str:
        raise AssertionError("phonics prompts must not use str() of the profile")

    monkeypatch.setattr(EnglishStudentKnowledge, "__str__", fail)
    monkeypatch.setattr(KiswahiliStudentKnowledge, "__str__", fail)

    en_prompt = build_decodability_prompt(
        "decodability_evaluation_schwa_rule", EN_ITEM, "en"
    )
    sw_prompt = build_decodability_prompt(DEFAULT_PROMPT_NAME, ITEM, "sw")

    assert '- Known sight words: "the"' in en_prompt
    assert '- Known whole words: "mama"' in sw_prompt


def test_get_prompt_formats_student_knowledge_with_str() -> None:
    student_knowledge = KiswahiliStudentKnowledge(
        graphemes={"a", "m", "n"},
        whole_words={"mama"},
    )

    prompt = get_prompt(
        "decodability_evaluation",
        language="Kiswahili",
        text="Mama anaona kuku.",
        student_knowledge=student_knowledge,
    )

    assert "language of Kiswahili" in prompt
    assert str(student_knowledge) in prompt
    assert "Mama anaona kuku." in prompt


def test_get_prompt_rejects_unknown_prompt_name() -> None:
    with pytest.raises(KeyError) as error:
        get_prompt("not-a-real-prompt")

    assert "Unknown prompt 'not-a-real-prompt'" in str(error.value)
    assert "decodability_evaluation" in str(error.value)


def test_build_decodability_prompt_uses_registered_language_integration() -> None:
    prompt = build_decodability_prompt(
        DEFAULT_PROMPT_NAME,
        {
            "benchmark_item_id": "item-1",
            "content": "Mama anaona kuku.",
            "student_knowledge": {
                "graphemes": ["a", "m", "n"],
                "whole_words": ["mama"],
            },
        },
        "sw",
    )

    assert "language of Kiswahili" in prompt
    assert '- Known graphemes: "a", "m", "n"' in prompt
    assert '- Known whole words: "mama"' in prompt
    assert "Mama anaona kuku." in prompt


def test_build_decodability_prompt_rejects_unknown_language() -> None:
    with pytest.raises(ConfigError, match="Unknown language registry key 'xx'"):
        build_decodability_prompt(
            DEFAULT_PROMPT_NAME,
            {
                "benchmark_item_id": "item-1",
                "content": "Text.",
                "student_knowledge": {},
            },
            "xx",
        )


@pytest.mark.parametrize("language", sorted(KNOWLEDGE_FORMAT))
def test_knowledge_format_is_supported_by_the_language_model(language: str) -> None:
    model = DECODABILITY_REGISTRY[language]["student_knowledge_model"]
    assert model().render(KNOWLEDGE_FORMAT[language]).startswith("- Known ")


def test_build_decodability_prompt_requires_a_knowledge_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(KNOWLEDGE_FORMAT, "sw")
    with pytest.raises(
        ConfigError, match="No student-knowledge format for language 'sw'"
    ):
        build_decodability_prompt(DEFAULT_PROMPT_NAME, ITEM, "sw")


def test_build_decodability_prompt_reports_invalid_student_knowledge() -> None:
    with pytest.raises(ConfigError, match="student_knowledge.*item-1"):
        build_decodability_prompt(
            DEFAULT_PROMPT_NAME,
            {
                "benchmark_item_id": "item-1",
                "content": "Text.",
                "student_knowledge": {"graphemes": ["not-valid"]},
            },
            "sw",
        )


def test_every_prompt_template_has_a_field_builder() -> None:
    """The builder map and prompts.yaml must not drift apart.

    A template with no builder is unreachable; a builder with no template raises
    at render time. Either way the mismatch should fail here, not in a run.
    """
    assert set(PROMPT_FIELD_BUILDERS) == set(_load_prompts())


@pytest.mark.parametrize("prompt_name", sorted(PROMPT_FIELD_BUILDERS))
def test_every_prompt_renders_with_its_own_builder(prompt_name: str) -> None:
    """Each template's declared fields are all supplied by its builder."""
    prompt = build_decodability_prompt(prompt_name, ITEM, "sw")

    assert "language of Kiswahili" in prompt
    assert '- Known graphemes: "a", "m", "n"' in prompt
    # No placeholder was left unrendered.
    assert "{" not in prompt


def test_grapheme_breakdown_prompt_gets_the_broken_down_text() -> None:
    """The variant must receive slash-separated graphemes, not the raw text.

    Its template claims "slashes (/) have been added", and it declares the same
    field names as the baseline, so passing raw content would render cleanly
    while telling the model something untrue about its own input.
    """
    prompt = build_decodability_prompt(
        "decodability_evaluation_grapheme_breakdown", ITEM, "sw"
    )

    assert "m/a/m/a" in prompt
    assert "Mama anaona kuku." not in prompt


def test_valid_graphemes_prompt_includes_the_inventory() -> None:
    """The variant hands the model the language's full grapheme inventory."""
    prompt = build_decodability_prompt(
        "decodability_evaluation_valid_graphemes", ITEM, "sw"
    )

    assert "- Vowels:" in prompt
    assert "- Consonants:" in prompt
    # The raw text is still the text for this variant.
    assert "Mama anaona kuku." in prompt


def test_build_decodability_prompt_rejects_unknown_prompt_name() -> None:
    with pytest.raises(ConfigError) as error:
        build_decodability_prompt("not-a-real-prompt", ITEM, "sw")

    assert "Unknown prompt 'not-a-real-prompt'" in str(error.value)
    assert DEFAULT_PROMPT_NAME in str(error.value)

"""Tests for loading and rendering packaged prompts."""

import pytest
from decodability.kiswahili import KiswahiliStudentKnowledge
from phonics_benchmark.config import DEFAULT_PROMPT_NAME, ConfigError
from phonics_benchmark.prompts import (
    PROMPT_FIELD_BUILDERS,
    _load_prompts,
    build_decodability_prompt,
    get_prompt,
)

ITEM = {
    "benchmark_item_id": "item-1",
    "content": "Mama anaona kuku.",
    "student_knowledge": {"graphemes": ["a", "m", "n"], "whole_words": ["MAMA"]},
}


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
                "whole_words": ["MAMA"],
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

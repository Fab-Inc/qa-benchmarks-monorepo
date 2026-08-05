"""Tests for loading and rendering packaged prompts."""

import pytest
from decodability.kiswahili import KiswahiliStudentKnowledge
from phonics_benchmark.config import ConfigError
from phonics_benchmark.prompts import build_decodability_prompt, get_prompt


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
            {
                "benchmark_item_id": "item-1",
                "content": "Text.",
                "student_knowledge": {"graphemes": ["not-valid"]},
            },
            "sw",
        )

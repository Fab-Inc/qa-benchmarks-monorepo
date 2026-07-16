"""Tests for loading and rendering packaged prompts."""

import pytest
from phonics_benchmark.prompts import build_decodability_prompt, get_prompt


def test_get_prompt_renders_generated_tusome_item(tusome_stage_11_item):
    prompt = get_prompt(
        "decodability_evaluation",
        language="kiswahili",
        text=tusome_stage_11_item["content"],
        known_graphemes=tusome_stage_11_item["graphemes"],
        known_sight_words=tusome_stage_11_item["whole_words"],
    )

    assert "language of kiswahili" in prompt
    assert tusome_stage_11_item["content"] in prompt
    assert tusome_stage_11_item["graphemes"] in prompt
    assert tusome_stage_11_item["whole_words"] in prompt
    assert "ng'ombe" in prompt


def test_get_prompt_rejects_unknown_prompt_name():
    with pytest.raises(KeyError) as error:
        get_prompt("not-a-real-prompt")

    assert "Unknown prompt 'not-a-real-prompt'" in str(error.value)
    assert "decodability_evaluation" in str(error.value)


def test_build_decodability_prompt_adapts_joined_benchmark_item() -> None:
    """The production adapter renders the QFB prompt from joined item fields."""

    prompt = build_decodability_prompt(
        {
            "text": "Mama anaona kuku.",
            "student_knowledge": {
                "graphemes": ["a", "m", "n"],
                "whole_words": ["mama"],
            },
        },
        "kiswahili",
    )

    assert "language of kiswahili" in prompt
    assert 'graphemes the student knows are:\n["a", "m", "n"]' in prompt
    assert 'sight words the student knows are:\n["mama"]' in prompt
    assert "Mama anaona kuku." in prompt

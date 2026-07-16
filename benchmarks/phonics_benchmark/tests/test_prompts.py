"""Tests for loading and rendering packaged prompts."""

import pytest
from phonics_benchmark.prompts import get_prompt


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

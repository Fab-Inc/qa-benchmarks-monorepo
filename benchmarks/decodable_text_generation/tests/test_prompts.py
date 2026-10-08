"""Tests for generation prompt rendering."""

from typing import Any

import pytest
from decodable_text_generation.prompts import (
    build_generation_prompt,
    get_decoding_rules,
    get_scenario,
    load_scenarios,
    render_target,
)
from phonics_benchmark.config import ConfigError

EN_ITEM: dict[str, Any] = {
    "benchmark_item_id": "p1_words",
    "student_knowledge_id": "p1",
    "scenario_id": "words",
    "language_iso_code": "en",
    "jagged": False,
    "student_knowledge": {
        "gpcs": [["a", "æ"], ["m", "m"], ["t", "t"]],
        "sight_words": ["the"],
    },
}

SW_ITEM: dict[str, Any] = {
    "benchmark_item_id": "p2_story",
    "student_knowledge_id": "p2",
    "scenario_id": "story",
    "language_iso_code": "sw",
    "jagged": False,
    "student_knowledge": {
        "graphemes": ["a", "m", "t"],
        "clusters": [],
        "cluster_patterns": [],
        "whole_words": ["mama"],
    },
}


def test_child_prompt_renders_all_sections() -> None:
    prompt = build_generation_prompt("decodable_text_generation", EN_ITEM, "en")

    assert "DECODING RULES" in prompt
    assert "sight word" in prompt  # en decoding rules wording
    assert "English" in prompt
    assert "- Known GPCs: a -> /æ/" in prompt
    assert "- Known sight words: the" in prompt
    assert "Generate a list of 20 words" in prompt
    assert "<text>" in prompt
    assert "TARGET" not in prompt


def test_prompt_surfaces_known_affixes() -> None:
    """Marked gpc graphemes land in the prompt's affixes line."""
    item = {
        **EN_ITEM,
        "student_knowledge": {
            "gpcs": [["a", "æ"], ["-ed", "t"], ["un-", "ʌn"]],
            "sight_words": ["the"],
            "affixes": ["-ed", "un-"],
        },
    }
    prompt = build_generation_prompt("decodable_text_generation", item, "en")

    assert "- Known affixes: -ed, un-" in prompt
    assert "ed -> /t/" in prompt  # stripped pair still listed as letter knowledge


def test_sw_prompt_uses_swahili_rules_and_knowledge() -> None:
    prompt = build_generation_prompt("decodable_text_generation", SW_ITEM, "sw")

    assert "whole words" in prompt  # sw decoding rules wording
    assert '- Known whole words: "mama"' in prompt
    assert "Kiswahili" in prompt


def test_curriculum_prompt_renders_target() -> None:
    item = {
        **EN_ITEM,
        "target_knowledge": {"gpcs": [["n", "n"]], "sight_words": ["was"]},
    }
    prompt = build_generation_prompt("decodable_text_generation_curriculum", item, "en")

    assert "TARGET" in prompt
    assert "- Target GPCs: n -> /n/" in prompt
    assert "- Target sight words: was" in prompt


def test_curriculum_prompt_without_target_fails() -> None:
    with pytest.raises(ConfigError, match="target"):
        build_generation_prompt("decodable_text_generation_curriculum", EN_ITEM, "en")


def test_unknown_prompt_name_fails() -> None:
    with pytest.raises(ConfigError, match="Unknown prompt"):
        build_generation_prompt("not_a_prompt", EN_ITEM, "en")


def test_unknown_scenario_fails() -> None:
    item = {**EN_ITEM, "scenario_id": "poem"}
    with pytest.raises(ConfigError, match="Unknown scenario_id 'poem'"):
        build_generation_prompt("decodable_text_generation", item, "en")


def test_unknown_language_fails() -> None:
    with pytest.raises(ConfigError, match="Unknown language registry key"):
        build_generation_prompt("decodable_text_generation", EN_ITEM, "xx")


def test_invalid_student_knowledge_fails() -> None:
    item = {
        **EN_ITEM,
        "student_knowledge": {"gpcs": [["zz", "zz"]], "sight_words": []},
    }
    with pytest.raises(ConfigError, match="Invalid student_knowledge"):
        build_generation_prompt("decodable_text_generation", item, "en")


def test_get_scenario_and_load_scenarios() -> None:
    scenarios = load_scenarios()
    assert set(scenarios) == {"words", "sentences", "story"}
    assert get_scenario("sentences").startswith("Write 10")


def test_get_decoding_rules_per_language() -> None:
    en = get_decoding_rules("en")
    sw = get_decoding_rules("sw")
    assert en["word_list_name"] == "sight words"
    assert sw["word_list_name"] == "whole words"
    assert "GPC" in en["decoding_rules"]
    assert "grapheme" in sw["decoding_rules"]


def test_get_decoding_rules_unknown_language() -> None:
    with pytest.raises(ConfigError, match="No decoding rules"):
        get_decoding_rules("xx")


def test_render_target_formats_pairs_and_words() -> None:
    rendered = render_target({"gpcs": [["n", "n"]], "sight_words": ["was"]})
    assert "- Target GPCs: n -> /n/" in rendered
    assert "- Target sight words: was" in rendered

    assert render_target({"graphemes": ["b", "a"]}) == "- Target graphemes: a, b"

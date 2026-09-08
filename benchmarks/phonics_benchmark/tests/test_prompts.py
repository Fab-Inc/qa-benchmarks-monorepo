"""Tests for scenario/prompt loading and prompt rendering."""

from typing import Any

import pytest
from phonics_benchmark.config import ConfigError
from phonics_benchmark.prompts import (
    _load_prompts,
    build_generation_prompt,
    get_prompt,
    get_scenario,
    get_scenarios,
)

PROMPT_NAME = "decodable_text_generation"


@pytest.fixture
def item(tusome_stage_11_item: dict[str, Any]) -> dict[str, Any]:
    return tusome_stage_11_item


class TestScenarios:
    """The packaged scenarios.yaml."""

    def test_every_expected_scenario_is_present(self) -> None:
        """The four scenarios the item builder crosses profiles with."""
        assert set(get_scenarios()) == {
            "story",
            "informational",
            "dialogue",
            "sentences",
        }

    def test_every_scenario_states_a_length_target(self) -> None:
        """Each scenario defines its own length so the task is comparable."""
        assert all("50 words" in text for text in get_scenarios().values())

    def test_scenarios_are_non_empty_prose(self) -> None:
        """A blank scenario would render an empty TASK block."""
        assert all(text.strip() for text in get_scenarios().values())

    def test_returns_a_copy(self) -> None:
        """Callers cannot mutate the cached scenario mapping."""
        get_scenarios()["story"] = "clobbered"
        assert get_scenarios()["story"] != "clobbered"

    def test_get_scenario_returns_one(self) -> None:
        """A single scenario can be looked up by id."""
        assert get_scenario("story") == get_scenarios()["story"]

    def test_unknown_scenario_lists_the_available_ones(self) -> None:
        """The error message tells the caller what it could have asked for."""
        with pytest.raises(ConfigError, match="Unknown scenario 'nope'"):
            get_scenario("nope")


class TestGetPrompt:
    """Raw template loading and formatting."""

    def test_renders_named_placeholders(self) -> None:
        """Templates are formatted with str.format named fields."""
        rendered = get_prompt(
            PROMPT_NAME,
            language="Kiswahili",
            student_knowledge='- Known graphemes: "a"',
            scenario="Write a story.",
        )
        assert "Kiswahili" in rendered
        assert "Write a story." in rendered

    def test_unknown_prompt_lists_the_available_ones(self) -> None:
        """A typo'd prompt name names what exists instead."""
        with pytest.raises(KeyError, match="Unknown prompt 'nope'"):
            get_prompt("nope")


class TestBuildGenerationPrompt:
    """Rendering one benchmark item into its prompt."""

    def test_renders_scenario_and_student_knowledge(self, item: dict[str, Any]) -> None:
        """Both injected blocks reach the prompt."""
        prompt = build_generation_prompt(PROMPT_NAME, item, "sw")
        assert item["scenario"] in prompt
        assert "Known graphemes" in prompt

    def test_uses_the_language_display_name(self, item: dict[str, Any]) -> None:
        """The registry supplies the prose language name, not the ISO code."""
        prompt = build_generation_prompt(PROMPT_NAME, item, "sw")
        assert "language of Kiswahili" in prompt

    def test_student_knowledge_is_rendered_by_the_language_model(
        self, item: dict[str, Any]
    ) -> None:
        """The pydantic model's __str__ *is* the STUDENT KNOWLEDGE block."""
        prompt = build_generation_prompt(PROMPT_NAME, item, "sw")
        for line in ("Known graphemes:", "Known clusters:", "Known whole words:"):
            assert line in prompt

    def test_states_the_decoding_rules(self, item: dict[str, Any]) -> None:
        """The rules are spelled out so the task is not rule-guessing.

        Without them the benchmark would measure whether a model can infer our
        definition of decodable rather than whether it can write to it.
        """
        prompt = build_generation_prompt(PROMPT_NAME, item, "sw")
        assert "DECODING RULES" in prompt
        assert "whole word" in prompt
        assert "consonant cluster" in prompt

    def test_states_the_output_contract(self, item: dict[str, Any]) -> None:
        """The response must be wrapped in tags the report can extract."""
        prompt = build_generation_prompt(PROMPT_NAME, item, "sw")
        assert "<text>" in prompt
        assert "</text>" in prompt

    def test_every_scenario_renders(self, item: dict[str, Any]) -> None:
        """No scenario contains anything that breaks str.format."""
        for scenario_id, scenario in get_scenarios().items():
            prompt = build_generation_prompt(
                PROMPT_NAME, {**item, "scenario": scenario}, "sw"
            )
            assert scenario in prompt, scenario_id

    def test_every_template_renders_with_the_standard_fields(
        self, item: dict[str, Any]
    ) -> None:
        """Every prompt in prompts.yaml is renderable by the single code path.

        Guards against adding a template with a placeholder nothing supplies.
        """
        for prompt_name in _load_prompts():
            assert build_generation_prompt(prompt_name, item, "sw")

    def test_unknown_prompt_raises_config_error(self, item: dict[str, Any]) -> None:
        """A bad prompt_name is a config problem, surfaced as ConfigError."""
        with pytest.raises(ConfigError, match="Unknown prompt 'nope'"):
            build_generation_prompt("nope", item, "sw")

    def test_unknown_language_raises_config_error(self, item: dict[str, Any]) -> None:
        """An unregistered language fails before any model call is made."""
        with pytest.raises(ConfigError, match="Unknown language registry key"):
            build_generation_prompt(PROMPT_NAME, item, "zz")

    def test_missing_scenario_raises_config_error(self, item: dict[str, Any]) -> None:
        """An item without a scenario has no task to state."""
        with pytest.raises(ConfigError, match="Missing scenario"):
            build_generation_prompt(PROMPT_NAME, {**item, "scenario": "  "}, "sw")

    def test_invalid_student_knowledge_raises_config_error(
        self, item: dict[str, Any]
    ) -> None:
        """Knowledge the language model rejects fails fast, naming the item."""
        broken = {**item, "student_knowledge": {"graphemes": ["NOT_A_GRAPHEME"]}}
        with pytest.raises(ConfigError, match="Invalid student_knowledge"):
            build_generation_prompt(PROMPT_NAME, broken, "sw")

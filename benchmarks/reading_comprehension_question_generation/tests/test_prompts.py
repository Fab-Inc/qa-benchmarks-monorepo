import inspect

from reading_comprehension_question_generation.prompts import (
    DEFAULT_PROMPT_NAME,
    NUM_PAIRS,
    build_generation_prompt,
    prompt_for_item,
)


def test_prompt_holds_the_passage_language_and_a_json_array_instruction() -> None:
    passage = 'Tom said "go" and {x} {0} {{y}} [1] \\ end.'

    prompt = build_generation_prompt(DEFAULT_PROMPT_NAME, passage, "English")

    assert prompt.endswith(passage)
    assert "Write the questions and the answers in English." in prompt
    assert "JSON array" in prompt
    assert '{"question": "...", "answer": "..."}' in prompt


def test_prompt_renders_a_kiswahili_passage_in_the_item_language() -> None:
    passage = "Pili aliona paka mdogo karibu na mti."

    prompt = build_generation_prompt(DEFAULT_PROMPT_NAME, passage, "Kiswahili")

    assert prompt.endswith(passage)
    assert "Write the questions and the answers in Kiswahili." in prompt
    assert "English" not in prompt


def test_only_the_passage_and_item_language_reach_the_prompt() -> None:
    assert list(inspect.signature(build_generation_prompt).parameters) == [
        "prompt_name",
        "text",
        "language",
    ]
    item = {
        "benchmark_item_id": "A-1",
        "text_content": "Passage.",
        "language": "English",
        "source": "SECRET-SOURCE",
        "text_title": "SECRET-TITLE",
        "grade": "SECRET-GRADE",
    }

    prompt = prompt_for_item(DEFAULT_PROMPT_NAME, item, "english")

    for secret in ("SECRET-SOURCE", "SECRET-TITLE", "SECRET-GRADE", "A-1"):
        assert secret not in prompt
    changed = {**item, "source": "other", "text_title": "other", "grade": "9"}
    assert prompt_for_item(DEFAULT_PROMPT_NAME, changed, "swahili") == prompt
    assert prompt == build_generation_prompt(DEFAULT_PROMPT_NAME, "Passage.", "English")


def test_prompt_asks_for_three_pairs_and_the_count_is_not_an_argument() -> None:
    prompt = build_generation_prompt(DEFAULT_PROMPT_NAME, "Passage.", "English")

    assert NUM_PAIRS == 3
    assert "exactly 3 question and answer pairs" in prompt
    assert "exactly 3 objects" in prompt
    for function in (build_generation_prompt, prompt_for_item):
        names = list(inspect.signature(function).parameters)
        assert not [n for n in names if "pair" in n or "num" in n or "count" in n]

import inspect

from reading_comprehension_question_generation.prompts import (
    DEFAULT_PROMPT_NAME,
    NUM_PAIRS,
    build_generation_prompt,
    prompt_for_item,
)


def test_prompt_holds_the_passage_language_grade_and_a_json_array_instruction() -> None:
    passage = 'Tom said "go" and {x} {0} {{y}} [1] \\ end.'

    prompt = build_generation_prompt(DEFAULT_PROMPT_NAME, passage, "English", 3)

    assert prompt.endswith(passage)
    assert "helping a teacher check whether early primary learners" in prompt
    # One sentence states the language and the grade (as the number 3) and asks for
    # the questions and answers in that language; no separate Language/Grade lines.
    assert (
        "The passage is in English and is for grade 3 learners; "
        "write the questions and the answers in English." in prompt
    )
    assert not [
        ln for ln in prompt.splitlines() if ln.startswith(("Language", "Grade"))
    ]
    for bullet in ("answerable", "short and clear", "repeat"):
        assert bullet not in prompt
    assert "JSON array" in prompt
    assert '{"question": "...", "answer": "..."}' in prompt


def test_prompt_renders_a_kiswahili_passage_in_the_item_language() -> None:
    passage = "Pili aliona paka mdogo karibu na mti."

    prompt = build_generation_prompt(DEFAULT_PROMPT_NAME, passage, "Kiswahili", 12)

    assert prompt.endswith(passage)
    assert "is in Kiswahili and is for grade 12 learners" in prompt
    assert "answers in Kiswahili." in prompt
    assert "English" not in prompt


def test_only_the_passage_language_and_grade_reach_the_prompt() -> None:
    assert list(inspect.signature(build_generation_prompt).parameters) == [
        "prompt_name",
        "text",
        "language",
        "grade",
    ]
    item = {
        "benchmark_item_id": "A-1",
        "text_content": "Passage.",
        "language": "English",
        "source": "SECRET-SOURCE",
        "text_title": "SECRET-TITLE",
        "grade": 3,
    }

    prompt = prompt_for_item(DEFAULT_PROMPT_NAME, item, "english")

    for secret in ("SECRET-SOURCE", "SECRET-TITLE", "A-1"):
        assert secret not in prompt
    changed = {**item, "source": "other", "text_title": "other"}
    assert prompt_for_item(DEFAULT_PROMPT_NAME, changed, "swahili") == prompt
    assert prompt == build_generation_prompt(
        DEFAULT_PROMPT_NAME, "Passage.", "English", 3
    )
    # The item's own grade is what the prompt shows, as a number.
    assert "grade 3 learners" in prompt
    assert "grade 7 learners" in prompt_for_item(
        DEFAULT_PROMPT_NAME, {**item, "grade": 7}, "english"
    )


def test_prompt_asks_for_three_pairs_and_the_count_is_not_an_argument() -> None:
    prompt = build_generation_prompt(DEFAULT_PROMPT_NAME, "Passage.", "English", 2)

    assert NUM_PAIRS == 3
    assert "exactly 3 question and answer pairs" in prompt
    assert "exactly 3 objects" in prompt
    for function in (build_generation_prompt, prompt_for_item):
        names = list(inspect.signature(function).parameters)
        assert not [n for n in names if "pair" in n or "num" in n or "count" in n]

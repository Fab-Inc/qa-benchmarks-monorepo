"""Building the fab-bench inputs file from a Linguistic Knowledge dataset.

The inputs file is the interface between this benchmark and the shared runner:
one rendered JSON object per item (``benchmark_item_id``, ``prompt``,
``system_prompt``), exactly what models see. Everything the model does not see --
``expected_answers``, ``domain``, ``answer_type``, options -- stays in the
dataset and is joined back on ``question_id`` at scoring time.
"""

import json
import tempfile
from pathlib import Path

from linguistic_knowledge_benchmark.shared_utils import (
    english_language_name,
    load_dataset,
)


def benchmark_item_id(question: dict) -> str:
    """The inputs-file item id for a dataset question.

    This is the join key responses.py uses to attach the question's metadata
    back to its result rows, so write and read must string-ify the id in exactly
    the same way -- keep them both going through this function.
    """
    return str(question["question_id"])


def _build_system_prompts(language_name: str) -> dict[str, str]:
    """Return {answer_type: prompt}.

    The medium of evaluation is English for every language: questions are written
    in English and ask about the linguistics of ``language_name``. Every prompt
    is prefixed with that context so the model knows which language the question
    is about, regardless of answer type.
    """
    prefix = (
        "The following question is written in English and concerns the"
        f" linguistics of {english_language_name(language_name)}. "
    )
    return {
        "multiple_choice": (
            prefix + "Answer the multiple-choice question by selecting the letter"
            " corresponding to the correct answer (A, B, C or D)."
            "\nOnly provide the letter for your answer."
            "\nStop exactly after the letter."
        ),
        "short_form": (
            prefix + "Answer the question."
            "\nProvide only the answer in 1-3 words. Do not explain."
        ),
    }


def _build_user_message(question: dict) -> str:
    """Build the user message for a question, folding MCQ options into the text."""
    text = question["question"]
    if question.get("answer_type") == "multiple_choice" and question.get("options"):
        lines = [text, ""]
        for letter in ("A", "B", "C", "D"):
            if letter in question["options"]:
                lines.append(f"{letter}. {question['options'][letter]}")
        return "\n".join(lines)
    return text


def build_inputs_rows(questions: list[dict], language_name: str) -> list[dict]:
    """Render dataset questions as inputs-file rows, in dataset order.

    Each row carries only what the model sees: the item id (dataset
    ``question_id`` stringified), the user prompt, and the system prompt for the
    question's ``answer_type``. An unknown ``answer_type`` fails here, at build
    time, rather than when a run sends the item.
    """
    system_prompts = _build_system_prompts(language_name)
    rows = []
    for question in questions:
        answer_type = question["answer_type"]
        if answer_type not in system_prompts:
            raise ValueError(
                f"Question {question['question_id']} has unknown "
                f"answer_type '{answer_type}'; add a system prompt for it."
            )
        rows.append(
            {
                "benchmark_item_id": benchmark_item_id(question),
                "prompt": _build_user_message(question),
                "system_prompt": system_prompts[answer_type],
            }
        )
    return rows


def write_inputs_file(dataset_path: Path, language_name: str, dest: Path) -> Path:
    """Write ``dest`` as the inputs file for a dataset; return its path.

    Written atomically (temp file + rename) so a run that reads it mid-write
    can't see a partial file.
    """
    rows = build_inputs_rows(load_dataset(dataset_path), language_name)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=dest.parent,
        prefix=dest.name,
        suffix=".tmp",
        delete=False,
    ) as tmp:
        for row in rows:
            tmp.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp_path = Path(tmp.name)
    tmp_path.replace(dest)
    return dest

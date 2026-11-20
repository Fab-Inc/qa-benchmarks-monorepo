"""Reading fab-bench result files as this benchmark's responses table.

fab-bench writes ``responses_<model>.jsonl`` inside an experiment's ``responses/``
folder, with row fields named differently from the CSVs run-bench used to write
(``benchmark_item_id`` rather than ``question_id``, ``latency_seconds``,
``tokens_used*``...), and without the item metadata the model never saw
(``domain``, ``answer_type``, ``expected_answers``). ``load_responses`` reads one
of those files and returns a DataFrame with the old CSV's columns, joined back to
the dataset on ``question_id``, so judged CSVs and every downstream consumer keep
their existing schema.
"""

import warnings
from pathlib import Path

import pandas as pd

from linguistic_knowledge_benchmark.inputs import benchmark_item_id

_RENAMES = {
    "latency_seconds": "latency",
    "tokens_used": "total_tokens",
    "tokens_used_completion": "completion_tokens",
    "tokens_used_reasoning": "reasoning_tokens",
}

_RESPONSES_COLUMNS = [
    "question_id",
    "domain",
    "answer_type",
    "expected_answers",
    "output",
    "latency",
    "total_tokens",
    "completion_tokens",
    "reasoning_tokens",
    "call_status",
    "error_type",
    "error_message",
    "retry_count",
]


def responses_path(experiment_dir: Path, model_name: str) -> Path:
    """The experiment's results file for one model."""
    return Path(experiment_dir) / "responses" / f"responses_{model_name}.jsonl"


def load_responses(
    experiment_dir: Path, model_name: str, questions: list[dict]
) -> pd.DataFrame:
    """Load one model's results as the legacy responses table.

    Args:
        experiment_dir: The fab-bench experiment folder (holds ``responses/``).
        model_name: Model whose ``responses_<model>.jsonl`` to read.
        questions: The dataset's questions, for the metadata join.

    Returns:
        DataFrame with the old responses CSV columns; ``retry_count`` is 0 (the
        runner's --retry-errors rewrites rows in place instead of counting
        passes). Rows for ids absent from the dataset get empty metadata.
    """
    path = responses_path(experiment_dir, model_name)
    rows = pd.read_json(path, lines=True).rename(columns=_RENAMES)
    questions_by_id = {benchmark_item_id(q): q for q in questions}
    rows["question_id"] = rows["benchmark_item_id"].map(
        lambda item_id: questions_by_id.get(str(item_id), {}).get("question_id")
    )
    unmatched = rows["question_id"].isna()
    if unmatched.any():
        message = (
            f"{unmatched.sum()}/{len(rows)} benchmark_item_id(s) in "
            f"{path.name} do not match the dataset -- check --dataset."
        )
        if unmatched.all():
            raise ValueError(message)
        warnings.warn(message)
    for key in ("domain", "answer_type", "expected_answers"):
        rows[key] = rows["benchmark_item_id"].map(
            lambda item_id: questions_by_id.get(str(item_id), {}).get(key)
        )
    rows["expected_answers"] = rows["expected_answers"].map(
        lambda answers: "|".join(answers) if isinstance(answers, list) else answers
    )
    rows["retry_count"] = 0
    # Columns absent from the file (error rows' token fields, ...) come out as
    # NaN here rather than needing a fill for each one.
    return rows.reindex(columns=_RESPONSES_COLUMNS)

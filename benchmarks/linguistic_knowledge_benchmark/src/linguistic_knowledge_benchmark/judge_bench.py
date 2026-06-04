"""
Linguistic Knowledge Eval Judge Module.

Scores model answers from run_bench using regex (MC)
or LLM judge (short_form/true_false).
"""

import json
import re
from pathlib import Path

import pandas as pd
from fdllm import get_caller
from fdllm.llmtypes import LLMMessage
from pydantic import BaseModel
from tqdm import tqdm

from linguistic_knowledge_benchmark.run_bench import (
    CACHE_BASE_DIR,
    load_dataset,
    parse_answer,
)

DEFAULT_JUDGE_MODEL = "claude-sonnet-4-5-20250929"


class JudgeResponse(BaseModel):
    """Structured response from the LLM judge."""

    is_correct: bool
    justification: str


def _build_judge_system_prompt(language_name: str) -> str:
    """Build the judge system prompt for a given language."""
    return (
        f"You are an expert evaluator for {language_name} language knowledge"
        " questions. You will be given a question, the model's answer, and one"
        " or more acceptable expected answers."
        "\n\nYour task is to determine if the model's answer matches one of"
        " the expected answers. Be strict: only accept answers that match"
        " one of the provided expected answers. Do NOT accept synonyms,"
        " paraphrases, or alternative forms that are not listed."
        "\n\nThe only acceptable variations are:"
        " capitalization, leading/trailing hyphens, and spacing."
        "\n\nFor true/false answers, accept 'True'/'False', 'true'/'false',"
        " 'Yes'/'No', or clear equivalents."
    )


JUDGE_FALLBACK_SYSTEM_PROMPT = (
    "You are an expert evaluator. A model was asked a question and its"
    " response could not be parsed by a simple regex."
    "\n\nYou will be given the question, the model's raw response, the"
    " expected answer, and the answer type (multiple_choice or true_false)."
    "\n\nFor multiple-choice: determine if the model's response indicates"
    " the correct letter (A, B, C, or D), even if the response contains"
    " extra explanation or formatting."
    "\n\nFor true/false: determine if the model's response indicates"
    " True or False, accepting 'Yes'/'No', 'Correct'/'Incorrect',"
    " or clear equivalents."
    "\n\nRespond with whether the model's answer is correct."
)


def score_multiple_choice(
    output: str, expected_answers: list[str]
) -> tuple[bool | None, str]:
    """Score a multiple-choice answer using regex parsing.

    Uses parse_answer() to extract the answer letter, then compares with
    expected.

    Args:
        output: Raw model output string
        expected_answers: List with single expected letter (e.g., ["B"])

    Returns:
        Tuple of (is_correct, justification). is_correct is None when
        the answer cannot be parsed. Justification is always empty
        string since no LLM judge is involved.
    """
    parsed = parse_answer(output)
    if parsed is None:
        return None, ""
    expected = expected_answers[0]
    return parsed == expected, ""


def score_true_false_regex(
    output: str, expected_answers: list[str]
) -> tuple[bool, str] | None:
    """Attempt regex scoring for true/false questions.

    Returns None if the output cannot be reliably parsed,
    signaling that LLM judge should be used as fallback.

    Args:
        output: Raw model output string
        expected_answers: ["True"] or ["False"]

    Returns:
        Tuple of (is_correct, justification) or None if unparseable.
        Justification is always empty string when regex succeeds.
    """
    if not output or pd.isna(output):
        return None

    cleaned = str(output).strip().lower()
    expected_lower = expected_answers[0].lower()

    first_word = re.split(r"[\s.,;:!]", cleaned)[0]
    if first_word in ("true", "false"):
        return first_word == expected_lower, ""

    return None


def score_with_llm_judge(
    question: str,
    output: str,
    expected_answers: list[str],
    answer_type: str,
    judge_caller,
    system_prompt: str,
) -> tuple[bool | None, str]:
    """Score an answer using an LLM judge with structured output.

    Args:
        question: The original question text
        output: The model's raw output
        expected_answers: List of acceptable answers
        answer_type: "short_form", "true_false", or "multiple_choice"
        judge_caller: fdllm caller instance for the judge model
        system_prompt: System prompt to use for the judge

    Returns:
        Tuple of (is_correct, justification). is_correct is None if the
        judge call fails.
    """
    user_prompt = (
        f"Question: {question}\n\n"
        f"Model's answer: {output}\n\n"
        f"Acceptable expected answers: {', '.join(expected_answers)}\n\n"
        f"Answer type: {answer_type}\n\n"
        "Is the model's answer correct?"
    )

    messages = [
        LLMMessage(Role="system", Message=system_prompt),
        LLMMessage(Role="user", Message=user_prompt),
    ]

    try:
        response = judge_caller.call(
            messages, max_tokens=1000, response_schema=JudgeResponse
        )
        result = json.loads(response.Message)
        return result["is_correct"], result["justification"]
    except Exception as e:
        return None, f"Judge error: {e}"


def judge_results(
    model_name: str,
    *,
    language_name: str,
    dataset_path: Path,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    force: bool = False,
) -> pd.DataFrame:
    """Score model answers from the Linguistic Knowledge eval.

    Reads resps_{model_name}.csv from the answers directory, scores each answer,
    and saves resps_{model_name}_judged.csv to the answers_judged directory.

    Args:
        model_name: Model whose results to judge
        language_name: Display name of the language (e.g. "Luganda").
        dataset_path: Path to the dataset JSON file.
        judge_model: Model to use as LLM judge
        force: Force re-judge even if judged results exist

    Returns:
        DataFrame with added is_correct and justification columns
    """
    language_dir = CACHE_BASE_DIR / f"cache_{language_name.lower()}"
    input_dir = language_dir / "answers"
    judged_dir = language_dir / "answers_judged"
    judge_system_prompt = _build_judge_system_prompt(language_name)

    # Check input file exists
    input_path = input_dir / f"resps_{model_name}.csv"
    if not input_path.exists():
        raise FileNotFoundError(
            f"No results found for {model_name}. "
            f"Run 'run-bench --model {model_name} "
            f"--language {language_name.lower()}' first."
        )

    # Check judged cache
    output_path = judged_dir / f"resps_{model_name}_judged.csv"
    if not force and output_path.exists():
        print(
            f"Judged results already exist for {model_name}. Use --force to re-judge."
        )
        return pd.read_csv(output_path)

    # Load step 1 results
    results_df = pd.read_csv(input_path)

    # Create judge caller only if needed for short_form
    needs_llm_judge = results_df["answer_type"].isin(["short_form"]).any()
    judge_caller = None
    if needs_llm_judge:
        try:
            judge_caller = get_caller(judge_model)
        except NotImplementedError:
            raise ValueError(
                f"Judge model '{judge_model}' not registered in custom_models.yaml."
            ) from None

    # Load original dataset to get question text for judge prompts
    questions = load_dataset(dataset_path)
    question_map = {q["id"]: q["question"] for q in questions}

    # Score each row
    is_correct_list: list[bool | None] = []
    justification_list = []

    for _, row in tqdm(
        results_df.iterrows(), total=len(results_df), desc=f"Judging {model_name}"
    ):
        output = row["output"]
        expected_answers = str(row["expected_answers"]).split("|")
        answer_type = row["answer_type"]
        question_id = row["id"]

        # Keep missing outputs as null (unscorable), not incorrect.
        if not output or pd.isna(output):
            is_correct_list.append(None)
            justification_list.append("Missing model output")
            continue

        if answer_type == "multiple_choice":
            is_correct, justification = score_multiple_choice(output, expected_answers)
        elif answer_type == "true_false":
            # Regex only; fallback via --judge-fallback
            result = score_true_false_regex(output, expected_answers)
            if result is not None:
                is_correct, justification = result
            else:
                is_correct, justification = None, ""
        elif answer_type == "short_form":
            question_text = question_map.get(question_id, "")
            is_correct, justification = score_with_llm_judge(
                question_text,
                output,
                expected_answers,
                answer_type,
                judge_caller,
                system_prompt=judge_system_prompt,
            )
        else:
            is_correct = None
            justification = ""

        is_correct_list.append(is_correct)
        justification_list.append(justification)

    results_df["is_correct"] = is_correct_list
    results_df["justification"] = justification_list

    # Save judged results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"Judged results saved to {output_path}")

    return results_df


def judge_results_fallback(
    model_name: str,
    *,
    language_name: str,
    dataset_path: Path,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    force: bool = False,
) -> pd.DataFrame:
    """Re-judge null MC/TF rows from answers_judged using LLM fallback.

    Reads the existing judged CSV from answers_judged/, copies it to
    answers_judged_fallback/, and runs the LLM judge only on rows where
    is_correct is null and answer_type is multiple_choice or true_false.

    Args:
        model_name: Model whose results to re-judge
        language_name: Display name of the language (e.g. "Luganda").
        dataset_path: Path to the dataset JSON file.
        judge_model: Model to use as LLM judge
        force: Force re-judge even if fallback results exist

    Returns:
        DataFrame with added is_correct_fallback and justification_fallback
    """
    language_dir = CACHE_BASE_DIR / f"cache_{language_name.lower()}"
    input_dir = language_dir / "answers_judged"
    fallback_dir = language_dir / "answers_judged_fallback"

    # Read existing judged CSV
    input_path = input_dir / f"resps_{model_name}_judged.csv"
    if not input_path.exists():
        raise FileNotFoundError(
            f"No judged results for {model_name} at {input_path}. "
            f"Run 'judge-bench --model {model_name} "
            f"--language {language_name.lower()}' first."
        )

    output_path = fallback_dir / f"resps_{model_name}_judged.csv"
    if not force and output_path.exists():
        print(
            f"Fallback results already exist for {model_name}. Use --force to re-judge."
        )
        return pd.read_csv(output_path)

    df = pd.read_csv(input_path)

    # Find null is_correct rows that are MC or TF
    null_mask = df["is_correct"].isna() & df["answer_type"].isin(
        ["multiple_choice", "true_false"]
    )
    null_count = int(null_mask.sum())
    print(f"  {model_name}: {null_count} null rows to re-evaluate with LLM fallback")

    # Initialize fallback columns
    is_correct_fallback: list[bool | None] = [None] * len(df)
    justification_fallback = [""] * len(df)

    if null_count == 0:
        df["is_correct_fallback"] = is_correct_fallback
        df["justification_fallback"] = justification_fallback
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return df

    # Create judge caller
    try:
        judge_caller = get_caller(judge_model)
    except NotImplementedError:
        raise ValueError(
            f"Judge model '{judge_model}' not registered in custom_models.yaml."
        ) from None

    # Load questions for judge prompts
    questions = load_dataset(dataset_path)
    question_map = {q["id"]: q["question"] for q in questions}

    # Score only null MC/TF rows
    for idx in tqdm(
        df[null_mask].index,
        desc=f"Fallback judging {model_name}",
    ):
        row = df.loc[idx]
        question_text = question_map.get(row["id"], "")
        expected_answers = str(row["expected_answers"]).split("|")
        result, justification = score_with_llm_judge(
            question_text,
            row["output"],
            expected_answers,
            row["answer_type"],
            judge_caller,
            system_prompt=JUDGE_FALLBACK_SYSTEM_PROMPT,
        )
        is_correct_fallback[idx] = result
        justification_fallback[idx] = justification

    df["is_correct_fallback"] = is_correct_fallback
    df["justification_fallback"] = justification_fallback

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Fallback results saved to {output_path}")
    return df


def compute_summary(judged_df: pd.DataFrame, model_name: str) -> dict:
    """Compute summary statistics from judged results.

    When ``is_correct_fallback`` is present (from ``--judge-fallback``),
    accuracy is computed from the best-available result: fallback where
    populated, else the original ``is_correct``.

    Args:
        judged_df: DataFrame with is_correct column (and optionally
            is_correct_fallback)
        model_name: Model name for the summary

    Returns:
        Dict with model, accuracy, and per-type accuracy
    """
    # Use fallback where available, else original is_correct
    if "is_correct_fallback" in judged_df.columns:
        effective = judged_df["is_correct_fallback"].fillna(judged_df["is_correct"])
    else:
        effective = judged_df["is_correct"]

    scored_mask = effective.notna()
    scored_df = effective[scored_mask]
    total_scored = len(scored_df)
    correct = scored_df.sum()
    accuracy = correct / total_scored * 100 if total_scored > 0 else 0.0

    by_type = {}
    for atype in judged_df["answer_type"].unique():
        type_mask = judged_df["answer_type"] == atype
        subset = effective[type_mask]
        subset_scored = subset[subset.notna()]
        if len(subset_scored) == 0:
            by_type[atype] = 0.0
        else:
            by_type[atype] = round(subset_scored.sum() / len(subset_scored) * 100, 2)

    return {
        "model": model_name,
        "accuracy": round(accuracy, 2),
        "scored_count": total_scored,
        "unscored_count": int((~scored_mask).sum()),
        "accuracy_mc": by_type.get("multiple_choice", 0.0),
        "accuracy_short_form": by_type.get("short_form", 0.0),
        "accuracy_true_false": by_type.get("true_false", 0.0),
    }

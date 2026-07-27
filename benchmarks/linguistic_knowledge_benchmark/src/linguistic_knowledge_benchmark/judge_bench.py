"""
Linguistic Knowledge Eval Judge Module.

Scores model responses from run_bench using regex (MCQ)
or LLM judge (short_form/true_false).
"""

import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from fdllm import get_caller
from fdllm.llmtypes import LLMMessage
from pydantic import BaseModel
from tqdm import tqdm

from linguistic_knowledge_benchmark.run_bench import (
    git_hash,
    load_dataset,
    parse_answer,
    update_log,
)
from linguistic_knowledge_benchmark.shared_utils import english_language_name, run_dir

DEFAULT_JUDGE_MODEL = "claude-sonnet-4-5-20250929"
SCORE_METHOD_REGEX = "regex"
SCORE_METHOD_LLM_JUDGE = "llm_judge"
SCORE_METHOD_LLM_FALLBACK = "llm_fallback"
SCORE_METHOD_UNSCORED = "unscored"


class JudgeResponse(BaseModel):
    """Structured response from the LLM judge."""

    is_correct: bool
    justification: str


def _build_judge_system_prompt(language_name: str) -> str:
    """Build the judge system prompt for a given language."""
    return (
        f"You are an expert evaluator for {english_language_name(language_name)}"
        " language knowledge questions. You will be given a question, the"
        " model's answer, and one or more acceptable expected answers."
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
) -> tuple[bool | None, str, str | None]:
    """Score a multiple-choice response using regex parsing.

    Uses parse_answer() to extract the response letter, then compares with
    expected.

    Args:
        output: Raw model output string
        expected_answers: List with single expected letter (e.g., ["B"])

    Returns:
        Tuple of (is_correct, justification, parsed_answer). is_correct is None
        when the response cannot be parsed. Justification is always empty string
        since no LLM judge is involved.
    """
    parsed = parse_answer(output)
    if parsed is None:
        return None, "", None
    expected = expected_answers[0]
    return parsed == expected, "", parsed


def score_true_false_regex(
    output: str, expected_answers: list[str]
) -> tuple[bool, str, str] | None:
    """Attempt regex scoring for true/false questions.

    Returns None if the output cannot be reliably parsed,
    signaling that LLM judge should be used as fallback.

    Args:
        output: Raw model output string
        expected_answers: ["True"] or ["False"]

    Returns:
        Tuple of (is_correct, justification, parsed_answer) or None if unparseable.
        Justification is always empty string when regex succeeds.
    """
    if not output or pd.isna(output):
        return None

    cleaned = str(output).strip().lower()
    expected_lower = expected_answers[0].lower()

    first_word = re.split(r"[\s.,;:!]", cleaned)[0]
    if first_word in ("true", "false"):
        parsed = first_word.capitalize()
        return first_word == expected_lower, "", parsed

    return None


def score_with_llm_judge(
    question: str,
    output: str,
    expected_answers: list[str],
    answer_type: str,
    judge_caller,
    system_prompt: str,
) -> tuple[bool | None, str]:
    """Score a response using an LLM judge with structured output.

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
    run_id: str | None = None,
) -> pd.DataFrame:
    """Score model responses from the Linguistic Knowledge eval.

    Reads responses_{model_name}.csv from the responses directory, scores each
    response, and saves responses_judged_{model_name}.csv to the
    responses_judged directory.

    Args:
        model_name: Model whose results to judge
        language_name: Display name of the language (e.g. "Luganda").
        dataset_path: Path to the dataset JSON file.
        judge_model: Model to use as LLM judge
        force: Force re-judge even if judged results exist
        run_id: Optional run ID; reads/writes under
            cache/<language>/runs/<run_id>/.
            Defaults to 'default' when not specified.

    Returns:
        DataFrame with added is_correct and justification columns
    """
    language_dir = run_dir(language_name, run_id)
    input_dir = language_dir / "responses"
    judged_dir = language_dir / "responses_judged"
    judge_system_prompt = _build_judge_system_prompt(language_name)

    # Check input file exists
    input_path = input_dir / f"responses_{model_name}.csv"
    if not input_path.exists():
        raise FileNotFoundError(
            f"No results found for {model_name}. "
            f"Run 'run-bench --model {model_name} "
            f"--language {language_name.lower()}' first."
        )

    # Check judged cache
    output_path = judged_dir / f"responses_judged_{model_name}.csv"
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
    question_map = {q["question_id"]: q["question"] for q in questions}

    # Score each row
    is_correct_list: list[bool | None] = []
    justification_list = []
    score_method_list = []
    parsed_answer_list = []

    for _, row in tqdm(
        results_df.iterrows(), total=len(results_df), desc=f"Judging {model_name}"
    ):
        output = row["output"]
        expected_answers = str(row["expected_answers"]).split("|")
        answer_type = row["answer_type"]
        question_id = row["question_id"]

        if row.get("call_status") == "error":
            error_type = row.get("error_type", "")
            error_message = row.get("error_message", "")
            is_correct_list.append(None)
            justification_list.append(
                f"Model call error: {error_type}: {error_message}".strip()
            )
            score_method_list.append(SCORE_METHOD_UNSCORED)
            parsed_answer_list.append(None)
            continue

        # Keep missing outputs as null (unscorable), not incorrect.
        if not output or pd.isna(output):
            is_correct_list.append(None)
            justification_list.append("Missing model output")
            score_method_list.append(SCORE_METHOD_UNSCORED)
            parsed_answer_list.append(None)
            continue

        if answer_type == "multiple_choice":
            is_correct, justification, parsed_answer = score_multiple_choice(
                output, expected_answers
            )
            score_method = (
                SCORE_METHOD_REGEX
                if parsed_answer is not None
                else SCORE_METHOD_UNSCORED
            )
        elif answer_type == "true_false":
            # Regex only; fallback via --judge-fallback
            result = score_true_false_regex(output, expected_answers)
            if result is not None:
                is_correct, justification, parsed_answer = result
                score_method = SCORE_METHOD_REGEX
            else:
                is_correct, justification = None, ""
                parsed_answer = None
                score_method = SCORE_METHOD_UNSCORED
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
            parsed_answer = None
            score_method = (
                SCORE_METHOD_LLM_JUDGE
                if is_correct is not None
                else SCORE_METHOD_UNSCORED
            )
        else:
            is_correct = None
            justification = ""
            parsed_answer = None
            score_method = SCORE_METHOD_UNSCORED

        is_correct_list.append(is_correct)
        justification_list.append(justification)
        score_method_list.append(score_method)
        parsed_answer_list.append(parsed_answer)

    results_df["is_correct"] = is_correct_list
    results_df["justification"] = justification_list
    results_df["score_method"] = score_method_list
    results_df["parsed_answer"] = parsed_answer_list

    # Save judged results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    update_log(
        language_dir,
        model_name,
        {
            "responses_judged_timestamp-utc": datetime.now(UTC).isoformat(),
            "responses_judged_git-hash": git_hash(),
        },
    )
    print(f"Judged results saved to {output_path}")

    return results_df


def judge_results_fallback(
    model_name: str,
    *,
    language_name: str,
    dataset_path: Path,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    force: bool = False,
    run_id: str | None = None,
) -> pd.DataFrame:
    """Re-judge null MCQ/TF rows from responses_judged using LLM fallback.

    Reads the existing judged CSV from responses_judged/, copies it to
    responses_judged_fallback/, and runs the LLM judge only on rows where
    is_correct is null and answer_type is multiple_choice or true_false.

    Args:
        model_name: Model whose results to re-judge
        language_name: Display name of the language (e.g. "Luganda").
        dataset_path: Path to the dataset JSON file.
        judge_model: Model to use as LLM judge
        force: Force re-judge even if fallback results exist
        run_id: Optional run ID; reads/writes under
            cache/<language>/runs/<run_id>/.
            Defaults to 'default' when not specified.

    Returns:
        DataFrame with added is_correct_fallback and justification_fallback
    """
    language_dir = run_dir(language_name, run_id)
    input_dir = language_dir / "responses_judged"
    fallback_dir = language_dir / "responses_judged_fallback"

    # Read existing judged CSV
    input_path = input_dir / f"responses_judged_{model_name}.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"No result to judge for {model_name}. ")

    output_path = fallback_dir / f"responses_judged_{model_name}.csv"
    if not force and output_path.exists():
        print(
            f"Fallback results already exist for {model_name}. Use --force to re-judge."
        )
        return pd.read_csv(output_path)

    df = pd.read_csv(input_path)
    if "score_method" not in df.columns:
        df["score_method"] = SCORE_METHOD_UNSCORED
    if "parsed_answer" not in df.columns:
        df["parsed_answer"] = None

    # Find null is_correct rows that are MCQ or TF
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
        update_log(
            language_dir,
            model_name,
            {
                "responses_judged_fallback_timestamp-utc": datetime.now(
                    UTC
                ).isoformat(),
                "responses_judged_fallback_git-hash": git_hash(),
            },
        )
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
    question_map = {q["question_id"]: q["question"] for q in questions}

    # Score only null MCQ/TF rows
    for idx in tqdm(
        df[null_mask].index,
        desc=f"Fallback judging {model_name}",
    ):
        row = df.loc[idx]
        question_text = question_map.get(row["question_id"], "")
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
        df.loc[idx, "score_method"] = (
            SCORE_METHOD_LLM_FALLBACK if result is not None else SCORE_METHOD_UNSCORED
        )

    df["is_correct_fallback"] = is_correct_fallback
    df["justification_fallback"] = justification_fallback

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    update_log(
        language_dir,
        model_name,
        {
            "responses_judged_fallback_timestamp-utc": datetime.now(UTC).isoformat(),
            "responses_judged_fallback_git-hash": git_hash(),
        },
    )
    print(f"Fallback results saved to {output_path}")
    return df


def compute_summary(judged_df: pd.DataFrame, model_name: str) -> dict[str, object]:
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
        "accuracy_mcq": by_type.get("multiple_choice", 0.0),
        "accuracy_short_form": by_type.get("short_form", 0.0),
    }


def build_detailed(
    judged_by_model: dict[str, pd.DataFrame], dataset_path: str | Path
) -> pd.DataFrame:
    """Build a per-question detail table across models.

    One row per question (from the dataset, so every item appears even if a model
    skipped it), with the expected answer and, for each model, two columns:
    ``<model>_output`` (the raw model output) and ``<model>_correct`` (True/False,
    or blank when unscored). Correctness uses the fallback verdict where present,
    matching ``compute_summary``.

    Args:
        judged_by_model: model name -> its judged_df (question_id, output,
            is_correct[, is_correct_fallback]).
        dataset_path: the eval dataset JSON (question_id, domain, grade, question,
            answer_type, expected_answers).

    Returns:
        DataFrame ordered by question_id: metadata columns then per-model columns.
    """
    ds = pd.read_json(dataset_path)
    meta_cols = [
        c
        for c in [
            "question_id",
            "domain",
            "grade",
            "answer_type",
            "question",
            "expected_answers",
        ]
        if c in ds.columns
    ]
    detailed = ds[meta_cols].copy()
    if "expected_answers" in detailed.columns:
        detailed["expected_answers"] = detailed["expected_answers"].map(
            lambda v: ", ".join(map(str, v)) if isinstance(v, list) else v
        )

    for model_name, jdf in judged_by_model.items():
        if not {"question_id", "output", "is_correct"}.issubset(jdf.columns):
            continue  # nothing judged for this model (e.g. an empty result frame)
        if "is_correct_fallback" in jdf.columns:
            corr = jdf["is_correct_fallback"].fillna(jdf["is_correct"])
        else:
            corr = jdf["is_correct"]
        per_model = pd.DataFrame(
            {
                "question_id": jdf["question_id"].values,
                f"{model_name}_output": jdf["output"].values,
                f"{model_name}_correct": corr.values,
            }
        )
        detailed = detailed.merge(per_model, on="question_id", how="left")

    return detailed.sort_values("question_id").reset_index(drop=True)

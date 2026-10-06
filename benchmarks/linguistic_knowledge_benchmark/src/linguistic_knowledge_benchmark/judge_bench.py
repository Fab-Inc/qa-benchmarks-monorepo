"""
Linguistic Knowledge Eval Judge Module.

Scores model responses from run_bench using regex (MCQ)
or LLM judge (short_form).
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
    CALL_STATUS_ERROR,
    git_hash,
    load_dataset,
    update_log,
)
from linguistic_knowledge_benchmark.shared_utils import (
    english_language_name,
    experiment_dir,
)

# Share of a model's rows that may have call_status="error" and still be judged.
# Above this the model is skipped entirely: its accuracy would be computed from
# whichever items happened to succeed, which is a biased subset of the dataset, and
# that number would then sit in the summary CSV next to models that answered
# everything. Such a model needs re-running ('run-bench --retry-errors'), not judging,
# so it's kept out of judging AND out of the summary rather than footnoted in it.
MAX_CALL_ERROR_RATE = 0.05

# Independent judges spanning different providers
# Kept to 3 (odd) so majority_vote() never ties.
DEFAULT_JUDGE_MODELS: list[str] = [
    "claude-sonnet-5",
    "gpt-5.6-terra",
    "gemini-3.6-flash",
]
SCORE_METHOD_REGEX = "regex"
SCORE_METHOD_LLM_JUDGE = "llm_judge"
SCORE_METHOD_LLM_FALLBACK = "llm_fallback"
SCORE_METHOD_UNSCORED = "unscored"

# MCQ letter-extraction regexes.
ANSWER_PATTERNS_MCQ = [
    r"^\s*([ABCD])(?:[\.(?:\s*\n)]+.*)*$",
    r"^\s*\(([ABCD])\)",  # letter in parentheses, e.g. (C)
    r"^<think>[\s\S]*?</think>[\s\S]*?([ABCD])$",  # for deepseek R1
    r"^## Step 1[\s\S]*?([ABCD])[\.\s]*$",  # for llama 4
    r'[\s\S]*\n([A-D])"?$',  # for claude 4
    # for glm 4.1v-9b-thinking
    r"<answer>(?:<\|begin_of_box\|>)?([ABCD])(?:<\|end_of_box\|>)?$",
]
ANSWER_REGEXES_MCQ = [re.compile(pat) for pat in ANSWER_PATTERNS_MCQ]


def parse_answer_mcq(resp: str) -> str | None:
    """
    Parse a multiple-choice model response to extract the answer letter (A-D).
    Returns None if no valid answer found.
    """
    if pd.isna(resp) or resp is None:
        return None

    resp = str(resp).strip()

    for regex in ANSWER_REGEXES_MCQ:
        match = regex.match(resp)
        if match is not None:
            return match.groups()[0]

    return None


THINK_BLOCK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)


def strip_think_block(output: str) -> str:
    """Remove <think>...</think> reasoning blocks from a model response.

    Some (reasoning) models prefix their answer with a long chain-of-thought
    block; it's of no use to the LLM judge and can dwarf the actual answer in
    length, so it's stripped from the text sent to the judge (the raw output is
    still what's persisted to the judged CSV).
    """
    if not isinstance(output, str):
        return output
    return THINK_BLOCK_RE.sub("", output).strip()


def count_think_blocks(outputs: pd.Series) -> int:
    """Number of rows in `outputs` containing a <think> tag."""
    return int(
        outputs.astype(str).str.contains("<think>", case=False, regex=False).sum()
    )


class TooManyCallErrorsError(Exception):
    """A model's responses hold a larger share of failed calls than
    MAX_CALL_ERROR_RATE allows, so it is not judged at all."""


def check_call_error_rate(
    df: pd.DataFrame, model_name: str, *, language_name: str | None = None
) -> None:
    """Raise TooManyCallErrorsError when more than MAX_CALL_ERROR_RATE of `df`'s rows
    are call errors.

    The limit is read from the module global on every call rather than bound as a
    default argument, so it stays the single place to change it (and tests exercising
    error-row handling can raise it out of the way).

    A frame with no call_status column predates that column (or is empty), and counts
    as zero errors rather than being treated as suspect. The message names the re-run
    command, since re-running is the only thing that makes such a model judgeable.
    """
    n_rows = len(df)
    if n_rows == 0 or "call_status" not in df.columns:
        return
    n_errors = int((df["call_status"].astype(str) == CALL_STATUS_ERROR).sum())
    rate = n_errors / n_rows
    if rate <= MAX_CALL_ERROR_RATE:
        return
    retry_cmd = f"run-bench --model {model_name} --retry-errors"
    if language_name:
        retry_cmd += f" --language {language_name.lower()}"
    raise TooManyCallErrorsError(
        f"{n_errors}/{n_rows} rows failed to call ({rate:.1%}), above the "
        f"{MAX_CALL_ERROR_RATE:.0%} MAX_CALL_ERROR_RATE. Not judged and left out of "
        f"the summary -- re-run first: {retry_cmd}"
    )


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
    )


JUDGE_FALLBACK_SYSTEM_PROMPT = (
    "You are an expert evaluator. A model was asked a question and its"
    " response could not be parsed by a simple regex."
    "\n\nYou will be given the question, the model's raw response, the"
    " expected answer, and the answer type (multiple_choice)."
    "\n\nFor multiple-choice: determine if the model's response indicates"
    " the correct letter (A, B, C, or D), even if the response contains"
    " extra explanation or formatting."
    "\n\nRespond with whether the model's answer is correct."
)


def score_multiple_choice(
    output: str, expected_answers: list[str]
) -> tuple[bool | None, str, str | None]:
    """Score a multiple-choice response using regex parsing.

    Uses parse_answer_mcq() to extract the response letter, then compares with
    expected.

    Args:
        output: Raw model output string
        expected_answers: List with single expected letter (e.g., ["B"])

    Returns:
        Tuple of (is_correct, justification, parsed_answer). is_correct is None
        when the response cannot be parsed. Justification is always empty string
        since no LLM judge is involved.
    """
    parsed = parse_answer_mcq(output)
    if parsed is None:
        return None, "", None
    expected = expected_answers[0]
    return parsed == expected, "", parsed


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
        answer_type: "short_form" or "multiple_choice"
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
            messages, max_tokens=None, response_schema=JudgeResponse
        )
        result = json.loads(response.Message)
        return result["is_correct"], result["justification"]
    except Exception as e:
        return None, f"Judge error: {e}"


def score_with_llm_judges(
    question: str,
    output: str,
    expected_answers: list[str],
    answer_type: str,
    judge_callers: dict[str, object],
    system_prompt: str,
) -> dict[str, tuple[bool | None, str]]:
    """Score a response independently with each of several judges.

    Args:
        judge_callers: judge model name -> fdllm caller instance, in the order the
            judges should be consulted (iteration order is preserved in the result).

    Returns:
        judge model name -> (is_correct, justification), one entry per judge in
        `judge_callers`.
    """
    return {
        model_name: score_with_llm_judge(
            question, output, expected_answers, answer_type, caller, system_prompt
        )
        for model_name, caller in judge_callers.items()
    }


def majority_vote(verdicts: list[bool | None]) -> tuple[bool | None, float | None]:
    """Majority vote across judge verdicts, ignoring judges whose call failed (None).

    Returns (majority_verdict, agreement): agreement is the fraction of RESPONDING
    judges that agree with the majority. Both are None if no judge responded; the
    verdict is None (unscored) on an exact tie, since arbitrarily picking a side
    would hide a genuine split from the human-review gate.
    """
    responded = [v for v in verdicts if v is not None]
    if not responded:
        return None, None
    true_count = sum(responded)
    false_count = len(responded) - true_count
    if true_count == false_count:
        return None, 0.5
    verdict = true_count > false_count
    agreement = max(true_count, false_count) / len(responded)
    return verdict, agreement


def judge_results(
    model_name: str,
    *,
    language_name: str,
    dataset_path: Path,
    judge_models: list[str] | None = None,
    force: bool = False,
    experiment_id: str | None = None,
) -> pd.DataFrame:
    """Score model responses from the Linguistic Knowledge eval.

    Reads responses_{model_name}.csv from the responses directory, scores each
    response, and saves responses_judged_{model_name}.csv to the
    responses_judged directory.

    Args:
        model_name: Model whose results to judge
        language_name: Display name of the language (e.g. "Kiswahili").
        dataset_path: Path to the dataset JSON file.
        judge_models: Models to use as independent LLM judges for short_form rows.
            Defaults to a single judge, DEFAULT_JUDGE_MODELS[:1] (matching the
            original single-judge behavior/cost); pass more for an ensemble. With
            more than one, `is_correct`/`justification` reflect the majority
            verdict (ties are left unscored) and its primary judge's rationale;
            `judge_agreement`/`judge_disagreement_flag` summarize consensus, and
            each judge's own verdict lands in
            `is_correct_<judge>`/`justification_<judge>`.
        force: Force re-judge even if judged results exist
        experiment_id: Optional experiment ID; reads/writes under
            cache/<language>/experiments/<experiment_id>/.
            Defaults to 'default' when not specified.

    Returns:
        DataFrame with added is_correct and justification columns
    """
    judge_models = judge_models if judge_models else DEFAULT_JUDGE_MODELS[:1]
    language_dir = experiment_dir(language_name, experiment_id)
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

    # Load step 1 results
    results_df = pd.read_csv(input_path)

    # Gate on the run's call-error rate BEFORE the judged-cache shortcut below, so a
    # model that was judged in an earlier pass (when the gate didn't exist, or with
    # --force) still can't slip into this run's summary.
    check_call_error_rate(results_df, model_name, language_name=language_name)

    # Check judged cache
    output_path = judged_dir / f"responses_judged_{model_name}.csv"
    if not force and output_path.exists():
        print(
            f"Judged results already exist for {model_name}. Use --force to re-judge."
        )
        return pd.read_csv(output_path)

    # Create judge callers only if needed for short_form, fail fast if any judge
    # model isn't registered (same policy as the old single-judge behavior).
    needs_llm_judge = results_df["answer_type"].isin(["short_form"]).any()
    judge_callers: dict[str, object] = {}
    if needs_llm_judge:
        for jm in judge_models:
            try:
                judge_callers[jm] = get_caller(jm)
            except NotImplementedError:
                raise ValueError(
                    f"Judge model '{jm}' not registered in custom_models.yaml."
                ) from None

    # Load original dataset to get question text for judge prompts
    questions = load_dataset(dataset_path)
    question_map = {q["question_id"]: q["question"] for q in questions}

    # Whether a model wraps its answer in a <think>...</think> block is a
    # model-level characteristic, not a per-row fluke, so check across all of
    # this model's responses once up front rather than silently stripping it
    # row by row below.
    n_think = count_think_blocks(results_df["output"])
    if n_think:
        print(
            f"  {model_name}: {n_think} row(s) contain <think> reasoning blocks; "
            "stripping before sending to the LLM judge."
        )

    # Score each row
    is_correct_list: list[bool | None] = []
    justification_list = []
    score_method_list = []
    parsed_answer_list = []
    judge_agreement_list: list[float | None] = []
    judge_disagreement_flag_list: list[bool] = []
    per_judge_correct: dict[str, list[bool | None]] = {jm: [] for jm in judge_models}
    per_judge_justification: dict[str, list[str | None]] = {
        jm: [] for jm in judge_models
    }

    def _append_no_judge_row() -> None:
        judge_agreement_list.append(None)
        judge_disagreement_flag_list.append(False)
        for jm in judge_models:
            per_judge_correct[jm].append(None)
            per_judge_justification[jm].append(None)

    # tqdm tracks only the rows that actually go to an LLM judge (short_form);
    # MCQ rows are scored by regex in this same loop but shouldn't count toward
    # the bar's total, or it reads as far more LLM calls happening than really are.
    n_llm_judged = int((results_df["answer_type"] == "short_form").sum())
    progress = tqdm(total=n_llm_judged, desc=f"Judging {model_name}")

    for _, row in results_df.iterrows():
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
            _append_no_judge_row()
            continue

        # Keep missing outputs as null (unscorable), not incorrect.
        if not output or pd.isna(output):
            is_correct_list.append(None)
            justification_list.append("Missing model output")
            score_method_list.append(SCORE_METHOD_UNSCORED)
            parsed_answer_list.append(None)
            _append_no_judge_row()
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
            _append_no_judge_row()
        elif answer_type == "short_form":
            question_text = question_map.get(question_id, "")
            per_judge = score_with_llm_judges(
                question_text,
                strip_think_block(output),
                expected_answers,
                answer_type,
                judge_callers,
                system_prompt=judge_system_prompt,
            )
            is_correct, judge_agreement = majority_vote(
                [v[0] for v in per_judge.values()]
            )
            # Primary judge's own rationale, for a single at-a-glance column; the
            # full per-judge breakdown lives in justification_<judge> below.
            justification = per_judge[judge_models[0]][1]
            parsed_answer = None
            score_method = (
                SCORE_METHOD_LLM_JUDGE
                if is_correct is not None
                else SCORE_METHOD_UNSCORED
            )
            judge_agreement_list.append(judge_agreement)
            judge_disagreement_flag_list.append(
                judge_agreement is not None and judge_agreement < 1.0
            )
            for jm, (verdict, just) in per_judge.items():
                per_judge_correct[jm].append(verdict)
                per_judge_justification[jm].append(just)
            progress.update(1)
        else:
            is_correct = None
            justification = ""
            parsed_answer = None
            score_method = SCORE_METHOD_UNSCORED
            _append_no_judge_row()

        is_correct_list.append(is_correct)
        justification_list.append(justification)
        score_method_list.append(score_method)
        parsed_answer_list.append(parsed_answer)

    progress.close()

    results_df["is_correct"] = is_correct_list
    results_df["justification"] = justification_list
    results_df["score_method"] = score_method_list
    results_df["parsed_answer"] = parsed_answer_list
    results_df["judge_agreement"] = judge_agreement_list
    results_df["judge_disagreement_flag"] = judge_disagreement_flag_list
    for jm in judge_models:
        results_df[f"is_correct_{jm}"] = per_judge_correct[jm]
        results_df[f"justification_{jm}"] = per_judge_justification[jm]

    # Save judged results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    update_log(
        language_dir,
        model_name,
        {
            "responses_judged_timestamp_utc": datetime.now(UTC).isoformat(),
            "responses_judged_git_hash": git_hash(),
            "short_form_llm_judges": judge_models,
        },
    )
    print(f"Judged results saved to {output_path}")

    return results_df


def judge_results_fallback(
    model_name: str,
    *,
    language_name: str,
    dataset_path: Path,
    judge_models: list[str] | None = None,
    force: bool = False,
    experiment_id: str | None = None,
) -> pd.DataFrame:
    """Re-judge null MCQ rows from responses_judged using LLM fallback.

    Reads the existing judged CSV from responses_judged/, copies it to
    responses_judged_fallback/, and runs the LLM judge only on rows where
    is_correct is null and answer_type is multiple_choice.

    Args:
        model_name: Model whose results to re-judge
        language_name: Display name of the language (e.g. "Kiswahili").
        dataset_path: Path to the dataset JSON file.
        judge_models: Models to use as independent LLM judges. Defaults to a
            single judge, DEFAULT_JUDGE_MODELS[:1]; see judge_results() for how
            more than one is combined (same majority-vote/agreement scheme, with
            `_fallback` column names here: `judge_agreement_fallback`,
            `judge_disagreement_flag_fallback`,
            `is_correct_fallback_<judge>`/`justification_fallback_<judge>`).
        force: Force re-judge even if fallback results exist
        experiment_id: Optional experiment ID; reads/writes under
            cache/<language>/experiments/<experiment_id>/.
            Defaults to 'default' when not specified.

    Returns:
        DataFrame with added is_correct_fallback and justification_fallback
    """
    judge_models = judge_models if judge_models else DEFAULT_JUDGE_MODELS[:1]
    language_dir = experiment_dir(language_name, experiment_id)
    input_dir = language_dir / "responses_judged"
    fallback_dir = language_dir / "responses_judged_fallback"

    # Read existing judged CSV
    input_path = input_dir / f"responses_judged_{model_name}.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"No result to judge for {model_name}. ")

    df = pd.read_csv(input_path)
    # Same gate as judge_results(): the judged CSV carries call_status through, so a
    # model with too many failed calls is kept out of the fallback pass and its
    # summary too, not just out of the standard one.
    check_call_error_rate(df, model_name, language_name=language_name)

    output_path = fallback_dir / f"responses_judged_{model_name}.csv"
    if not force and output_path.exists():
        print(
            f"Fallback results already exist for {model_name}. Use --force to re-judge."
        )
        return pd.read_csv(output_path)
    if "score_method" not in df.columns:
        df["score_method"] = SCORE_METHOD_UNSCORED
    if "parsed_answer" not in df.columns:
        df["parsed_answer"] = None

    # Find null is_correct rows that are MCQ and actually have an output to
    # re-evaluate. Rows where the model call itself failed (or otherwise left no
    # output, e.g. an empty response) have nothing for the fallback judge to look
    # at -- sending them would waste a call and can't produce a real verdict, so
    # they're left unscored rather than included here.
    has_output = df["output"].notna() & (df["output"].astype(str).str.strip() != "")
    mcq_null_mask = df["is_correct"].isna() & (df["answer_type"] == "multiple_choice")
    null_mask = mcq_null_mask & has_output
    null_count = int(null_mask.sum())
    no_output_count = int((mcq_null_mask & ~has_output).sum())
    print(
        f"  {model_name}: {null_count} null MCQ rows to re-evaluate with LLM fallback"
        + (
            f" ({no_output_count} skipped -- no model output, e.g. call errors)"
            if no_output_count
            else ""
        )
    )

    # Same model-level <think> block check as judge_results(), scoped to the
    # rows actually being sent to the fallback judge.
    n_think = count_think_blocks(df.loc[null_mask, "output"])
    if n_think:
        print(
            f"  {model_name}: {n_think} row(s) contain <think> reasoning blocks; "
            "stripping before sending to the LLM judge."
        )

    # Initialize fallback columns
    is_correct_fallback: list[bool | None] = [None] * len(df)
    justification_fallback: list[str | None] = [""] * len(df)
    judge_agreement_fallback: list[float | None] = [None] * len(df)
    judge_disagreement_flag_fallback: list[bool] = [False] * len(df)
    per_judge_correct_fallback: dict[str, list[bool | None]] = {
        jm: [None] * len(df) for jm in judge_models
    }
    per_judge_justification_fallback: dict[str, list[str | None]] = {
        jm: [None] * len(df) for jm in judge_models
    }

    def _write_fallback_columns() -> None:
        df["is_correct_fallback"] = is_correct_fallback
        df["justification_fallback"] = justification_fallback
        df["judge_agreement_fallback"] = judge_agreement_fallback
        df["judge_disagreement_flag_fallback"] = judge_disagreement_flag_fallback
        for jm in judge_models:
            df[f"is_correct_fallback_{jm}"] = per_judge_correct_fallback[jm]
            df[f"justification_fallback_{jm}"] = per_judge_justification_fallback[jm]

    if null_count == 0:
        _write_fallback_columns()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        update_log(
            language_dir,
            model_name,
            {
                "responses_judged_fallback_timestamp_utc": datetime.now(
                    UTC
                ).isoformat(),
                "responses_judged_fallback_git_hash": git_hash(),
                "mcq_llm_judges_fallback": judge_models,
            },
        )
        return df

    # Create judge callers, fail fast if any judge model isn't registered.
    judge_callers: dict[str, object] = {}
    for jm in judge_models:
        try:
            judge_callers[jm] = get_caller(jm)
        except NotImplementedError:
            raise ValueError(
                f"Judge model '{jm}' not registered in custom_models.yaml."
            ) from None

    # Load questions for judge prompts
    questions = load_dataset(dataset_path)
    question_map = {q["question_id"]: q["question"] for q in questions}

    # Score only null MCQ rows
    for idx in tqdm(
        df[null_mask].index,
        desc=f"Fallback judging {model_name}",
    ):
        row = df.loc[idx]
        question_text = question_map.get(row["question_id"], "")
        expected_answers = str(row["expected_answers"]).split("|")
        per_judge = score_with_llm_judges(
            question_text,
            strip_think_block(row["output"]),
            expected_answers,
            row["answer_type"],
            judge_callers,
            system_prompt=JUDGE_FALLBACK_SYSTEM_PROMPT,
        )
        result, agreement = majority_vote([v[0] for v in per_judge.values()])
        is_correct_fallback[idx] = result
        justification_fallback[idx] = per_judge[judge_models[0]][1]
        judge_agreement_fallback[idx] = agreement
        judge_disagreement_flag_fallback[idx] = (
            agreement is not None and agreement < 1.0
        )
        for jm, (verdict, just) in per_judge.items():
            per_judge_correct_fallback[jm][idx] = verdict
            per_judge_justification_fallback[jm][idx] = just
        df.loc[idx, "score_method"] = (
            SCORE_METHOD_LLM_FALLBACK if result is not None else SCORE_METHOD_UNSCORED
        )

    _write_fallback_columns()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    update_log(
        language_dir,
        model_name,
        {
            "responses_judged_fallback_timestamp_utc": datetime.now(UTC).isoformat(),
            "responses_judged_fallback_git_hash": git_hash(),
            "mcq_llm_judges_fallback": judge_models,
        },
    )
    print(f"Fallback results saved to {output_path}")
    return df


def _agreement_stats(
    judged_df: pd.DataFrame, agreement_col: str, flag_col: str
) -> tuple[float | None, int]:
    """(mean agreement, disagreement count) for one judging pass's columns.

    The mean is None when the column is absent or has no scored row: a pass run
    with a single judge records nothing, and 0.0 there would read as total
    disagreement rather than "not measured".
    """
    mean: float | None = None
    if agreement_col in judged_df.columns:
        scored = judged_df[agreement_col].dropna()
        if len(scored):
            mean = round(scored.mean(), 4)
    count = int(judged_df[flag_col].sum()) if flag_col in judged_df.columns else 0
    return mean, count


def compute_summary(judged_df: pd.DataFrame, model_name: str) -> dict[str, object]:
    """Compute summary statistics from judged results.

    When ``is_correct_fallback`` is present (from ``--judge-fallback``),
    accuracy is computed from the best-available result: fallback where
    populated, else the original ``is_correct``.

    Every accuracy is ``None`` when it has no scored row to average over, rather
    than 0.0, so an unmeasured score cannot be read as a measured 0%.

    Judge agreement is reported once per pass and never combined, because the two
    passes ask different questions about different rows.
    ``short_form_judge_agreement_mean`` is the ensemble deciding whether a
    free-text answer means the same as the expected one;
    ``mcq_judge_fallback_agreement_mean`` is the (usually single-judge) pass
    deciding which letter an unparseable MCQ answer pointed at. A single average
    over both would describe neither.

    The ``mcq_judge_fallback_*`` keys appear only when ``judged_df`` comes from a
    fallback pass, so a standard pass's summary has no fallback column.

    Args:
        judged_df: DataFrame with is_correct column (and optionally
            is_correct_fallback)
        model_name: Model name for the summary

    Returns:
        Dict with model, accuracy, per-type accuracy, and the short_form judge
        stats -- plus the mcq fallback ones for a fallback frame
    """
    # judge_results_fallback() writes the primary CSV plus its own columns, so this
    # is what distinguishes the two passes' frames.
    is_fallback_frame = "is_correct_fallback" in judged_df.columns

    # Use fallback where available, else original is_correct
    if is_fallback_frame:
        effective = judged_df["is_correct_fallback"].fillna(judged_df["is_correct"])
    else:
        effective = judged_df["is_correct"]

    scored_mask = effective.notna()
    scored_df = effective[scored_mask]
    total_scored = len(scored_df)
    correct = scored_df.sum()
    # None, not 0.0, when nothing was scored: an unmeasured accuracy is not the same
    # observation as a measured 0%, and the two are indistinguishable once written to
    # the summary CSV. Consumers read the empty cell as missing data.
    accuracy = round(correct / total_scored * 100, 2) if total_scored else None

    by_type: dict[str, float | None] = {}
    for atype in judged_df["answer_type"].unique():
        type_mask = judged_df["answer_type"] == atype
        subset = effective[type_mask]
        subset_scored = subset[subset.notna()]
        if len(subset_scored) == 0:
            by_type[atype] = None
        else:
            by_type[atype] = round(subset_scored.sum() / len(subset_scored) * 100, 2)

    # Short_form ensemble (primary pass) and MCQ fallback pass, kept apart -- see
    # this function's docstring for why they are not pooled into one figure.
    short_form_agreement, short_form_disagreements = _agreement_stats(
        judged_df, "judge_agreement", "judge_disagreement_flag"
    )

    summary: dict[str, object] = {
        "model": model_name,
        "accuracy": accuracy,
        "scored_count": total_scored,
        "unscored_count": int((~scored_mask).sum()),
        # No default: an answer_type absent from the dataset is also unmeasured.
        "accuracy_mcq": by_type.get("multiple_choice"),
        "accuracy_short_form": by_type.get("short_form"),
        "short_form_judge_agreement_mean": short_form_agreement,
        "short_form_judge_disagreement_count": short_form_disagreements,
    }

    if is_fallback_frame:
        mcq_agreement, mcq_disagreements = _agreement_stats(
            judged_df, "judge_agreement_fallback", "judge_disagreement_flag_fallback"
        )
        summary["mcq_judge_fallback_agreement_mean"] = mcq_agreement
        summary["mcq_judge_fallback_disagreement_count"] = mcq_disagreements

    return summary


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


def compute_judge_agreement(
    judged_df: pd.DataFrame, judge_models: list[str], *, fallback: bool = False
) -> pd.DataFrame:
    """Pairwise agreement rate between judges, over rows both judges scored.

    Args:
        judged_df: a judge_results()/judge_results_fallback() output.
        judge_models: the judges to compare (their is_correct_<judge>[_fallback]
            columns must be present).
        fallback: read the `is_correct_fallback_<judge>` columns instead of
            `is_correct_<judge>` (i.e. the columns judge_results_fallback() adds).

    Returns:
        A judge x judge DataFrame (diagonal 1.0 wherever a judge scored anything);
        NaN where two judges never scored a common row.
    """
    prefix = "is_correct_fallback_" if fallback else "is_correct_"
    cols = {jm: f"{prefix}{jm}" for jm in judge_models}
    missing = [jm for jm in judge_models if cols[jm] not in judged_df.columns]
    if missing:
        raise KeyError(f"judged_df is missing columns for judge(s): {missing}")

    rate = pd.DataFrame(index=judge_models, columns=judge_models, dtype=float)
    for a in judge_models:
        for b in judge_models:
            both = judged_df[cols[a]].notna() & judged_df[cols[b]].notna()
            if both.sum() == 0:
                rate.loc[a, b] = float("nan")
                continue
            rate.loc[a, b] = (
                judged_df.loc[both, cols[a]] == judged_df.loc[both, cols[b]]
            ).mean()
    return rate

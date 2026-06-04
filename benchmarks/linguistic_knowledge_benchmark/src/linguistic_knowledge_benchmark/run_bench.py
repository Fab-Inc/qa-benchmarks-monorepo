"""
Linguistic Knowledge Eval Module
Evaluates LLM performance on language knowledge questions.
Supports multiple_choice, short_form, and true_false answer types.
"""

import json
import re
import time
from pathlib import Path

import pandas as pd
from fdllm import get_caller
from fdllm.llmtypes import LLMMessage
from tqdm import tqdm

from linguistic_knowledge_benchmark.shared_utils import (
    EVAL_MODELS,
    RATE_LIMITED_MODELS,
    ROOT_DIR,
)

CACHE_BASE_DIR = ROOT_DIR / "cache" / "cache_linguistic_knowledge"


# Answer parsing patterns (the LLPK MCQ letter extractor; the only LLPK code
# reused by Linguistic Knowledge — judge_bench imports parse_answer from here).
REPAT = [
    r"^\s*([ABCD])(?:[\.(?:\s*\n)]+.*)*$",
    r"^\s*\(([ABCD])\)",  # letter in parentheses, e.g. (C)
    r"^<think>[\s\S]*?</think>[\s\S]*?([ABCD])$",  # for deepseek R1
    r"^## Step 1[\s\S]*?([ABCD])[\.\s]*$",  # for llama 4
    r'[\s\S]*\n([A-D])"?$',  # for claude 4
    # for glm 4.1v-9b-thinking
    r"<answer>(?:<\|begin_of_box\|>)?([ABCD])(?:<\|end_of_box\|>)?$",
]
REQ = [re.compile(pat) for pat in REPAT]


def parse_answer(resp: str) -> str | None:
    """
    Parse model response to extract the answer letter (A-D).
    Returns None if no valid answer found.
    """
    if pd.isna(resp) or resp is None:
        return None

    resp = str(resp).strip()

    for req in REQ:
        match = req.match(resp)
        if match is not None:
            return match.groups()[0]

    return None


def _build_system_prompts(language_name: str) -> dict[str, str]:
    """Return {answer_type: prompt}.

    The medium of evaluation is English for every language: questions are written
    in English and ask about the linguistics of ``language_name``. Every prompt
    is prefixed with that context so the model knows which language the question
    is about, regardless of answer type.
    """
    prefix = (
        "The following question is written in English and concerns the"
        f" linguistics of {language_name}. "
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
        "true_false": (
            prefix + "Answer the following question with exactly 'True' or 'False'."
            "\nDo not explain."
        ),
    }


def load_dataset(dataset_path: Path) -> list[dict]:
    """Load a Linguistic Knowledge evaluation dataset from a JSON file.

    Args:
        dataset_path: Path to the dataset JSON file.

    Returns:
        List of question dicts with keys: id, category, question,
        answer_type, expected_answers, rubric_id, etc.
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    with open(dataset_path, encoding="utf-8") as f:
        return json.load(f)


def run_benchmark(
    model_name: str,
    *,
    language_name: str,
    dataset_path: Path,
    force: bool = False,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Run the Linguistic Knowledge benchmark for a given model.

    Args:
        model_name: Model name (must match key in EVAL_MODELS / custom_models.yaml)
        language_name: Display name of the language (e.g. "Luganda", "Yoruba"),
            interpolated into the system prompts.
        dataset_path: Path to the dataset JSON file.
        force: Force re-run even if cached results exist
        output_dir: Directory to save results (default:
            cache/cache_linguistic_knowledge/cache_{language_name.lower()}/answers)
    Returns:
        DataFrame with columns: id, category, answer_type, expected_answers,
        output, latency, total_tokens, completion_tokens, reasoning_tokens
    """
    if output_dir is None:
        output_dir = CACHE_BASE_DIR / f"cache_{language_name.lower()}" / "answers"
    output_dir = Path(output_dir)

    # Validate model name
    if model_name not in EVAL_MODELS:
        print(f"ERROR: Model '{model_name}' not found in list_models.yaml.")
        print(f"Available models: {', '.join(EVAL_MODELS.keys())}")
        raise ValueError(f"Invalid model: {model_name}")

    # Check cache
    cache_path = output_dir / f"resps_{model_name}.csv"
    if not force and cache_path.exists():
        print(f"Results already exist for {model_name}. Use --force to re-run.")
        return pd.read_csv(cache_path)

    # Load dataset and build prompts
    questions = load_dataset(dataset_path)
    system_prompts = _build_system_prompts(language_name)

    # Create caller
    try:
        caller = get_caller(model_name)
    except NotImplementedError:
        raise ValueError(
            f"Model '{model_name}' is in list_models.yaml but not "
            f"registered in custom_models.yaml."
        ) from None

    # Run benchmark
    results = []
    for question in tqdm(questions, desc=f"Running {model_name}"):
        system_prompt = system_prompts[question["answer_type"]]
        try:
            messages = [
                LLMMessage(Role="system", Message=system_prompt),
                LLMMessage(Role="user", Message=question["question"]),
            ]
            response = caller.call(messages, max_tokens=None)

            results.append(
                {
                    "id": question["id"],
                    "category": question["category"],
                    "answer_type": question["answer_type"],
                    "expected_answers": "|".join(question["expected_answers"]),
                    "output": response.Message,
                    "latency": response.Latency,
                    "total_tokens": response.TokensUsed,
                    "completion_tokens": response.TokensUsedCompletion,
                    "reasoning_tokens": response.TokensUsedReasoning,
                }
            )
        except Exception as e:
            print(f"\nERROR on question {question['id']}: {e}")
            results.append(
                {
                    "id": question["id"],
                    "category": question["category"],
                    "answer_type": question["answer_type"],
                    "expected_answers": "|".join(question["expected_answers"]),
                    "output": "",
                    "latency": None,
                    "total_tokens": None,
                    "completion_tokens": None,
                    "reasoning_tokens": None,
                }
            )

        if model_name in RATE_LIMITED_MODELS:
            time.sleep(5)

    results_df = pd.DataFrame(results)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / f"resps_{model_name}.csv"
    results_df.to_csv(cache_path, index=False)
    print(f"Results saved to {cache_path}")
    return results_df

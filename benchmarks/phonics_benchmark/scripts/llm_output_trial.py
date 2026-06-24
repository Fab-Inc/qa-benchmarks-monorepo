import json
import re
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from fdllm import register_models
from phonics_benchmark.data_io import read_decodable_texts, read_students
from phonics_benchmark.llm import call_llm, get_prompt
from phonics_benchmark.models import (
    BenchmarkItem,
    DecodableText,
    StudentKnowledgeProfile,
)

OUTPUTS_DIR = Path(__file__).parents[1] / "outputs"
REPO_ROOT = Path(__file__).parents[3]

load_dotenv(REPO_ROOT / ".env")
register_models(REPO_ROOT / "custom_models.yaml")

# temporary subset to experiment with
models = [
    "gemini-3.1-pro-preview",
    "claude-sonnet-4-6",
    # "claude-haiku-4-5-20251001",
    # "gemini-2.5-flash-preview-09-2025",
    # "deepseek-v3-0324-fp8"
]

language = "kiswahili"  # hardcoded for now, will be replaced with CLI-supplied value

timestamp: str = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
results_path = OUTPUTS_DIR / f"{timestamp}_results_trial.jsonl"

decodable_texts = read_decodable_texts(language)
student_knowledge_profiles = read_students(language)


def strip_non_alpha(word: str) -> str:
    """Strip non-alphabetic characters from the start and end of a word.

    Alphabetic characters within the word are preserved, and ``str.isalpha``
    is Unicode-aware so this works for any alphabet/language.
    """
    start, end = 0, len(word)
    while start < end and not word[start].isalpha():
        start += 1
    while end > start and not word[end - 1].isalpha():
        end -= 1
    return word[start:end]


def parse_response(response: str) -> list[str]:
    """Split response word list by commas, then remove all remaining whitespace
    and special characters to leave pure alphabetic words.
    """
    # TODO: we probably want to handle this more scientifically in longer term
    # by evaluating and logging whether the response met the expected schema,
    # so that invalid responses are more transparent and don't get ignored.
    # For now just brute forcing everything into a list of words.

    if response == "[]" or response == "":
        return []

    if re.search(r"\ball\b", response, re.IGNORECASE):
        print("WARNING: model response contains the word 'all' so may be misformatted")

    words_raw = [word.strip() for word in response.split(",")]

    # WARNING: this will fail if the word is legitimately supposed to begin
    # or end with punctuation, e.g. apostrophe could be problematic
    words_stripped = [strip_non_alpha(word) for word in words_raw]

    # remove any empty strings
    words_filtered = [w for w in words_stripped if w]

    return words_filtered


benchmark_items = []
for text in decodable_texts:
    for profile in student_knowledge_profiles:
        benchmark_items.append(
            {
                "language": language,
                "student_knowledge_profile": profile,
                "decodable_text": text,
            }
        )

for model in models:
    print(f"Running decodability benchmark for model: {model}")
    for benchmark_item in benchmark_items:
        text: DecodableText = benchmark_item["decodable_text"]
        profile: StudentKnowledgeProfile = benchmark_item["student_knowledge_profile"]

        prompt = get_prompt(
            "decodability_evaluation",
            taught_knowledge=profile.taught_knowledge,
            text=text.content,
            language=language,
        )

        response_raw = call_llm(
            prompt,
            model=model,
            max_tokens=10000,
            tag=f"{text.text_id}_{profile.student_knowledge_id}",
        )

        # TODO: log LLM API config params (even defaults) if possible

        print(f"Text ID: {text.text_id}, Knowledge ID: {profile.student_knowledge_id}")
        # print(f"Prompt: {prompt}")
        # print(f"LLM Raw Response: {response_raw}")
        # print("-" * 80)

        response_parsed_words = parse_response(response_raw)

        print(f"Parsed response words: {response_parsed_words}")

        # TODO: mark against reference correct answer either from data or primerpro call

        result = {
            "model_name": model,
            "benchmark_item_id": "placeholder",
            "response_raw": response_raw,
            "response_parsed_words": response_parsed_words,
            "score": 999,
        }

        with results_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        # print("\n\n\n\n")

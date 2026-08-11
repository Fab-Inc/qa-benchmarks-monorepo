#!/usr/bin/env python3
"""
Smoke-test every model in benchmark_default_models_display_names.yaml with a single
throwaway question, via fdllm. Meant to be run before a full run-bench sweep over that
(much larger) model list, to flag models that are no longer supported/registered or
whose credentials/endpoint are currently broken -- without burning a full benchmark run
on them.

Prints one line per model as it's tested, then a final list of the models that errored.

Run:
    python benchmarks/linguistic_knowledge_benchmark/scripts/workflow/test_supported_models.py
    python .../test_supported_models.py --models-file path/to/other_models.yaml
    python .../test_supported_models.py --question "What is 2+2?"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from dotenv import load_dotenv
from fab_benchmark_publisher import download_file
from fdllm import get_caller, register_models
from fdllm.llmtypes import LLMMessage

BENCH_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = BENCH_DIR.parents[1]
DEFAULT_MODELS_FILE = BENCH_DIR / "list_models.yaml"

TEST_QUESTION = "What is the color of the sky?"


def setup_models() -> None:
    """Load API keys and register custom models, mirroring cmd.py's app callback."""
    load_dotenv(REPO_ROOT / ".env")
    download_file(
        bucket_name="fab-open-configs",
        file_name="custom_models.yaml",
        write_path=REPO_ROOT / "custom_models.yaml",
        require_auth=False,
    )
    register_models(str(REPO_ROOT / "custom_models.yaml"))


def load_model_names(models_file: Path) -> list[str]:
    with open(models_file, encoding="utf-8") as f:
        return list(yaml.safe_load(f))


def test_model(model_name: str, question: str) -> str | None:
    """Return None on success, or an error message string on failure."""
    try:
        caller = get_caller(model_name)
        messages = [LLMMessage(Role="user", Message=question)]
        caller.call(messages, max_tokens=None)
    except Exception as e:
        return f"{type(e).__name__}: {e}"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-file",
        type=Path,
        default=DEFAULT_MODELS_FILE,
        help=f"YAML file of model_name: display_name (default: {DEFAULT_MODELS_FILE})",
    )
    parser.add_argument(
        "--question",
        default=TEST_QUESTION,
        help=f"Test question to send to each model (default: {TEST_QUESTION!r})",
    )
    args = parser.parse_args()

    model_names = load_model_names(args.models_file)
    print(f"Testing {len(model_names)} models from {args.models_file}...\n")

    setup_models()

    failed: dict[str, str] = {}
    for i, model_name in enumerate(model_names, 1):
        error = test_model(model_name, args.question)
        if error is None:
            print(f"[{i}/{len(model_names)}] OK    {model_name}")
        else:
            print(f"[{i}/{len(model_names)}] ERROR {model_name}: {error}")
            failed[model_name] = error

    print(f"\n{len(model_names) - len(failed)}/{len(model_names)} models working.")
    if failed:
        print(f"\n{len(failed)} model(s) failed:")
        for model_name, error in failed.items():
            print(f"  - {model_name}: {error}")
    else:
        print("\nAll models working.")


if __name__ == "__main__":
    main()

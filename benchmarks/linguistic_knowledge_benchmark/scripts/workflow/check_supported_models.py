#!/usr/bin/env python3
"""
Smoke-test every model in benchmark_default_models_display_names.yaml with a single
throwaway question, via fdllm. Meant to be run before a full run-bench sweep over that
(much larger) model list, to flag models that are no longer supported/registered or
whose credentials/endpoint are currently broken -- without burning a full benchmark experiment
on them.

Prints one line per model as it's tested, then a final list of the models that errored,
then those failures GROUPED BY CATEGORY -- because the raw list of ~200 exception strings
is not the analysis: what you need before committing to a sweep is whether a failure means
the model is gone ("not deployed"), never was registered ("not registered"), needs a key
you don't have ("missing API key"), or was merely transient ("rate limited",
"connection error") and worth simply retrying.

Two optional artifacts make that reviewable outside the terminal:
  --report out.csv     one row per model: status, category, raw exception text. Sort or
                       filter it to pick the models worth retrying vs dropping.
  --annotate out.yaml  a copy of --models-file with every failing model commented out and
                       its category appended as a trailing comment, in the same style as
                       the hand-maintained list_models.yaml -- diff it against that file,
                       or use it directly as the sweep's model list. Never edits
                       --models-file in place.

Run:
    python benchmarks/linguistic_knowledge_benchmark/scripts/workflow/check_supported_models.py
    python .../check_supported_models.py --models-file path/to/other_models.yaml
    python .../check_supported_models.py --question "What is 2+2?"
    python .../check_supported_models.py --report check.csv --annotate list_models_checked.yaml
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import yaml
from dotenv import load_dotenv
from fab_benchmark_publisher import download_file
from fdllm import get_caller, register_models
from fdllm.llmtypes import LLMMessage

BENCH_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = BENCH_DIR.parents[1]
DEFAULT_MODELS_FILE = BENCH_DIR / "list_models_full.yaml"

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


def load_display_names(models_file: Path) -> dict[str, str]:
    with open(models_file, encoding="utf-8") as f:
        return {k: str(v) for k, v in (yaml.safe_load(f) or {}).items()}


# Error text -> the short reason to write next to the model in the YAML. Ordered: the
# first pattern that matches wins, so the specific ones come before the generic ones --
# in particular "no endpoints found for X", a deprecation notice and a paused HF endpoint
# all carry an HTTP 404/400 too, so they must be matched before the generic code-based
# patterns below them. The reason strings deliberately match the ones already used by
# hand in list_models.yaml ("not registered", "missing API key", "not deployed", ...) so
# an annotated file reads the same as the hand-maintained one.
# HTTP codes are matched with \b: without it, "403" matches inside a date-stamped model
# name like claude-3-haiku-20240307 and mislabels a plain unregistered model.
ERROR_CATEGORIES: tuple[tuple[str, str], ...] = (
    (
        "not registered",
        r"not a recognised model name|not registered|unknown model|no caller",
    ),
    ("missing API key", r"does not have api_key|api[_ -]?key|apikey|credential"),
    ("endpoint paused", r"endpoint is paused"),
    ("invalid model id", r"is not a valid model id"),
    ("deprecated", r"deprecat|no longer available|period has ended"),
    ("no endpoints", r"no endpoints? found|no endpoint|no provider"),
    ("rate limited", r"rate.?limit|quota|\b429\b|resource.?exhausted|overloaded"),
    (
        "not deployed",
        r"not deployed|model.?not.?found|does not exist|no such model"
        r"|not found the model|not_found|\b404\b",
    ),
    ("missing credentials", r"unauthor|forbidden|\b401\b|\b403\b"),
    (
        "connection error",
        r"timeout|timed out|connection|\bssl\b|\bdns\b|\b502\b|\b503\b|\b504\b",
    ),
    ("bad request", r"invalid|unsupported|\b400\b|max_tokens|context length"),
)

# Providers that name their replacement in the 404 body; captured so the reason can say
# what to migrate to, which is the whole point of noticing a deprecation.
SUCCESSOR_PATTERNS: tuple[str, ...] = (
    r"use models/([A-Za-z0-9.\-]+)",
    r"switching to ([A-Za-z0-9.\- ]+?) \(http",
    r"migrate to (?:the paid slug: )?([A-Za-z0-9/.\-]+)",
)


def classify_error(error: str) -> str:
    """Bucket a raw exception string into one of ERROR_CATEGORIES, else "other error".

    The point is triage, not precision: a run over the full list produces ~200 raw
    exception strings, and what you actually need to decide is "is this model gone,
    is a key missing, or was it just a transient failure worth retrying?".
    """
    for category, pattern in ERROR_CATEGORIES:
        if re.search(pattern, error, flags=re.IGNORECASE):
            return category
    return "other error"


def failure_reason(error: str) -> str:
    """The category, plus the replacement model when the provider names one -- e.g.
    "deprecated -> grok-4.3". This is what gets written as the YAML trailing comment."""
    category = classify_error(error)
    if category in ("deprecated", "not deployed"):
        for pattern in SUCCESSOR_PATTERNS:
            match = re.search(pattern, error, flags=re.IGNORECASE)
            if match:
                return f"{category} -> {match.group(1).strip()}"
    return category


def write_report(
    path: Path,
    model_names: list[str],
    display_names: dict[str, str],
    failed: dict[str, str],
) -> None:
    """One row per tested model: status, error category and the raw exception text."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "display_name", "status", "category", "error"])
        for model_name in model_names:
            error = failed.get(model_name)
            writer.writerow(
                [
                    model_name,
                    display_names.get(model_name, ""),
                    "error" if error else "ok",
                    failure_reason(error) if error else "",
                    error or "",
                ]
            )


def print_failure_breakdown(failed: dict[str, str]) -> None:
    """The failures grouped by category -- the actual analysis step: one glance tells
    you whether a bad run is 3 dead models or one missing API key hitting 20 of them."""
    if not failed:
        return
    by_category: dict[str, list[str]] = {}
    for model_name, error in failed.items():
        by_category.setdefault(classify_error(error), []).append(model_name)
    print("\nFailures by category:")
    for category, models in sorted(by_category.items(), key=lambda kv: -len(kv[1])):
        print(f"\n  {category} ({len(models)}):")
        for model_name in models:
            print(f"    - {model_name}: {failed[model_name][:120]}")


def write_annotated_models_file(
    path: Path,
    models_file: Path,
    failed: dict[str, str],
) -> None:
    """Copy `models_file` with every failing model commented out and its reason appended,
    ready to be diffed against / copied over the hand-maintained list. Lines that were
    already commented out, and models that passed, are left exactly as they are."""
    lines = models_file.read_text(encoding="utf-8").splitlines(keepends=True)
    out = []
    for line in lines:
        stripped = line.lstrip()
        if stripped and not stripped.startswith("#") and ":" in stripped:
            model_name = stripped.split(":", 1)[0].strip()
            if model_name in failed:
                body = line.rstrip("\r\n")
                eol = line[len(body) :]
                out.append(f"#{body} # {failure_reason(failed[model_name])}{eol}")
                continue
        out.append(line)
    path.write_text("".join(out), encoding="utf-8")


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
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="write a per-model CSV (model, display_name, status, category, error) here",
    )
    parser.add_argument(
        "--annotate",
        type=Path,
        default=None,
        help="write a copy of --models-file with failing models commented out and the "
        "failure reason appended (never edits --models-file itself)",
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
        print_failure_breakdown(failed)
    else:
        print("\nAll models working.")

    if args.report:
        write_report(
            args.report,
            model_names,
            load_display_names(args.models_file),
            failed,
        )
        print(f"\nWrote {args.report}")
    if args.annotate:
        write_annotated_models_file(args.annotate, args.models_file, failed)
        print(f"Wrote {args.annotate}")


if __name__ == "__main__":
    main()

"""Shared test fixtures and fakes for the benchmark test suite."""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

# Under --import-mode=importlib (set in the workspace pyproject), pytest does not
# add the tests dir to sys.path, so make `from conftest import ...` resolvable.
sys.path.insert(0, os.path.dirname(__file__))

# A model id that exists in list_models.yaml (so EVAL_MODELS validation passes).
VALID_MODEL = "claude-haiku-4-5-20251001"


def make_response(message: str = "A") -> SimpleNamespace:
    """A stand-in for an fdllm caller response object."""
    return SimpleNamespace(
        Message=message,
        Latency=0.1,
        TokensUsed=10,
        TokensUsedCompletion=5,
        TokensUsedReasoning=2,
    )


class FakeCaller:
    """Records calls and returns a canned response (or raises)."""

    def __init__(self, response=None, raise_exc: Exception | None = None):
        self._response = response if response is not None else make_response()
        self._raise = raise_exc
        self.calls: list = []

    def call(self, messages, max_tokens=None, response_schema=None):
        self.calls.append(messages)
        if self._raise is not None:
            raise self._raise
        return self._response


def write_dataset(path: Path, questions: list[dict]) -> Path:
    """Write a dataset JSON file and return its path."""
    path.write_text(json.dumps(questions), encoding="utf-8")
    return path


def sample_questions() -> list[dict]:
    """One question per supported answer_type."""
    return [
        {
            "id": 1,
            "category": "morphology",
            "question": "Pick the right letter.",
            "answer_type": "multiple_choice",
            "expected_answers": ["B"],
        },
        {
            "id": 2,
            "category": "vocabulary",
            "question": "Translate 'thing'.",
            "answer_type": "short_form",
            "expected_answers": ["kintu"],
        },
        {
            "id": 3,
            "category": "grammar",
            "question": "Is this true?",
            "answer_type": "true_false",
            "expected_answers": ["True"],
        },
    ]

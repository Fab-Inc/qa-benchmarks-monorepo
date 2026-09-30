"""Shared test fixtures and fakes for the benchmark test suite."""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

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


@pytest.fixture(autouse=True)
def offline_app_callback(monkeypatch, request):
    """Keep the test suite off the network and independent of the synced config files.

    cmd.py's app callback syncs custom_models.yaml + models.csv + providers.csv from GCS
    and then registers the models from the YAML. Without this fixture every CliRunner
    test would make three real downloads and overwrite the developer's own copies.

    register_models is stubbed alongside the download for the same reason: those files
    are gitignored, so on a fresh checkout (and in CI) they do not exist until something
    fetches them -- and with the download stubbed, register_models would raise on the
    missing YAML. Nothing here needs a real fdllm registry, since the tests mock
    run_benchmark/judge_results; model-name validation reads EVAL_MODELS from the
    tracked list_models.yaml, not from custom_models.yaml.

    Tests that exercise the sync itself opt out by patching `cmd.download_file`
    directly, and the `integration` marker disables the stubs entirely so the opt-in
    bucket test can reach the network.
    """
    if request.node.get_closest_marker("integration"):
        return
    try:
        from linguistic_knowledge_benchmark import cmd
    except ImportError:  # tests that don't import the CLI at all
        return
    monkeypatch.setattr(cmd, "_download_shared_configs", lambda: None)
    monkeypatch.setattr(cmd, "register_models", lambda _path: None)


def write_dataset(path: Path, questions: list[dict]) -> Path:
    """Write a dataset JSON file and return its path."""
    path.write_text(json.dumps(questions), encoding="utf-8")
    return path


def sample_questions() -> list[dict]:
    """One question per supported answer_type, using the native JSON field names."""
    return [
        {
            "question_id": 1,
            "domain": "morphology",
            "question": "Pick the right letter.",
            "answer_type": "multiple_choice",
            "options": {"A": "opt A", "B": "opt B", "C": "opt C", "D": "opt D"},
            "expected_answers": ["B"],
        },
        {
            "question_id": 2,
            "domain": "vocabulary",
            "question": "Translate 'thing'.",
            "answer_type": "short_form",
            "expected_answers": ["kintu"],
        },
    ]

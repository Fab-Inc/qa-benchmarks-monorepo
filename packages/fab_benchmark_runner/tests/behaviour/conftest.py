"""The harness behind the behaviour tests: a bench with fake models.

The behaviour tests describe what the runner is meant to do, through the same
entry points people use: ``create_experiment``, ``raise_max_tokens`` and a job
running a run config (``run_job``/``plan_job``, which ``fab-bench run`` calls).
Only the model calls are faked, by ``FakeModels``, and GCS by an in-memory fake.
Tests about what FDLLM is handed run through FDLLM instead, with only the providers
behind it faked, by ``FakeProviders``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import fdllm
import pytest
from fdllm.errors import EmptyLLMResponse
from mistralai.exceptions import MistralConnectionException
from PIL import Image

from fab_benchmark_runner import (
    ModelInput,
    RunPlan,
    RunResult,
    create_experiment,
    parse_run_config,
    plan_job,
    run_job,
)
from fab_benchmark_runner import jobs as jobs_module
from fab_benchmark_runner.gcs import parse_gs_uri

TEXT_INPUTS = (
    {
        "benchmark_item_id": "mcq-1",
        "prompt": "Which word is a noun? A) run B) cat",
        "system_prompt": "Answer with one letter.",
    },
    {
        "benchmark_item_id": "short-1",
        "prompt": "Name the noun in: The cat sat.",
        "system_prompt": "Answer with one word.",
    },
    {"benchmark_item_id": "open-1", "prompt": "Write a sentence about a cat."},
)
MODELS = ("model-a", "model-b", "model-c")


def png(color: str = "red", size: tuple[int, int] = (40, 30)) -> bytes:
    """Return a PNG file of one color."""

    buffer = BytesIO()
    Image.new("RGB", size, color=color).save(buffer, format="PNG")
    return buffer.getvalue()


def color_of(picture: Image.Image) -> tuple[int, int, int]:
    """Return the color of an image's top-left pixel."""

    return picture.convert("RGB").getpixel((0, 0))


Behaviour = Callable[[ModelInput, "int | None"], Mapping[str, Any]]


class FakeModels:
    """Stands in for FDLLM: answers every call and records it.

    By default a model answers ``"<model> answers <item>"``. Tests make particular
    calls fail, crash, or depend on the token limit.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int | None]] = []
        self.inputs: list[ModelInput] = []
        self._behaviours: dict[tuple[str | None, str], Behaviour] = {}
        self.before_call: Callable[[str, ModelInput], None] | None = None

    def __call__(
        self, model: str, model_input: ModelInput, max_tokens: int | None
    ) -> Mapping[str, Any]:
        item_id = model_input.benchmark_item_id
        if self.before_call is not None:
            self.before_call(model, model_input)
        self.calls.append((model, item_id, max_tokens))
        self.inputs.append(model_input)
        behaviour = self._behaviours.get((model, item_id)) or self._behaviours.get(
            (None, item_id)
        )
        if behaviour is not None:
            return behaviour(model_input, max_tokens)
        return {"output": f"{model} answers {item_id}"}

    def called(self, model: str | None = None) -> list[str]:
        """Return the items called, in order, for one model or all of them."""

        return [item for name, item, _ in self.calls if model in (None, name)]

    def forget_calls(self) -> None:
        self.calls.clear()
        self.inputs.clear()

    def fail(self, item_id: str, *, model: str | None = None) -> None:
        """The provider fails this call, as with an outage or a rejected request."""

        def behave(_input: ModelInput, _max_tokens: int | None) -> Mapping[str, Any]:
            raise MistralConnectionException("provider unavailable")

        self._behaviours[(model, item_id)] = behave

    def crash(self, item_id: str, *, model: str | None = None) -> None:
        """Something unexpected breaks, as a bug in the runner would."""

        def behave(_input: ModelInput, _max_tokens: int | None) -> Mapping[str, Any]:
            raise RuntimeError("unexpected failure")

        self._behaviours[(model, item_id)] = behave

    def recover(self, item_id: str, *, model: str | None = None) -> None:
        self._behaviours.pop((model, item_id), None)

    def need_tokens(
        self, item_id: str, tokens: int, *, empty_below: str = "stop_reason"
    ) -> None:
        """This item's answer needs ``tokens`` tokens.

        Below that the response is lost to the limit: with ``"stop_reason"`` FDLLM
        reports it, as its dev branch does; with ``"empty"`` the text is just empty,
        as on FDLLM's main branch.
        """

        def behave(_input: ModelInput, max_tokens: int | None) -> Mapping[str, Any]:
            if max_tokens is not None and max_tokens < tokens:
                if empty_below == "stop_reason":
                    raise EmptyLLMResponse("Empty response", stop_reason="length")
                return {"output": ""}
            return {"output": f"a {tokens}-token answer"}

        self._behaviours[(None, item_id)] = behave

    def cut_off(self, item_id: str, tokens: int) -> None:
        """Below ``tokens`` tokens this item's answer stops mid-sentence."""

        def behave(_input: ModelInput, max_tokens: int | None) -> Mapping[str, Any]:
            if max_tokens is not None and max_tokens < tokens:
                return {"output": "The answer is"}
            return {"output": "The answer is B."}

        self._behaviours[(None, item_id)] = behave


@dataclass
class ProviderRequest:
    """One request as FDLLM's caller received it: its messages and limit."""

    model: str
    messages: list[Any]
    max_tokens: int | None

    @property
    def user_message(self) -> Any:
        return self.messages[-1]


class FakeProviders:
    """Stands in for the providers behind FDLLM, so a run goes through FDLLM.

    The runner builds FDLLM's messages and images as for a real call; each request
    is recorded as FDLLM's caller received it, and answered. Every model takes
    images unless a test sets ``vision[model] = False``, as for a registry entry
    without ``Vision: true``.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.vision: dict[str, bool] = {}
        self.requests: list[ProviderRequest] = []
        monkeypatch.setattr(jobs_module, "prepare_fdllm", self._prepare)
        monkeypatch.setattr(fdllm, "get_caller", self._caller)

    def request(self, prompt: str, model: str = "model-a") -> ProviderRequest:
        [request] = [
            r
            for r in self.requests
            if r.model == model and r.user_message.Message == prompt
        ]
        return request

    def _prepare(
        self, _registry: Any, models: Iterable[str], *, env_path: Any
    ) -> dict[str, Any]:
        return {
            model: {"Api_Interface": "OpenAI", "Vision": self.vision.get(model, True)}
            for model in models
        }

    def _caller(self, model: str) -> Any:
        providers = self

        class Caller:
            def call(self, messages: list[Any], max_tokens: int | None) -> Any:
                providers.requests.append(
                    ProviderRequest(model, list(messages), max_tokens)
                )
                return SimpleNamespace(Message=f"{model} answers", ResponseModel=model)

        return Caller()


class Bench:
    """A working folder with a model registry, fake models and a fake GCS."""

    def __init__(self, root: Path, storage: Any) -> None:
        self.root = root
        self.storage = storage
        self.models = FakeModels()
        self.registry = root / "custom_models.yaml"
        self.registry.write_text(
            json.dumps({f"provider-{m}": {"models": {m: {}}} for m in MODELS}),
            encoding="utf-8",
        )

    def write_inputs(
        self,
        inputs: Iterable[Mapping[str, Any]] = TEXT_INPUTS,
        *,
        name: str = "inputs.jsonl",
    ) -> Path:
        path = self.root / name
        path.write_text(
            "".join(json.dumps(dict(row)) + "\n" for row in inputs), encoding="utf-8"
        )
        return path

    def create(
        self,
        inputs: Iterable[Mapping[str, Any]] = TEXT_INPUTS,
        *,
        dest: str = "experiment",
        max_tokens: int | None = 1000,
        benchmark: str = "lk",
        **options: Any,
    ) -> str:
        """Create an experiment from an inputs file; return where it is."""

        location = dest if dest.startswith("gs://") else str(self.root / dest)
        create_experiment(
            self.write_inputs(inputs),
            dest=location,
            benchmark=benchmark,
            max_tokens=max_tokens,
            storage_client=self.storage,
            **options,
        )
        return location

    def config(self, *models: str, experiment: str | None = None, **settings: Any):
        raw = {
            "experiment": experiment or str(self.root / "experiment"),
            "models": list(models or ("model-a",)),
            "custom_models_path": str(self.registry),
            "model_workers": 1,
            **settings,
        }
        return parse_run_config(raw, base_dir=self.root, source="run.json")

    def run(
        self,
        *models: str,
        experiment: str | None = None,
        job: str = "job",
        through_fdllm: bool = False,
        **settings: Any,
    ) -> RunResult:
        """Run models on an experiment as one job; ``job`` names its workdir.

        Models are ``FakeModels`` unless ``through_fdllm``, which needs the
        ``providers`` fixture.
        """

        return run_job(
            self.config(*models, experiment=experiment, **settings),
            workdir=self.root / job,
            download_public_registry=False,
            call_model=None if through_fdllm else self.models,
            storage_client=self.storage,
        )

    def plan(
        self,
        *models: str,
        experiment: str | None = None,
        job: str = "job",
        through_fdllm: bool = False,
        **settings: Any,
    ) -> RunPlan:
        return plan_job(
            self.config(*models, experiment=experiment, **settings),
            workdir=self.root / job,
            download_public_registry=False,
            call_model=None if through_fdllm else self.models,
            storage_client=self.storage,
        )

    def put(self, uri: str, content: bytes) -> None:
        """Write a file to the fake GCS, as a new generation."""

        bucket_name, name = parse_gs_uri(uri)
        generation = self.storage.next_generation
        self.storage.next_generation += 1
        self.storage.objects.setdefault(bucket_name, {})[name] = (generation, content)

    def remove(self, uri: str) -> None:
        bucket_name, name = parse_gs_uri(uri)
        del self.storage.objects[bucket_name][name]

    def image_downloads(self) -> list[str]:
        """Return the GCS images downloaded so far: objects in an images folder."""

        return [name for name in self.storage.downloads if "/images/" in name]

    def folder(self, experiment: str = "experiment") -> Path:
        return self.root / experiment

    def files(self, experiment: str = "experiment") -> set[str]:
        """Return the files in a local experiment folder, relative to it."""

        folder = self.folder(experiment)
        return {
            path.relative_to(folder).as_posix()
            for path in folder.rglob("*")
            if path.is_file()
        }

    def rows(
        self, model: str = "model-a", experiment: str = "experiment"
    ) -> list[dict]:
        path = self.folder(experiment) / "responses" / f"responses_{model}.jsonl"
        return [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        ]

    def row(self, item_id: str, model: str = "model-a", **kwargs: Any) -> dict:
        [row] = [
            r for r in self.rows(model, **kwargs) if r["benchmark_item_id"] == item_id
        ]
        return row

    def record(self, model: str = "model-a", experiment: str = "experiment") -> dict:
        path = self.folder(experiment) / "models" / f"{model}.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def gcs_files(self, bucket: str = "bucket") -> set[str]:
        return set(self.storage.objects.get(bucket, {}))


@pytest.fixture
def bench(tmp_path: Path, storage: Any) -> Bench:
    return Bench(tmp_path, storage)


@pytest.fixture
def providers(monkeypatch: pytest.MonkeyPatch) -> FakeProviders:
    return FakeProviders(monkeypatch)

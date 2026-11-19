"""Shared fixtures: a small benchmark adapter, run files and a fake GCS client."""

from __future__ import annotations

import base64
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import google_crc32c
import pytest
from google.api_core.exceptions import NotFound, PreconditionFailed

from fab_benchmark_runner import Benchmark


@dataclass(frozen=True)
class LanguageParams:
    language: str
    prompt_name: str = "default"

    def __post_init__(self) -> None:
        if not isinstance(self.language, str) or not self.language.strip():
            raise ValueError("'language' must be a non-empty string")


def build_language_prompt(item: Mapping[str, Any], params: LanguageParams) -> str:
    return f"[{params.language}/{params.prompt_name}] {item['content']}"


TEST_BENCHMARK = Benchmark(
    name="test-bench",
    params_type=LanguageParams,
    build_prompt=build_language_prompt,
)

DEFAULT_ITEMS = (
    {"benchmark_item_id": "item-1", "content": "A cat sat."},
    {"benchmark_item_id": "item-2", "content": "The dog ran."},
)


@pytest.fixture
def benchmark() -> Benchmark:
    return TEST_BENCHMARK


@pytest.fixture
def write_items(tmp_path: Path) -> Callable[..., Path]:
    def write(items: Iterable[Mapping[str, Any]] = DEFAULT_ITEMS) -> Path:
        items_path = tmp_path / "items.jsonl"
        items_path.write_text(
            "".join(json.dumps(dict(item)) + "\n" for item in items),
            encoding="utf-8",
        )
        return items_path

    return write


@pytest.fixture
def write_registry(tmp_path: Path) -> Callable[..., Path]:
    """Write a registry with one provider group per model unless groups are given."""

    def write(
        models: Iterable[str] = ("model-a", "model-b"),
        *,
        groups: Mapping[str, Iterable[str]] | None = None,
    ) -> Path:
        if groups is None:
            groups = {f"provider-{name}": [name] for name in models}
        registry = {
            group: {"models": {name: {} for name in names}}
            for group, names in groups.items()
        }
        registry_path = tmp_path / "custom_models.yaml"
        registry_path.write_text(json.dumps(registry), encoding="utf-8")
        return registry_path

    return write


@pytest.fixture
def write_run(
    tmp_path: Path,
    write_items: Callable[..., Path],
    write_registry: Callable[..., Path],
) -> Callable[..., Path]:
    """Write items, a registry and a run config; return the config path."""

    def write(
        *,
        models: Iterable[str] = ("model-a", "model-b"),
        items: Iterable[Mapping[str, Any]] = DEFAULT_ITEMS,
        groups: Mapping[str, Iterable[str]] | None = None,
        **overrides: Any,
    ) -> Path:
        write_items(items)
        write_registry(models, groups=groups)
        config = {
            "run_id": "experiment-a",
            "language": "sw",
            "items_path": "items.jsonl",
            "output_dir": "outputs",
            "models": list(models),
            "model_workers": 1,
            "custom_models_path": "custom_models.yaml",
            **overrides,
        }
        config_path = tmp_path / "run.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        return config_path

    return write


@pytest.fixture
def read_rows() -> Callable[[Iterable[Path]], list[dict[str, Any]]]:
    def read(paths: Iterable[Path]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for path in paths:
            rows.extend(
                json.loads(line)
                for line in Path(path).read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        return rows

    return read


class FakeListedBlob:
    def __init__(self, name: str, generation: int, content: bytes) -> None:
        self.name = name
        self.generation = generation
        checksum = google_crc32c.Checksum(content)
        self.crc32c = base64.b64encode(checksum.digest()).decode("ascii")


class FakeBlob:
    def __init__(self, client: FakeStorageClient, bucket_name: str, name: str) -> None:
        self.client = client
        self.bucket_name = bucket_name
        self.name = name
        self.generation: int | None = None

    def reload(self) -> None:
        try:
            self.generation = self.client.objects[self.bucket_name][self.name][0]
        except KeyError as e:
            raise NotFound(self.name) from e

    def download_to_filename(
        self, filename: str, *, if_generation_match: int | None = None
    ) -> None:
        try:
            generation, content = self.client.objects[self.bucket_name][self.name]
        except KeyError as e:
            raise NotFound(self.name) from e
        if if_generation_match is not None and generation != if_generation_match:
            raise PreconditionFailed(self.name)
        Path(filename).write_bytes(content)

    def upload_from_filename(self, filename: str, *, if_generation_match: int) -> None:
        self.client.upload_preconditions.append((self.name, if_generation_match))
        if self.name in self.client.fail_next_upload:
            self.client.fail_next_upload.remove(self.name)
            raise RuntimeError("simulated transient upload failure")
        bucket = self.client.objects.setdefault(self.bucket_name, {})
        current_generation = bucket.get(self.name, (0, b""))[0]
        if current_generation != if_generation_match:
            raise PreconditionFailed(self.name)
        self.generation = self.client.next_generation
        self.client.next_generation += 1
        bucket[self.name] = (self.generation, Path(filename).read_bytes())
        if self.name in self.client.commit_then_fail_next_upload:
            self.client.commit_then_fail_next_upload.remove(self.name)
            raise RuntimeError("simulated lost upload acknowledgement")


class FakeBucket:
    def __init__(self, client: FakeStorageClient, name: str) -> None:
        self.client = client
        self.name = name

    def blob(self, name: str) -> FakeBlob:
        return FakeBlob(self.client, self.name, name)


class FakeStorageClient:
    """An in-memory GCS with generations and injectable upload failures."""

    def __init__(self) -> None:
        self.objects: dict[str, dict[str, tuple[int, bytes]]] = {}
        self.next_generation = 100
        self.upload_preconditions: list[tuple[str, int]] = []
        self.fail_next_upload: set[str] = set()
        self.commit_then_fail_next_upload: set[str] = set()

    def bucket(self, name: str) -> FakeBucket:
        return FakeBucket(self, name)

    def list_blobs(self, bucket_name: str, *, prefix: str) -> list[FakeListedBlob]:
        return [
            FakeListedBlob(name, generation, content)
            for name, (generation, content) in self.objects.get(bucket_name, {}).items()
            if name.startswith(prefix)
        ]


@pytest.fixture
def storage() -> FakeStorageClient:
    return FakeStorageClient()

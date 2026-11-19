"""The inputs file a benchmark hands to the runner: one rendered input per item.

Each line is a JSON object with ``benchmark_item_id``, ``prompt``, an optional
``system_prompt`` and an optional ``images`` list. Each image has a ``uri`` (a
``gs://`` URI, an absolute path, or a path relative to the inputs file) and a
``detail`` of ``low`` or ``high``. Images go into the item's one user message, in
order. The inputs file itself can be local or on GCS. The runner never builds
prompts: what is in this file is what models see.
"""

from __future__ import annotations

import json
import posixpath
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from google.api_core.exceptions import NotFound

from fab_benchmark_runner.errors import ConfigError
from fab_benchmark_runner.gcs import (
    create_storage_client,
    download_file_with_generation,
    parse_gs_uri,
)
from fab_benchmark_runner.ids import IMAGE_DETAILS, canonical_json, image_identity
from fab_benchmark_runner.results import write_text_atomic

ITEM_ID_KEY = "benchmark_item_id"
INPUT_KEYS = frozenset({ITEM_ID_KEY, "prompt", "system_prompt", "images"})
IMAGE_KEYS = frozenset({"uri", "detail", "sha256"})
STORED_IMAGE_KEYS = IMAGE_KEYS | {"source_uri"}


@dataclass(frozen=True)
class ImageRef:
    """Where one image is, how it is sent, and, once known, its file's SHA-256.

    ``source_uri`` is set only on an image copied into an experiment folder: it says
    where the copy came from.
    """

    uri: str
    detail: str
    sha256: str | None = None
    source_uri: str | None = None

    def __post_init__(self) -> None:
        for name in ("uri", "source_uri"):
            value = getattr(self, name)
            if value is None and name == "source_uri":
                continue
            if not isinstance(value, str) or not value or value != value.strip():
                raise ValueError(
                    f"Image {name} must be a non-empty string without surrounding "
                    "spaces"
                )
        if self.detail not in IMAGE_DETAILS:
            raise ValueError(
                f"Image detail must be 'low' or 'high', not {self.detail!r}"
            )
        if self.sha256 is not None:
            image_identity({"detail": self.detail, "sha256": self.sha256})

    @property
    def is_gcs(self) -> bool:
        return self.uri.startswith("gs://")


@dataclass(frozen=True)
class ModelInput:
    """Everything one model call receives for one item."""

    benchmark_item_id: str
    prompt: str
    system_prompt: str | None = None
    images: tuple[ImageRef, ...] = field(default=())

    def __post_init__(self) -> None:
        item_id = self.benchmark_item_id
        if not isinstance(item_id, str) or not item_id or item_id != item_id.strip():
            raise ValueError(
                f"{ITEM_ID_KEY} must be a non-empty string without surrounding spaces"
            )
        if not isinstance(self.prompt, str) or not self.prompt:
            raise ValueError("prompt must be a non-empty string")
        if self.system_prompt is not None and (
            not isinstance(self.system_prompt, str) or not self.system_prompt
        ):
            raise ValueError("system_prompt must be a non-empty string or null")
        if not isinstance(self.images, tuple) or not all(
            isinstance(image, ImageRef) for image in self.images
        ):
            raise ValueError("images must be a tuple of ImageRef")

    def to_row(self) -> dict[str, Any]:
        """Return the input as a JSON row; a text-only row has no images key."""

        row: dict[str, Any] = {
            ITEM_ID_KEY: self.benchmark_item_id,
            "prompt": self.prompt,
            "system_prompt": self.system_prompt,
        }
        if self.images:
            row["images"] = [_image_row(image) for image in self.images]
        return row


def read_inputs(
    path: str | Path, *, storage_client: Any | None = None
) -> list[ModelInput]:
    """Read and check an inputs file, local or at a ``gs://`` URI.

    Relative image paths are resolved against the file's folder, so a file next to
    its images works the same locally and on GCS. Unknown keys are refused, so a
    misspelled field can't silently drop part of an input.
    """

    return load_inputs_file(path, storage_client=storage_client)[0]


def load_inputs_file(
    path: str | Path, *, storage_client: Any | None = None
) -> tuple[list[ModelInput], int | None]:
    """Read an inputs file; also return the GCS generation read, if on GCS."""

    source = str(path)
    if not source.startswith("gs://"):
        inputs_path = Path(path)
        try:
            text = inputs_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            raise ConfigError(f"Could not read inputs file {inputs_path}: {e}") from e
        return _parse_inputs(
            text, source=source, base=inputs_path.resolve().parent
        ), None

    try:
        bucket_name, object_name = parse_gs_uri(source)
    except ValueError as e:
        raise ConfigError(f"Invalid inputs file URI {source}: {e}") from e
    client = storage_client or create_storage_client()
    with tempfile.TemporaryDirectory(prefix="fab-bench-inputs-") as temporary:
        local_copy = Path(temporary) / "inputs.jsonl"
        try:
            generation = download_file_with_generation(
                bucket_name=bucket_name,
                file_name=object_name,
                write_path=local_copy,
                storage_client=client,
            )
        except NotFound as e:
            raise ConfigError(f"Inputs file {source} does not exist") from e
        try:
            text = local_copy.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            raise ConfigError(f"Could not read inputs file {source}: {e}") from e
    folder = posixpath.dirname(object_name)
    base = f"gs://{bucket_name}/{folder}/" if folder else f"gs://{bucket_name}/"
    return _parse_inputs(text, source=source, base=base), generation


def _parse_inputs(text: str, *, source: str, base: Path | str) -> list[ModelInput]:
    inputs: list[ModelInput] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        where = f"{source}:{line_number}"
        if not line.strip():
            raise ConfigError(f"{where}: blank lines are not allowed")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise ConfigError(f"{where}: invalid JSON: {e}") from e
        inputs.append(parse_input_row(row, where=where, base=base))
    check_inputs(inputs, source=source)
    return inputs


def parse_input_row(
    row: Any, *, where: str, base: Path | str | None = None, stored: bool = False
) -> ModelInput:
    """Turn one JSON row into a ModelInput.

    With ``base``, a local folder or a ``gs://`` folder ending in ``/``, relative
    image paths are resolved against it. A ``stored`` row comes from an
    experiment's own inputs: its images must carry their SHA-256, and relative
    paths stay relative to the experiment folder.
    """

    if not isinstance(row, dict):
        raise ConfigError(f"{where}: an input must be a JSON object")
    unknown = sorted(set(row) - INPUT_KEYS)
    if unknown:
        raise ConfigError(f"{where}: unknown keys {unknown}")
    images_value = row.get("images", [])
    if not isinstance(images_value, list):
        raise ConfigError(f"{where}: images must be a list")
    try:
        images = tuple(
            _parse_image(
                image, where=f"{where}: image {index}", base=base, stored=stored
            )
            for index, image in enumerate(images_value, start=1)
        )
        return ModelInput(
            benchmark_item_id=row.get(ITEM_ID_KEY),
            prompt=row.get("prompt"),
            system_prompt=row.get("system_prompt"),
            images=images,
        )
    except ValueError as e:
        if isinstance(e, ConfigError):
            raise
        raise ConfigError(f"{where}: {e}") from e


def check_inputs(inputs: Iterable[ModelInput], *, source: str | Path) -> None:
    """Require at least one input and unique item IDs."""

    seen: set[str] = set()
    for model_input in inputs:
        if model_input.benchmark_item_id in seen:
            raise ConfigError(
                f"{source}: duplicate {ITEM_ID_KEY} '{model_input.benchmark_item_id}'"
            )
        seen.add(model_input.benchmark_item_id)
    if not seen:
        raise ConfigError(f"{source}: no inputs")


def write_inputs(inputs: Iterable[ModelInput], path: str | Path) -> Path:
    """Write an inputs file, for benchmarks that build their inputs in Python."""

    inputs = list(inputs)
    check_inputs(inputs, source=path)
    inputs_path = Path(path)
    inputs_path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(
        inputs_path,
        "".join(canonical_json(model_input.to_row()) + "\n" for model_input in inputs),
    )
    return inputs_path


def _parse_image(
    image: Any, *, where: str, base: Path | str | None, stored: bool
) -> ImageRef:
    if not isinstance(image, Mapping):
        raise ConfigError(f"{where}: an image must be a JSON object")
    unknown = sorted(set(image) - (STORED_IMAGE_KEYS if stored else IMAGE_KEYS))
    if unknown:
        raise ConfigError(f"{where}: unknown keys {unknown}")
    if stored and image.get("sha256") is None:
        raise ConfigError(f"{where}: a stored image must have its sha256")
    uri = image.get("uri")
    if base is not None and isinstance(uri, str) and uri.strip() == uri and uri:
        uri = _resolve_image_uri(uri, base, where=where)
    try:
        return ImageRef(
            uri=uri,
            detail=image.get("detail"),
            sha256=image.get("sha256"),
            source_uri=image.get("source_uri"),
        )
    except ValueError as e:
        raise ConfigError(f"{where}: {e}") from e


def _resolve_image_uri(uri: str, base: Path | str, *, where: str) -> str:
    if uri.startswith("gs://"):
        return uri
    if isinstance(base, Path):
        local_path = Path(uri).expanduser()
        if not local_path.is_absolute():
            local_path = base / local_path
        return str(local_path.resolve())
    if uri.startswith(("/", "~")):
        raise ConfigError(
            f"{where}: an inputs file on GCS can only point at images on GCS, not "
            f"at the local path {uri}"
        )
    bucket_name, _, folder = base.removeprefix("gs://").partition("/")
    name = posixpath.normpath(posixpath.join(folder, uri))
    if name == "." or name == ".." or name.startswith("../"):
        raise ConfigError(f"{where}: image path {uri} leaves the bucket")
    return f"gs://{bucket_name}/{name}"


def _image_row(image: ImageRef) -> dict[str, Any]:
    row: dict[str, Any] = {"detail": image.detail, "uri": image.uri}
    if image.sha256 is not None:
        row["sha256"] = image.sha256
    if image.source_uri is not None:
        row["source_uri"] = image.source_uri
    return row

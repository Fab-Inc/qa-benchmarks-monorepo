"""The inputs file a benchmark hands to the runner: one rendered input per item.

Each line is a JSON object with ``benchmark_item_id``, ``prompt``, an optional
``system_prompt`` and an optional ``images`` list. The runner never builds
prompts: what is in this file is what models see. INPUTS.md, next to the package's
README, specifies the format; ``tests/behaviour/test_inputs_file.py`` checks it.
"""

from __future__ import annotations

import hashlib
import json
import posixpath
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from google.api_core.exceptions import NotFound

from fab_benchmark_runner.errors import ConfigError
from fab_benchmark_runner.gcs import (
    create_storage_client,
    download_file_with_generation,
    download_to_path,
    parse_gs_uri,
)
from fab_benchmark_runner.ids import IMAGE_DETAILS, canonical_json, is_sha256
from fab_benchmark_runner.results import file_sha256, write_text_atomic

ITEM_ID_KEY = "benchmark_item_id"
INPUT_KEYS = frozenset({ITEM_ID_KEY, "prompt", "system_prompt", "images"})
IMAGE_KEYS = frozenset({"uri", "detail", "sha256"})
STORED_IMAGE_KEYS = IMAGE_KEYS | {"source_uri"}
IMAGE_READ_WORKERS = 32
PROBLEMS_IN_MESSAGE = 20


class InputsError(ConfigError):
    """Every problem found in some inputs, each naming its file and line."""

    def __init__(self, problems: Sequence[str], *, source: str | None = None) -> None:
        self.problems = tuple(problems)
        super().__init__(_problems_message(self.problems, source=source))


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
        problems = _image_problems(
            uri=self.uri,
            detail=self.detail,
            sha256=self.sha256,
            source_uri=self.source_uri,
        )
        if problems:
            raise ValueError("; ".join(f"Image {problem}" for problem in problems))

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
        problems = _item_problems(
            benchmark_item_id=self.benchmark_item_id,
            prompt=self.prompt,
            system_prompt=self.system_prompt,
        )
        if not isinstance(self.images, tuple) or not all(
            isinstance(image, ImageRef) for image in self.images
        ):
            problems.append("images must be a tuple of ImageRef")
        if problems:
            raise ValueError("; ".join(problems))

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


@dataclass(frozen=True)
class InputsCheck:
    """What checking an inputs file found; a valid file has no ``problems``.

    ``items`` counts the lines that are valid items. ``image_files`` counts the
    distinct image files they use, and ``image_uses`` every use of one, so an image
    in three items counts once and three times.
    """

    source: str
    items: int
    image_files: int
    image_uses: int
    problems: tuple[str, ...]


def read_inputs(
    path: str | Path, *, storage_client: Any | None = None
) -> list[ModelInput]:
    """Read and check an inputs file, local or at a ``gs://`` URI.

    Relative image paths are resolved against the file's folder, so a file next to
    its images works the same locally and on GCS. Unknown keys are refused, so a
    misspelled field can't silently drop part of an input. Raises InputsError
    listing every problem in the file.
    """

    return load_inputs_file(path, storage_client=storage_client)[0]


def load_inputs_file(
    path: str | Path, *, storage_client: Any | None = None
) -> tuple[list[ModelInput], int | None]:
    """Read an inputs file; also return the GCS generation read, if on GCS."""

    text, source, base, generation = _read_inputs_text(
        path, storage_client=storage_client
    )
    rows, problems = _parse_inputs(text, source=source, base=base)
    if problems:
        raise InputsError([problem for _, problem in problems], source=source)
    return [model_input for _, model_input in rows], generation


def check_inputs_file(
    path: str | Path, *, storage_client: Any | None = None
) -> InputsCheck:
    """Check an inputs file against every rule in INPUTS.md and list every problem.

    Each image is read once, up to 32 at a time, to check that it exists and, where
    the file gives its SHA-256, that it matches. Whether an image opens is checked
    when a run sends it. Raises ConfigError only when the file itself can't be read.
    """

    text, source, base, _ = _read_inputs_text(path, storage_client=storage_client)
    rows, problems = _parse_inputs(text, source=source, base=base)
    uses = [
        (line, index, image)
        for line, model_input in rows
        for index, image in enumerate(model_input.images, start=1)
    ]
    uris = list(dict.fromkeys(image.uri for _, _, image in uses))
    if storage_client is None and any(uri.startswith("gs://") for uri in uris):
        storage_client = create_storage_client()
    digests = _hash_images(uris, storage_client=storage_client)
    reported: set[str] = set()
    for line, index, image in uses:
        where = f"{source}:{line}: image {index}"
        digest = digests[image.uri]
        if digest is None and image.uri not in reported:
            reported.add(image.uri)
            problems.append((line, f"{where}: {image.uri} does not exist"))
        elif digest is not None and image.sha256 not in (None, digest):
            problems.append(
                (
                    line,
                    f"{where}: {image.uri} has SHA-256 {digest}, not the "
                    f"{image.sha256} the file gives",
                )
            )
    problems.sort(key=lambda problem: problem[0])
    return InputsCheck(
        source=source,
        items=len(rows),
        image_files=len(uris),
        image_uses=len(uses),
        problems=tuple(problem for _, problem in problems),
    )


def parse_input_row(
    row: Any, *, where: str, base: Path | str | None = None, stored: bool = False
) -> ModelInput:
    """Turn one JSON row into a ModelInput, or raise InputsError with every problem.

    With ``base``, a local folder or a ``gs://`` folder ending in ``/``, relative
    image paths are resolved against it. A ``stored`` row comes from an
    experiment's own inputs: its images must carry their SHA-256, and relative
    paths stay relative to the experiment folder.
    """

    if not isinstance(row, dict):
        raise InputsError([f"{where}: an input must be a JSON object"])
    problems: list[str] = []
    unknown = sorted(set(row) - INPUT_KEYS)
    if unknown:
        problems.append(f"unknown keys {unknown}")
    images: list[ImageRef] = []
    images_value = row.get("images", [])
    if not isinstance(images_value, list):
        problems.append("images must be a list")
    else:
        for index, image in enumerate(images_value, start=1):
            image_ref, image_problems = _parse_image(image, base=base, stored=stored)
            problems.extend(f"image {index}: {problem}" for problem in image_problems)
            if image_ref is not None:
                images.append(image_ref)
    problems.extend(
        _item_problems(
            benchmark_item_id=row.get(ITEM_ID_KEY),
            prompt=row.get("prompt"),
            system_prompt=row.get("system_prompt"),
        )
    )
    if problems:
        raise InputsError([f"{where}: {problem}" for problem in problems])
    return ModelInput(
        benchmark_item_id=row[ITEM_ID_KEY],
        prompt=row["prompt"],
        system_prompt=row.get("system_prompt"),
        images=tuple(images),
    )


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


def fetch_image(uri: str, downloads: Path, storage_client: Any | None) -> Path | None:
    """Return a local path holding the image, downloading it if on GCS.

    Returns None when there is no such image. A download goes into ``downloads``,
    named by the URI's hash; the caller removes it when done.
    """

    if not uri.startswith("gs://"):
        path = Path(uri)
        return path if path.is_file() else None
    downloads.mkdir(parents=True, exist_ok=True)
    target = downloads / hashlib.sha256(uri.encode("utf-8")).hexdigest()
    try:
        download_to_path(uri, target, storage_client=storage_client)
    except NotFound:
        return None
    return target


def _read_inputs_text(
    path: str | Path, *, storage_client: Any | None
) -> tuple[str, str, Path | str, int | None]:
    """Return the file's text, how to name it, its folder and its GCS generation."""

    source = str(path)
    if not source.startswith("gs://"):
        inputs_path = Path(path)
        try:
            text = inputs_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            raise ConfigError(f"Could not read inputs file {inputs_path}: {e}") from e
        return text, source, inputs_path.resolve().parent, None

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
    return text, source, base, generation


def _parse_inputs(
    text: str, *, source: str, base: Path | str
) -> tuple[list[tuple[int, ModelInput]], list[tuple[int, str]]]:
    """Return each valid line's input and every problem, both with line numbers."""

    lines = text.splitlines()
    if not lines:
        return [], [(0, f"{source}: no inputs")]
    rows: list[tuple[int, ModelInput]] = []
    problems: list[tuple[int, str]] = []
    first_lines: dict[Any, int] = {}
    for line_number, line in enumerate(lines, start=1):
        where = f"{source}:{line_number}"
        if not line.strip():
            problems.append((line_number, f"{where}: blank lines are not allowed"))
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            problems.append((line_number, f"{where}: invalid JSON: {e}"))
            continue
        try:
            rows.append((line_number, parse_input_row(row, where=where, base=base)))
        except InputsError as e:
            problems.extend((line_number, problem) for problem in e.problems)
        item_id = row.get(ITEM_ID_KEY) if isinstance(row, dict) else None
        if not isinstance(item_id, str):
            continue
        if item_id in first_lines:
            problems.append(
                (
                    line_number,
                    f"{where}: duplicate {ITEM_ID_KEY} '{item_id}', first on line "
                    f"{first_lines[item_id]}",
                )
            )
        else:
            first_lines[item_id] = line_number
    return rows, problems


def _item_problems(
    *, benchmark_item_id: Any, prompt: Any, system_prompt: Any
) -> list[str]:
    problems = []
    if isinstance(benchmark_item_id, int | float) and not isinstance(
        benchmark_item_id, bool
    ):
        problems.append(
            f"{ITEM_ID_KEY} must be a string: write the number {benchmark_item_id} "
            f'as "{benchmark_item_id}"'
        )
    elif not _is_trimmed_text(benchmark_item_id):
        problems.append(
            f"{ITEM_ID_KEY} must be a non-empty string without surrounding spaces"
        )
    if not isinstance(prompt, str) or not prompt.strip():
        problems.append("prompt must be a string with more than whitespace")
    if system_prompt is not None and (
        not isinstance(system_prompt, str) or not system_prompt.strip()
    ):
        problems.append(
            "system_prompt must be null or a string with more than whitespace"
        )
    return problems


def _image_problems(
    *, uri: Any, detail: Any, sha256: Any, source_uri: Any
) -> list[str]:
    problems = []
    for name, value in (("uri", uri), ("source_uri", source_uri)):
        if name == "source_uri" and value is None:
            continue
        if not _is_trimmed_text(value):
            problems.append(
                f"{name} must be a non-empty string without surrounding spaces"
            )
    if detail not in IMAGE_DETAILS:
        problems.append(f"detail must be 'low' or 'high', not {detail!r}")
    if sha256 is not None and not is_sha256(sha256):
        problems.append(f"sha256 must be 64 lowercase hex characters, not {sha256!r}")
    return problems


def _parse_image(
    image: Any, *, base: Path | str | None, stored: bool
) -> tuple[ImageRef | None, list[str]]:
    if not isinstance(image, Mapping):
        return None, ["an image must be a JSON object"]
    problems = []
    unknown = sorted(set(image) - (STORED_IMAGE_KEYS if stored else IMAGE_KEYS))
    if unknown:
        problems.append(f"unknown keys {unknown}")
    if stored and image.get("sha256") is None:
        problems.append("a stored image must have its sha256")
    uri = image.get("uri")
    fields = {
        "detail": image.get("detail"),
        "sha256": image.get("sha256"),
        "source_uri": image.get("source_uri"),
    }
    problems.extend(_image_problems(uri=uri, **fields))
    if base is not None and _is_trimmed_text(uri):
        try:
            uri = _resolve_image_uri(uri, base)
        except ValueError as e:
            problems.append(str(e))
    if problems:
        return None, problems
    return ImageRef(uri=uri, **fields), []


def _resolve_image_uri(uri: str, base: Path | str) -> str:
    if uri.startswith("gs://"):
        return uri
    if isinstance(base, Path):
        local_path = Path(uri).expanduser()
        if not local_path.is_absolute():
            local_path = base / local_path
        return str(local_path.resolve())
    if uri.startswith(("/", "~")):
        raise ValueError(
            "an inputs file on GCS can only point at images on GCS, not at the "
            f"local path {uri}"
        )
    bucket_name, _, folder = base.removeprefix("gs://").partition("/")
    name = posixpath.normpath(posixpath.join(folder, uri))
    if name == "." or name == ".." or name.startswith("../"):
        raise ValueError(f"image path {uri} leaves the bucket")
    return f"gs://{bucket_name}/{name}"


def _hash_images(
    uris: list[str], *, storage_client: Any | None
) -> dict[str, str | None]:
    """Return each image's SHA-256, or None for an image that doesn't exist."""

    if not uris:
        return {}
    with (
        tempfile.TemporaryDirectory(prefix="fab-bench-check-") as temporary,
        ThreadPoolExecutor(max_workers=IMAGE_READ_WORKERS) as pool,
    ):
        futures = [
            pool.submit(_hash_image, uri, Path(temporary), storage_client)
            for uri in uris
        ]
        try:
            return {
                uri: future.result() for uri, future in zip(uris, futures, strict=True)
            }
        except BaseException:
            pool.shutdown(cancel_futures=True)
            raise


def _hash_image(uri: str, downloads: Path, storage_client: Any | None) -> str | None:
    path = fetch_image(uri, downloads, storage_client)
    if path is None:
        return None
    try:
        return file_sha256(path)
    finally:
        if uri.startswith("gs://"):
            path.unlink(missing_ok=True)


def _is_trimmed_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and value == value.strip()


def _problems_message(problems: Sequence[str], *, source: str | None) -> str:
    if len(problems) == 1:
        return problems[0]
    where = f" in {source}" if source else ""
    lines = [f"{len(problems)} problems{where}:"]
    lines.extend(f"  {problem}" for problem in problems[:PROBLEMS_IN_MESSAGE])
    hidden = len(problems) - PROBLEMS_IN_MESSAGE
    if hidden > 0:
        more = f"  … and {hidden} more"
        if source:
            more += f"; fab-bench check-inputs {source} lists them all"
        lines.append(more)
    return "\n".join(lines)


def _image_row(image: ImageRef) -> dict[str, Any]:
    row: dict[str, Any] = {"detail": image.detail, "uri": image.uri}
    if image.sha256 is not None:
        row["sha256"] = image.sha256
    if image.source_uri is not None:
        row["source_uri"] = image.source_uri
    return row

"""Experiments: a benchmark's inputs, stored once so every model gets the same.

An experiment folder, local or on GCS, holds:

- ``inputs.jsonl``: one input per item, each image with its file's SHA-256;
- ``images/``: only images copied in on request, named by their SHA-256;
- ``experiment.json``: what defines the experiment, written last, so an
  interrupted creation never looks finished.

Jobs later add ``models/`` and ``responses/`` but never change these files. The one
thing that changes after creation is ``max_tokens``, which can be raised in place.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any

from google.api_core.exceptions import NotFound, PreconditionFailed

from fab_benchmark_runner.errors import ConfigError
from fab_benchmark_runner.gcs import (
    create_storage_client,
    download_file_with_generation,
    download_to_path,
    list_gcs_objects,
    parse_gs_uri,
    upload_file_with_generation,
)
from fab_benchmark_runner.ids import canonical_json, sha256_json
from fab_benchmark_runner.inputs import (
    ImageRef,
    ModelInput,
    check_inputs,
    load_inputs_file,
    parse_input_row,
)
from fab_benchmark_runner.provenance import git_dirty, git_hash
from fab_benchmark_runner.results import (
    file_sha256,
    utc_now,
    write_bytes_atomic,
    write_text_atomic,
)
from fab_benchmark_runner.run_sync import RunSync, file_crc32c

EXPERIMENT_RECORD_NAME = "experiment.json"
INPUTS_FILE_NAME = "inputs.jsonl"
IMAGES_DIR = "images"
EXPERIMENT_RECORD_FORMAT = 1
DEFAULT_EXPERIMENT_ID_LENGTH = 16
IDENTITY_KEYS = ("benchmark", "experiment_id", "inputs_sha256")
RAISE_MAX_TOKENS_HINT = "raise it with fab-bench raise-max-tokens"
IMAGE_READ_WORKERS = 32
_EXPERIMENT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_IMAGE_SUFFIX = re.compile(r"\.[a-z0-9]{1,5}")


@dataclass(frozen=True)
class Experiment:
    """A stored experiment: where it is, its record, and its inputs."""

    location: str
    record: Mapping[str, Any]
    inputs: tuple[ModelInput, ...]

    @property
    def experiment_id(self) -> str:
        return self.record["experiment_id"]

    @property
    def benchmark(self) -> str:
        return self.record["benchmark"]

    @property
    def max_tokens(self) -> int | None:
        return self.record["max_tokens"]

    @property
    def inputs_sha256(self) -> str:
        return self.record["inputs_sha256"]


@dataclass(frozen=True)
class ExperimentCreation:
    """What ``create_experiment`` did: ``created``, ``exists`` or ``dry-run``."""

    status: str
    experiment: Experiment


@dataclass(frozen=True)
class MaxTokensChange:
    """What ``raise_max_tokens`` did: ``raised`` or ``unchanged``."""

    status: str
    previous: int | None
    max_tokens: int | None
    record: Mapping[str, Any]


def default_experiment_id(*, benchmark: str, inputs_sha256: str) -> str:
    """Return the ID used when none is given: a short hash of what defines it.

    ``max_tokens`` is left out, because raising it keeps the experiment.
    """

    return sha256_json({"benchmark": benchmark, "inputs_sha256": inputs_sha256})[
        :DEFAULT_EXPERIMENT_ID_LENGTH
    ]


def create_experiment(
    inputs: str | Path | Iterable[ModelInput],
    *,
    dest: str | Path,
    benchmark: str,
    max_tokens: int | None,
    experiment_id: str | None = None,
    labels: Mapping[str, str] | None = None,
    copy_images: bool = False,
    dry_run: bool = False,
    storage_client: Any | None = None,
) -> ExperimentCreation:
    """Store a benchmark's inputs as an experiment in ``dest``.

    ``inputs`` is an inputs file, local or at a ``gs://`` URI, or ModelInputs built
    in Python. Every image is read once, up to 32 at a time, to record its SHA-256;
    with ``copy_images`` it is also copied into the experiment's ``images/``
    folder. Images at local paths must be copied when the experiment is on GCS,
    because jobs there can't read them.

    Creating an experiment that already exists with the same benchmark, ID, inputs
    and ``max_tokens`` changes nothing. Raises ConfigError when ``dest`` holds a
    different experiment or other files, or this experiment with another
    ``max_tokens``: only ``raise_max_tokens`` changes the limit.
    """

    _check_settings(benchmark=benchmark, max_tokens=max_tokens, labels=labels)
    if experiment_id is not None:
        check_experiment_id(experiment_id)
    if (
        isinstance(inputs, str)
        and inputs.startswith("gs://")
        and storage_client is None
    ):
        storage_client = create_storage_client()
    model_inputs, inputs_source = _load_inputs(inputs, storage_client=storage_client)
    dest_text = str(dest)
    dest_is_gcs = dest_text.startswith("gs://")
    if dest_is_gcs:
        parse_gs_uri(dest_text.rstrip("/") + "/")
    local_images = sorted(
        {
            image.uri
            for item in model_inputs
            for image in item.images
            if not image.is_gcs
        }
    )
    if dest_is_gcs and local_images and not copy_images:
        raise ConfigError(
            f"Images at local paths, such as {local_images[0]}, can't be read by jobs "
            "on GCS; pass copy_images (--copy-images) to copy them into the experiment"
        )
    needs_gcs = dest_is_gcs or any(
        image.is_gcs for item in model_inputs for image in item.images
    )
    if needs_gcs and storage_client is None:
        storage_client = create_storage_client()

    with tempfile.TemporaryDirectory(prefix="fab-bench-experiment-") as temporary:
        staging = Path(temporary) / "experiment"
        staging.mkdir()
        stored_inputs = _store_images(
            model_inputs,
            staging=staging,
            copy_images=copy_images,
            downloads=Path(temporary) / "downloads",
            storage_client=storage_client,
        )
        inputs_text = "".join(
            canonical_json(item.to_row()) + "\n" for item in stored_inputs
        )
        inputs_sha256 = sha256(inputs_text.encode("utf-8")).hexdigest()
        if experiment_id is None:
            experiment_id = default_experiment_id(
                benchmark=benchmark, inputs_sha256=inputs_sha256
            )
        created_utc = utc_now()
        record = {
            "benchmark": benchmark,
            "created": {
                "git_dirty": git_dirty(),
                "git_hash": git_hash(),
                "inputs_source": inputs_source,
                "utc": created_utc,
            },
            "experiment_id": experiment_id,
            "image_count": len(
                {image.sha256 for item in stored_inputs for image in item.images}
            ),
            "inputs_sha256": inputs_sha256,
            "item_count": len(stored_inputs),
            "labels": dict(labels or {}),
            "max_tokens": max_tokens,
            "max_tokens_history": [{"max_tokens": max_tokens, "utc": created_utc}],
            "record_format": EXPERIMENT_RECORD_FORMAT,
        }
        experiment = Experiment(
            location=dest_text, record=record, inputs=tuple(stored_inputs)
        )

        if dest_is_gcs:
            existing, inputs_present = _existing_on_gcs(
                dest_text,
                inputs_text=inputs_text,
                storage_client=storage_client,
                pull_dir=Path(temporary) / "existing",
            )
        else:
            existing, inputs_present = _existing_locally(
                Path(dest), inputs_text=inputs_text
            )
        if existing is not None:
            _check_same_experiment(existing, record, location=dest_text)
            return ExperimentCreation("exists", replace(experiment, record=existing))
        if dry_run:
            return ExperimentCreation("dry-run", experiment)

        write_text_atomic(staging / INPUTS_FILE_NAME, inputs_text)
        write_text_atomic(
            staging / EXPERIMENT_RECORD_NAME, canonical_json(record, indent=2) + "\n"
        )
        if dest_is_gcs:
            _upload(
                staging,
                dest_text,
                skip_inputs=inputs_present,
                storage_client=storage_client,
            )
        else:
            _write_locally(staging, Path(dest), skip_inputs=inputs_present)
    return ExperimentCreation("created", experiment)


def read_experiment(directory: str | Path) -> Experiment:
    """Read a local experiment folder and check its inputs against its record."""

    folder = Path(directory)
    record_path = folder / EXPERIMENT_RECORD_NAME
    record = _read_record(record_path)
    inputs_path = folder / INPUTS_FILE_NAME
    try:
        inputs_bytes = inputs_path.read_bytes()
    except OSError as e:
        raise ConfigError(f"Could not read {inputs_path}: {e}") from e
    if sha256(inputs_bytes).hexdigest() != record["inputs_sha256"]:
        raise ConfigError(
            f"{inputs_path} doesn't match the inputs_sha256 in {record_path}"
        )
    inputs = tuple(
        parse_input_row(json.loads(line), where=f"{inputs_path}:{number}", stored=True)
        for number, line in enumerate(
            inputs_bytes.decode("utf-8").splitlines(), start=1
        )
    )
    check_inputs(inputs, source=inputs_path)
    return Experiment(location=str(folder), record=record, inputs=inputs)


def raise_max_tokens(
    location: str | Path, max_tokens: int | None, *, storage_client: Any | None = None
) -> MaxTokensChange:
    """Raise an experiment's token limit in place; None means no limit.

    Call IDs leave the limit out, so every result keeps its ID: results that
    finished within the old limit stay, and a run with ``retry_errors`` reruns the
    error rows, which include responses the old limit left empty. The record keeps
    each limit with when it was set. Lowering is refused, because results that
    finished above a lower limit are not ones a model could give under it.
    """

    _check_settings(benchmark="experiment", max_tokens=max_tokens, labels=None)
    text = str(location)
    if not text.startswith("gs://"):
        path = Path(location) / EXPERIMENT_RECORD_NAME
        if not path.exists():
            raise ConfigError(f"{location} holds no experiment ({path} is missing)")
        change = _raised(_read_record(path), max_tokens, location=text)
        if change.status == "raised":
            write_text_atomic(path, canonical_json(change.record, indent=2) + "\n")
        return change

    client = storage_client or create_storage_client()
    record_uri = f"{text.rstrip('/')}/{EXPERIMENT_RECORD_NAME}"
    bucket_name, object_name = parse_gs_uri(record_uri)
    with tempfile.TemporaryDirectory(prefix="fab-bench-raise-") as temporary:
        local = Path(temporary) / EXPERIMENT_RECORD_NAME
        try:
            generation = download_file_with_generation(
                bucket_name=bucket_name,
                file_name=object_name,
                write_path=local,
                storage_client=client,
            )
        except NotFound as e:
            raise ConfigError(
                f"{text} holds no experiment ({record_uri} is missing)"
            ) from e
        change = _raised(_read_record(local), max_tokens, location=text)
        if change.status == "raised":
            write_text_atomic(local, canonical_json(change.record, indent=2) + "\n")
            try:
                upload_file_with_generation(
                    source_path=local,
                    bucket_name=bucket_name,
                    file_name=object_name,
                    if_generation_match=generation,
                    storage_client=client,
                )
            except PreconditionFailed as e:
                raise ConfigError(
                    f"{record_uri} changed while its limit was being raised; run the "
                    "command again"
                ) from e
    return change


def check_experiment_id(experiment_id: str) -> str:
    """Require 1 to 128 letters, digits, '.', '-' or '_', starting alphanumerically."""

    if not isinstance(experiment_id, str) or not _EXPERIMENT_ID.fullmatch(
        experiment_id
    ):
        raise ConfigError(
            f"Experiment ID {experiment_id!r} must be 1 to 128 letters, digits, '.', "
            "'-' or '_', starting with a letter or digit"
        )
    return experiment_id


def _check_settings(
    *, benchmark: str, max_tokens: int | None, labels: Mapping[str, str] | None
) -> None:
    if (
        not isinstance(benchmark, str)
        or not benchmark
        or benchmark != benchmark.strip()
    ):
        raise ConfigError("benchmark must be a non-empty name")
    if max_tokens is not None and (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or max_tokens < 1
    ):
        raise ConfigError("max_tokens must be a positive integer, or None for no limit")
    if labels is not None and not isinstance(labels, Mapping):
        raise ConfigError("labels must map names to strings")
    for key, value in (labels or {}).items():
        if not isinstance(key, str) or not key or not isinstance(value, str):
            raise ConfigError(
                f"Labels must map names to strings, not {key!r}: {value!r}"
            )


def _load_inputs(
    inputs: str | Path | Iterable[ModelInput], *, storage_client: Any | None
) -> tuple[list[ModelInput], dict[str, Any] | None]:
    """Return the inputs and, for an inputs file on GCS, which version was read."""

    if isinstance(inputs, str | Path):
        model_inputs, generation = load_inputs_file(
            inputs, storage_client=storage_client
        )
        if generation is None:
            return model_inputs, None
        return model_inputs, {"generation": generation, "uri": str(inputs)}
    model_inputs = []
    for model_input in inputs:
        if not isinstance(model_input, ModelInput):
            raise ConfigError("inputs must be ModelInput values or an inputs file")
        model_inputs.append(
            replace(
                model_input,
                images=tuple(_absolute(image) for image in model_input.images),
            )
        )
    check_inputs(model_inputs, source="inputs")
    return model_inputs, None


def _absolute(image: ImageRef) -> ImageRef:
    if image.is_gcs:
        return image
    return replace(image, uri=str(Path(image.uri).expanduser().resolve()))


def _store_images(
    model_inputs: list[ModelInput],
    *,
    staging: Path,
    copy_images: bool,
    downloads: Path,
    storage_client: Any | None,
) -> list[ModelInput]:
    """Hash every distinct image once; copy it into the staging folder if asked.

    Images are read up to IMAGE_READ_WORKERS at a time, since reading many small
    files from GCS one by one is slow. The first image that fails, in input order,
    is the one reported.
    """

    uris = list(
        dict.fromkeys(image.uri for item in model_inputs for image in item.images)
    )
    hashed: dict[str, tuple[str, str | None]] = {}
    with ThreadPoolExecutor(max_workers=IMAGE_READ_WORKERS) as pool:
        futures = [
            pool.submit(_fetch_and_hash, uri, downloads, storage_client) for uri in uris
        ]
        try:
            for uri, future in zip(uris, futures, strict=True):
                source, digest = future.result()
                copied_name = None
                if copy_images:
                    copied_name = f"{IMAGES_DIR}/{digest}{_suffix(uri)}"
                    target = staging / copied_name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
                if uri.startswith("gs://"):
                    source.unlink(missing_ok=True)
                hashed[uri] = (digest, copied_name)
        except BaseException:
            pool.shutdown(cancel_futures=True)
            raise

    stored_inputs = []
    for model_input in model_inputs:
        images = []
        for image in model_input.images:
            digest, copied_name = hashed[image.uri]
            if image.sha256 is not None and image.sha256 != digest:
                raise ConfigError(
                    f"Image {image.uri} has SHA-256 {digest}, not the {image.sha256} "
                    "the inputs give"
                )
            stored_image = replace(image, sha256=digest)
            if copied_name is not None:
                stored_image = replace(
                    stored_image, uri=copied_name, source_uri=image.uri
                )
            images.append(stored_image)
        stored_inputs.append(replace(model_input, images=tuple(images)))
    return stored_inputs


def _fetch_and_hash(
    uri: str, downloads: Path, storage_client: Any | None
) -> tuple[Path, str]:
    source = _fetch(uri, downloads, storage_client)
    return source, file_sha256(source)


def _fetch(uri: str, downloads: Path, storage_client: Any | None) -> Path:
    if not uri.startswith("gs://"):
        path = Path(uri)
        if not path.is_file():
            raise ConfigError(f"Image {uri} does not exist")
        return path
    downloads.mkdir(parents=True, exist_ok=True)
    target = downloads / sha256(uri.encode("utf-8")).hexdigest()
    try:
        download_to_path(uri, target, storage_client=storage_client)
    except NotFound as e:
        raise ConfigError(f"Image {uri} does not exist") from e
    return target


def _suffix(uri: str) -> str:
    suffix = PurePosixPath(uri).suffix.lower()
    return suffix if _IMAGE_SUFFIX.fullmatch(suffix) else ""


def _existing_locally(
    folder: Path, *, inputs_text: str
) -> tuple[dict[str, Any] | None, bool]:
    """Return the folder's record if it has one, and whether its inputs match."""

    if not folder.exists():
        return None, False
    if not folder.is_dir():
        raise ConfigError(f"{folder} is not a folder")
    record_path = folder / EXPERIMENT_RECORD_NAME
    if record_path.exists():
        return _read_record(record_path), True
    names = {entry.name for entry in folder.iterdir() if not entry.name.startswith(".")}
    _check_only_experiment_files(names, location=str(folder))
    inputs_path = folder / INPUTS_FILE_NAME
    if not inputs_path.exists():
        return None, False
    if inputs_path.read_text(encoding="utf-8") != inputs_text:
        raise ConfigError(
            f"{inputs_path} is left from an unfinished creation with other inputs; "
            "remove it or choose another folder"
        )
    return None, True


def _existing_on_gcs(
    dest: str, *, inputs_text: str, storage_client: Any, pull_dir: Path
) -> tuple[dict[str, Any] | None, bool]:
    bucket_name, prefix = parse_gs_uri(dest.rstrip("/") + "/")
    prefix = prefix.rstrip("/") + "/"
    objects = {
        item.name.removeprefix(prefix): item
        for item in list_gcs_objects(
            bucket_name=bucket_name, prefix=prefix, storage_client=storage_client
        )
    }
    if EXPERIMENT_RECORD_NAME in objects:
        RunSync(pull_dir, dest, storage_client=storage_client).pull(
            relative_paths={EXPERIMENT_RECORD_NAME}
        )
        return _read_record(pull_dir / EXPERIMENT_RECORD_NAME), True
    _check_only_experiment_files(
        {name.split("/", 1)[0] for name in objects if name}, location=dest
    )
    remote_inputs = objects.get(INPUTS_FILE_NAME)
    if remote_inputs is None:
        return None, False
    pull_dir.mkdir(parents=True, exist_ok=True)
    local_inputs = pull_dir / INPUTS_FILE_NAME
    local_inputs.write_text(inputs_text, encoding="utf-8")
    if remote_inputs.crc32c != file_crc32c(local_inputs):
        raise ConfigError(
            f"{dest.rstrip('/')}/{INPUTS_FILE_NAME} is left from an unfinished "
            "creation with other inputs; remove it or choose another folder"
        )
    return None, True


def _check_only_experiment_files(names: set[str], *, location: str) -> None:
    others = sorted(names - {INPUTS_FILE_NAME, IMAGES_DIR})
    if others:
        raise ConfigError(
            f"{location} holds other files ({', '.join(others)}) but no "
            f"{EXPERIMENT_RECORD_NAME}; create the experiment in an empty folder"
        )


def _check_same_experiment(
    existing: Mapping[str, Any], record: Mapping[str, Any], *, location: str
) -> None:
    differences = [key for key in IDENTITY_KEYS if existing.get(key) != record[key]]
    if differences:
        raise ConfigError(
            f"{location} already holds experiment {existing.get('experiment_id')!r}, "
            f"which differs in {', '.join(differences)}; choose another folder"
        )
    if existing.get("max_tokens") != record["max_tokens"]:
        raise ConfigError(
            f"{location} already holds this experiment with max_tokens "
            f"{_limit_text(existing.get('max_tokens'))}, not "
            f"{_limit_text(record['max_tokens'])}. Creating it again never changes "
            f"the limit; {RAISE_MAX_TOKENS_HINT}"
        )


def _raised(
    record: Mapping[str, Any], max_tokens: int | None, *, location: str
) -> MaxTokensChange:
    current = record["max_tokens"]
    if max_tokens == current:
        return MaxTokensChange("unchanged", current, current, record)
    if _limit_rank(max_tokens) < _limit_rank(current):
        raise ConfigError(
            f"{location} has max_tokens {_limit_text(current)}; a limit can only be "
            f"raised, not lowered to {_limit_text(max_tokens)}"
        )
    history = [*record.get("max_tokens_history", [])]
    history.append({"max_tokens": max_tokens, "utc": utc_now()})
    raised = {**record, "max_tokens": max_tokens, "max_tokens_history": history}
    return MaxTokensChange("raised", current, max_tokens, raised)


def _limit_rank(max_tokens: int | None) -> float:
    return float("inf") if max_tokens is None else float(max_tokens)


def _limit_text(max_tokens: int | None) -> str:
    return "none (no limit)" if max_tokens is None else str(max_tokens)


def _read_record(path: Path) -> dict[str, Any]:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ConfigError(f"Could not read the experiment record {path}: {e}") from e
    required = (*IDENTITY_KEYS, "max_tokens", "max_tokens_history")
    if not isinstance(record, dict) or any(key not in record for key in required):
        raise ConfigError(f"{path} is not an experiment record")
    if record.get("record_format") != EXPERIMENT_RECORD_FORMAT:
        raise ConfigError(
            f"{path} has record format {record.get('record_format')!r}; this runner "
            f"reads format {EXPERIMENT_RECORD_FORMAT}"
        )
    return record


def _write_locally(staging: Path, folder: Path, *, skip_inputs: bool) -> None:
    """Write images, then inputs, then the record, each file atomically."""

    folder.mkdir(parents=True, exist_ok=True)
    images = staging / IMAGES_DIR
    if images.is_dir():
        (folder / IMAGES_DIR).mkdir(exist_ok=True)
        for image in sorted(images.iterdir()):
            target = folder / IMAGES_DIR / image.name
            if not target.exists():
                write_bytes_atomic(target, image.read_bytes())
    if not skip_inputs:
        write_bytes_atomic(
            folder / INPUTS_FILE_NAME, (staging / INPUTS_FILE_NAME).read_bytes()
        )
    write_bytes_atomic(
        folder / EXPERIMENT_RECORD_NAME,
        (staging / EXPERIMENT_RECORD_NAME).read_bytes(),
    )


def _upload(
    staging: Path, dest: str, *, skip_inputs: bool, storage_client: Any
) -> None:
    """Upload images, then inputs, then the record, never overwriting a file."""

    sync = RunSync(staging, dest, storage_client=storage_client)
    images = staging / IMAGES_DIR
    if images.is_dir():
        for image in sorted(images.iterdir()):
            sync.push_file(image)
    if not skip_inputs:
        sync.push_file(staging / INPUTS_FILE_NAME)
    sync.push_file(staging / EXPERIMENT_RECORD_NAME)

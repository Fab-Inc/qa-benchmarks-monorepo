"""Images at run time: fetched once per job and checked before any call.

A run sends the images of the calls it still has to make. Before its first call it
copies each one, once however many items and models use it, into a folder of its
own, and checks the copy against the SHA-256 recorded when the experiment was
created, and that Pillow can open it. A model then gets the item's input with each
image at its checked local copy, so what it sees is the file the experiment
recorded, never a source that changed in the meantime.
"""

from __future__ import annotations

import os
import shutil
import uuid
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

from google.api_core.exceptions import NotFound
from PIL import Image

from fab_benchmark_runner.errors import ConfigError
from fab_benchmark_runner.gcs import create_storage_client, download_to_path
from fab_benchmark_runner.inputs import ImageRef, ModelInput
from fab_benchmark_runner.results import file_sha256

FETCH_WORKERS = 32


class ImageFiles:
    """Checked local copies of an experiment's images, kept in ``folder``.

    ``experiment_location`` is where the experiment itself is, a local folder or a
    ``gs://`` URI: images copied into the experiment are referenced relative to it.
    """

    def __init__(
        self,
        folder: Path,
        *,
        experiment_location: str | Path,
        storage_client: Any | None = None,
    ) -> None:
        self.folder = Path(folder)
        self.experiment_location = str(experiment_location)
        self._storage_client = storage_client
        self._files: dict[str, Path] = {}

    def source(self, image: ImageRef) -> str:
        """Return where an image is read from: its URI, resolved if relative."""

        if image.is_gcs or Path(image.uri).is_absolute():
            return image.uri
        if self.experiment_location.startswith("gs://"):
            return f"{self.experiment_location.rstrip('/')}/{image.uri}"
        return str(Path(self.experiment_location) / image.uri)

    def fetch(self, images: Iterable[ImageRef]) -> None:
        """Copy and check every image not fetched yet, up to 32 at a time.

        Raises ConfigError naming the first image, in the given order, that is
        missing, differs from its recorded SHA-256, or can't be opened.
        """

        wanted: dict[str, ImageRef] = {}
        for image in images:
            if image.sha256 is None:
                raise ConfigError(f"Image {image.uri} has no recorded SHA-256")
            if image.sha256 not in self._files:
                wanted.setdefault(image.sha256, image)
        if not wanted:
            return
        self.folder.mkdir(parents=True, exist_ok=True)
        if self._storage_client is None and any(
            self.source(image).startswith("gs://") for image in wanted.values()
        ):
            self._storage_client = create_storage_client()
        with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
            futures = [
                (image, pool.submit(self._fetch_one, image))
                for image in wanted.values()
            ]
            try:
                for image, future in futures:
                    self._files[str(image.sha256)] = future.result()
            except BaseException:
                pool.shutdown(cancel_futures=True)
                raise

    def local_input(self, model_input: ModelInput) -> ModelInput:
        """Return the input with each image at its checked copy.

        Each image keeps its detail and SHA-256; its ``source_uri`` says where the
        copy came from.
        """

        if not model_input.images:
            return model_input
        return replace(
            model_input,
            images=tuple(
                ImageRef(
                    uri=str(self._file(image)),
                    detail=image.detail,
                    sha256=image.sha256,
                    source_uri=self.source(image),
                )
                for image in model_input.images
            ),
        )

    def _file(self, image: ImageRef) -> Path:
        try:
            return self._files[str(image.sha256)]
        except KeyError:
            raise RuntimeError(f"Image {image.uri} was not fetched") from None

    def _fetch_one(self, image: ImageRef) -> Path:
        source = self.source(image)
        target = self.folder / str(image.sha256)
        partial = self.folder / f".{image.sha256}.{uuid.uuid4().hex}.part"
        try:
            self._copy(source, partial)
            digest = file_sha256(partial)
            if digest != image.sha256:
                raise ConfigError(
                    f"Image {source} has changed since the experiment was created: "
                    f"its SHA-256 is {digest}, not the recorded {image.sha256}"
                )
            _check_opens(partial, source=source)
            os.replace(partial, target)
        finally:
            partial.unlink(missing_ok=True)
        return target

    def _copy(self, source: str, target: Path) -> None:
        if source.startswith("gs://"):
            try:
                download_to_path(source, target, storage_client=self._storage_client)
            except NotFound as e:
                raise ConfigError(f"Image {source} no longer exists") from e
            return
        try:
            shutil.copyfile(source, target)
        except FileNotFoundError as e:
            raise ConfigError(f"Image {source} no longer exists") from e


def _check_opens(path: Path, *, source: str) -> None:
    try:
        with Image.open(path) as picture:
            picture.verify()
    except Exception as e:  # Pillow raises many types for broken files
        # Pillow may quote the path, which doubles backslashes on Windows.
        reason = str(e)
        for form in (repr(str(path))[1:-1], str(path)):
            reason = reason.replace(form, source)
        raise ConfigError(
            f"Image {source} can't be opened as an image: {reason}"
        ) from e

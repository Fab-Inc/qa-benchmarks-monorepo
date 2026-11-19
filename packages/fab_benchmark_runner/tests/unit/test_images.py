"""Fetching images: where they are read from, the checks, and what calls get.

What runs do with images is described in ``tests/behaviour/test_images.py``; these
tests pin the mechanics underneath.
"""

from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from PIL import Image

from fab_benchmark_runner import ConfigError, ImageFiles, ImageRef, ModelInput


def _png(color: str = "red") -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (8, 6), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _ref(uri: str, content: bytes, detail: str = "high") -> ImageRef:
    return ImageRef(uri, detail, sha256(content).hexdigest())


@pytest.mark.parametrize(
    ("location", "uri", "source"),
    [
        ("/data/exp", "images/a.png", "/data/exp/images/a.png"),
        ("gs://bucket/exp/", "images/a.png", "gs://bucket/exp/images/a.png"),
        ("gs://bucket/exp", "/abs/a.png", "/abs/a.png"),
        ("/data/exp", "gs://other/a.png", "gs://other/a.png"),
    ],
)
def test_relative_uris_are_read_from_the_experiment(
    tmp_path: Path, location: str, uri: str, source: str
) -> None:
    files = ImageFiles(tmp_path, experiment_location=location)

    assert files.source(ImageRef(uri, "low", "a" * 64)) == source


def test_identical_files_are_fetched_once_whatever_their_uri(tmp_path: Path) -> None:
    content = _png()
    for name in ("a.png", "b.png"):
        (tmp_path / name).write_bytes(content)
    folder = tmp_path / "copies"
    files = ImageFiles(folder, experiment_location=tmp_path)

    files.fetch([_ref("a.png", content), _ref("b.png", content, "low")])

    assert [path.name for path in folder.iterdir()] == [sha256(content).hexdigest()]


def test_the_first_bad_image_in_order_is_reported(tmp_path: Path) -> None:
    good = _png()
    (tmp_path / "good.png").write_bytes(good)
    (tmp_path / "changed.png").write_bytes(_png("blue"))
    images = [
        _ref("good.png", good),
        _ref("missing.png", b"gone"),
        _ref("changed.png", _png("red") + b"x"),
    ]
    files = ImageFiles(tmp_path / "copies", experiment_location=tmp_path)

    with pytest.raises(ConfigError, match="missing.png no longer exists"):
        files.fetch(images)


def test_a_failed_check_leaves_no_partial_copy(tmp_path: Path) -> None:
    (tmp_path / "a.png").write_bytes(_png("blue"))
    folder = tmp_path / "copies"
    files = ImageFiles(folder, experiment_location=tmp_path)

    with pytest.raises(ConfigError, match="has changed since"):
        files.fetch([_ref("a.png", _png("red"))])

    assert list(folder.iterdir()) == []


def test_an_unreadable_image_is_named_by_its_source_not_its_copy(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.png").write_bytes(b"not an image")
    files = ImageFiles(tmp_path / "copies", experiment_location=tmp_path)

    with pytest.raises(ConfigError) as raised:
        files.fetch([_ref("a.png", b"not an image")])

    assert str(tmp_path / "a.png") in str(raised.value)
    assert ".part" not in str(raised.value)


def test_an_image_without_a_recorded_hash_is_refused(tmp_path: Path) -> None:
    files = ImageFiles(tmp_path, experiment_location=tmp_path)

    with pytest.raises(ConfigError, match="no recorded SHA-256"):
        files.fetch([ImageRef("a.png", "low")])


def test_a_call_gets_the_copy_with_its_source_detail_and_hash(tmp_path: Path) -> None:
    content = _png()
    (tmp_path / "a.png").write_bytes(content)
    files = ImageFiles(tmp_path / "copies", experiment_location=tmp_path)
    image = _ref("a.png", content, "low")
    files.fetch([image])
    text = ModelInput("t", "?")

    sent = files.local_input(ModelInput("v", "?", images=(image,)))

    [copy] = sent.images
    assert Path(copy.uri).read_bytes() == content
    assert (copy.detail, copy.sha256) == ("low", image.sha256)
    assert copy.source_uri == str(tmp_path / "a.png")
    assert files.local_input(text) is text


def test_gcs_is_only_opened_for_images_on_gcs(tmp_path: Path, storage: Any) -> None:
    content = _png()
    (tmp_path / "a.png").write_bytes(content)
    storage.objects["bucket"] = {"b.png": (1, content)}

    with patch(
        "fab_benchmark_runner.images.create_storage_client", return_value=storage
    ) as create:
        ImageFiles(tmp_path / "local", experiment_location=tmp_path).fetch(
            [_ref("a.png", content)]
        )
        create.assert_not_called()
        ImageFiles(tmp_path / "gcs", experiment_location=tmp_path).fetch(
            [_ref("gs://bucket/b.png", content)]
        )
        create.assert_called_once()

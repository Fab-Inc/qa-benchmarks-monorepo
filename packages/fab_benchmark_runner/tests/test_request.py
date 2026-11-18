"""Tests for model request values exposed to benchmark adapters."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import pytest
from PIL import Image

from fab_benchmark_runner import Benchmark, ImageInput, ModelRequest
from fab_benchmark_runner.request import prepare_request


@dataclass(frozen=True)
class ExampleParams:
    version: str = "v1"


def test_text_only_request_defaults_to_no_images() -> None:
    request = ModelRequest(prompt="What is 1 + 1?")

    assert request.prompt == "What is 1 + 1?"
    assert request.images == ()


def test_multimodal_request_preserves_image_order_and_metadata() -> None:
    first = ImageInput(data=b"first", detail="high", source_name="first.jpg")
    second = ImageInput(data=b"second", detail="low", source_name="second.png")

    request = ModelRequest(prompt="Compare the images", images=(first, second))

    assert request.images == (first, second)


@pytest.mark.parametrize("data", [b"", "not-bytes", None])
def test_image_data_must_be_nonempty_bytes(data: object) -> None:
    with pytest.raises(ValueError, match="non-empty bytes"):
        ImageInput(data=data)  # type: ignore[arg-type]


def test_image_detail_is_validated_at_runtime() -> None:
    with pytest.raises(ValueError, match="'low' or 'high'"):
        ImageInput(data=b"image", detail="auto")  # type: ignore[arg-type]


def test_request_images_must_be_an_image_tuple() -> None:
    with pytest.raises(TypeError, match="tuple of ImageInput"):
        ModelRequest(prompt="prompt", images=[ImageInput(data=b"image")])  # type: ignore[arg-type]


def test_benchmark_accepts_a_multimodal_request_builder() -> None:
    benchmark = Benchmark(
        name="visual",
        build_request=lambda _item, _params: ModelRequest(
            prompt="Look at this image",
            images=(ImageInput(data=b"image"),),
        ),
    )

    assert benchmark.build_request is not None
    assert benchmark.build_prompt is None


def test_build_request_does_not_change_existing_positional_fields() -> None:
    benchmark = Benchmark("text", lambda *_: "prompt", ExampleParams)

    assert benchmark.params_type is ExampleParams
    assert benchmark.build_request is None


@pytest.mark.parametrize(
    ("kwargs",),
    [
        ({},),
        (
            {
                "build_prompt": lambda _item, _params: "prompt",
                "build_request": lambda _item, _params: ModelRequest(prompt="prompt"),
            },
        ),
    ],
)
def test_benchmark_requires_exactly_one_request_builder(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        Benchmark(name="invalid", **kwargs)  # type: ignore[arg-type]


def test_prepare_request_records_source_and_fdllm_payload_hashes() -> None:
    source = _image_bytes(size=(3000, 1000))

    prepared = prepare_request(
        ModelRequest(
            prompt="Inspect the diagram",
            images=(ImageInput(source, detail="high", source_name="diagram.jpg"),),
        )
    )

    [image] = prepared.images
    assert (image.source_width, image.source_height) == (3000, 1000)
    assert (image.payload_width, image.payload_height) == (2048, 682)
    assert image.source_sha256 != image.payload_sha256
    assert image.identity() == {
        "sha256": image.payload_sha256,
        "detail": "high",
        "width": 2048,
        "height": 682,
    }
    assert image.provenance()["source_name"] == "diagram.jpg"
    assert image.to_llm_image().get_bytes() == image.payload


def test_prepared_image_creates_fresh_fdllm_images() -> None:
    [image] = prepare_request(
        ModelRequest("prompt", (ImageInput(_image_bytes()),))
    ).images

    first = image.to_llm_image()
    second = image.to_llm_image()

    assert first is not second
    assert first.Img is not second.Img
    assert first.get_bytes() == second.get_bytes() == image.payload


def test_source_name_does_not_change_image_identity() -> None:
    source = _image_bytes()
    first = prepare_request(
        ModelRequest("prompt", (ImageInput(source, source_name="first.jpg"),))
    )
    second = prepare_request(
        ModelRequest("prompt", (ImageInput(source, source_name="renamed.jpg"),))
    )

    assert first.image_identities() == second.image_identities()
    assert first.image_provenance() != second.image_provenance()


def test_prepare_request_rejects_undecodable_image_bytes() -> None:
    with pytest.raises(ValueError, match="Could not decode image 'broken.jpg'"):
        prepare_request(
            ModelRequest(
                "prompt",
                (ImageInput(b"not-an-image", source_name="broken.jpg"),),
            )
        )


def _image_bytes(*, size: tuple[int, int] = (12, 8)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, color="red").save(buffer, format="JPEG")
    return buffer.getvalue()

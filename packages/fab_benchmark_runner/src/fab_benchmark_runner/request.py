"""Model inputs shared by text-only and multimodal benchmarks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from typing import Any, Literal

from fdllm.llmtypes import LLMImage
from PIL import Image, UnidentifiedImageError

ImageDetail = Literal["low", "high"]


@dataclass(frozen=True)
class ImageInput:
    """One image supplied by a benchmark for a model request.

    ``data`` contains the original encoded image bytes. ``source_name`` is optional
    provenance only and must never affect the model-call identity.
    """

    data: bytes
    detail: ImageDetail = "high"
    source_name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.data, bytes) or not self.data:
            raise ValueError("ImageInput data must be non-empty bytes")
        if self.detail not in {"low", "high"}:
            raise ValueError("ImageInput detail must be 'low' or 'high'")
        if self.source_name is not None and (
            not isinstance(self.source_name, str) or not self.source_name.strip()
        ):
            raise ValueError("ImageInput source_name must be a non-empty string")


@dataclass(frozen=True)
class ModelRequest:
    """The ordered user input for one model call."""

    prompt: str
    images: tuple[ImageInput, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str):
            raise TypeError("ModelRequest prompt must be a string")
        if not isinstance(self.images, tuple) or not all(
            isinstance(image, ImageInput) for image in self.images
        ):
            raise TypeError("ModelRequest images must be a tuple of ImageInput values")


@dataclass(frozen=True)
class PreparedImage:
    """An image normalised exactly as FDLLM will send it to a provider."""

    payload: bytes
    detail: ImageDetail
    source_name: str | None
    source_sha256: str
    source_width: int
    source_height: int
    payload_sha256: str
    payload_width: int
    payload_height: int

    def to_llm_image(self) -> LLMImage:
        """Return a fresh FDLLM image so parallel calls share no mutable state."""

        image = Image.open(BytesIO(self.payload))
        image.load()
        return LLMImage(Img=image, Detail=self.detail)

    def identity(self) -> dict[str, Any]:
        """Return only fields that determine what the model receives."""

        return {
            "sha256": self.payload_sha256,
            "detail": self.detail,
            "width": self.payload_width,
            "height": self.payload_height,
        }

    def provenance(self) -> dict[str, Any]:
        """Return JSON-safe source and payload metadata for a result row."""

        return {
            "source_name": self.source_name,
            "source_sha256": self.source_sha256,
            "source_width": self.source_width,
            "source_height": self.source_height,
            "payload_sha256": self.payload_sha256,
            "payload_width": self.payload_width,
            "payload_height": self.payload_height,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class PreparedRequest:
    """A validated request ready for identity calculation and execution."""

    prompt: str
    images: tuple[PreparedImage, ...] = ()

    def image_identities(self) -> list[Mapping[str, Any]]:
        return [image.identity() for image in self.images]

    def image_provenance(self) -> list[Mapping[str, Any]]:
        return [image.provenance() for image in self.images]


def prepare_request(request: ModelRequest) -> PreparedRequest:
    """Normalise every image once, preserving its order in the request."""

    return PreparedRequest(
        prompt=request.prompt,
        images=tuple(prepare_image(image) for image in request.images),
    )


def prepare_image(image_input: ImageInput) -> PreparedImage:
    """Create FDLLM's provider payload and stable hashes for one source image."""

    try:
        source_image = Image.open(BytesIO(image_input.data))
        source_image.load()
    except (UnidentifiedImageError, OSError) as e:
        label = f" {image_input.source_name!r}" if image_input.source_name else ""
        raise ValueError(f"Could not decode image{label}: {e}") from e

    source_width, source_height = source_image.size
    llm_image = LLMImage(Img=source_image, Detail=image_input.detail)
    payload = llm_image.get_bytes()
    if payload is None or llm_image.Img is None:  # pragma: no cover - defensive
        raise ValueError("FDLLM did not produce an image payload")
    payload_width, payload_height = llm_image.Img.size

    return PreparedImage(
        payload=payload,
        detail=image_input.detail,
        source_name=image_input.source_name,
        source_sha256=sha256(image_input.data).hexdigest(),
        source_width=source_width,
        source_height=source_height,
        payload_sha256=sha256(payload).hexdigest(),
        payload_width=payload_width,
        payload_height=payload_height,
    )

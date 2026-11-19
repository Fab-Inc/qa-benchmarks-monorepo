"""Images: every model gets exactly the image files the experiment recorded.

Decisions this file describes:

- An item's images go into its one user message, in order, each with its detail
  level, low or high.
- Images are stored as originals. FDLLM resizes them as it sends them; that is
  treated as transport, like the rest of its request formatting. Each run records
  FDLLM's commit and Pillow's version, which is enough to rebuild what was sent.
- Before its first call, a run fetches the images of the calls it still has to
  make and checks each file against the SHA-256 recorded at creation, and that it
  is an image. An image that changed, disappeared or can't be opened stops the run
  before any call, naming the image: an answer to another file would be stored
  under the recorded file's call ID.
- A dry run fetches and checks the same images, so a broken image shows up before
  a launch.
- A job downloads each image once, however many items and models use it, and only
  the images of calls it still has to make. Nothing is kept between jobs, so every
  job checks the images it sends against their sources.
- Images copied into an experiment are read from its folder.
- A model the registry doesn't mark as taking images (``Vision: true``) is refused
  before any call.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest

from fab_benchmark_runner import ConfigError

from .conftest import Bench, FakeProviders, color_of, png

FIGURES = "gs://bucket/mathv/images"
VISUAL_INPUTS = (
    {
        "benchmark_item_id": "geometry-1",
        "prompt": "Which angle is largest?",
        "images": [{"uri": f"{FIGURES}/1.png", "detail": "high"}],
    },
    {
        "benchmark_item_id": "geometry-2",
        "prompt": "Compare the two figures.",
        "images": [
            {"uri": f"{FIGURES}/1.png", "detail": "high"},
            {"uri": f"{FIGURES}/2.png", "detail": "low"},
        ],
    },
)


@pytest.fixture
def figures(bench: Bench) -> None:
    """MathVision-style figures on GCS: 1.png is red, 2.png is blue."""

    bench.put(f"{FIGURES}/1.png", png("red"))
    bench.put(f"{FIGURES}/2.png", png("blue"))


def test_an_items_images_go_into_its_one_user_message_in_order(
    bench: Bench, providers: FakeProviders, figures: None
) -> None:
    bench.create(VISUAL_INPUTS)

    bench.run("model-a", through_fdllm=True)

    request = providers.request("Compare the two figures.")
    assert [message.Role for message in request.messages] == ["user"]
    sent = request.user_message.Images
    assert [(color_of(image.Img), image.Detail) for image in sent] == [
        ((255, 0, 0), "high"),
        ((0, 0, 255), "low"),
    ]


def test_images_are_stored_as_originals_and_fdllm_resizes_them_as_it_sends(
    bench: Bench, providers: FakeProviders
) -> None:
    """A 4096x1024 figure is recorded as it is; FDLLM sends it at 2048x512."""

    original = png("red", size=(4096, 1024))
    bench.put(f"{FIGURES}/wide.png", original)
    bench.create(
        [
            {
                "benchmark_item_id": "wide-1",
                "prompt": "Describe the figure.",
                "images": [{"uri": f"{FIGURES}/wide.png", "detail": "high"}],
            }
        ]
    )

    bench.run("model-a", through_fdllm=True)

    [image] = providers.request("Describe the figure.").user_message.Images
    assert image.Img.size == (2048, 512)
    assert bench.row("wide-1")["images"] == [
        {
            "detail": "high",
            "sha256": sha256(original).hexdigest(),
            "uri": f"{FIGURES}/wide.png",
        }
    ]
    assert "pillow" in bench.record()["runs"][0]["environment"]


def test_a_call_gets_each_image_as_a_local_copy_checked_against_the_experiment(
    bench: Bench, figures: None
) -> None:
    """What any call function receives, FDLLM's or a benchmark's own. The copies
    are the job's own and go when it ends."""

    bench.create(VISUAL_INPUTS)
    received = []

    def read_images(_model: str, model_input) -> None:
        for image in model_input.images:
            content = Path(image.uri).read_bytes()
            received.append((sha256(content).hexdigest(), image.sha256))

    bench.models.before_call = read_images

    bench.run("model-a", "model-b")

    assert len(received) == 6
    assert all(digest == recorded for digest, recorded in received)
    sent = [image for call in bench.models.inputs for image in call.images]
    assert all(image.source_uri.startswith(f"{FIGURES}/") for image in sent)
    assert not any(Path(image.uri).exists() for image in sent)


@pytest.mark.parametrize("change", ["overwritten", "deleted"])
def test_an_image_changed_since_creation_stops_the_run_before_any_call(
    bench: Bench, figures: None, change: str
) -> None:
    bench.create(VISUAL_INPUTS)
    if change == "overwritten":
        bench.put(f"{FIGURES}/2.png", png("green"))
    else:
        bench.remove(f"{FIGURES}/2.png")

    with pytest.raises(ConfigError, match=f"{FIGURES}/2.png"):
        bench.run("model-a")

    assert bench.models.calls == []


def test_a_file_that_is_not_an_image_stops_the_run_before_any_call(
    bench: Bench,
) -> None:
    """Creation only records the file's hash; the run is where it must open."""

    bench.put(f"{FIGURES}/1.png", b"not an image")
    bench.create(VISUAL_INPUTS[:1])

    with pytest.raises(ConfigError, match="can't be opened as an image"):
        bench.run("model-a")

    assert bench.models.calls == []


def test_a_dry_run_checks_the_images_a_run_would_send(
    bench: Bench, figures: None
) -> None:
    bench.create(VISUAL_INPUTS)
    assert bench.plan("model-a", "model-b").images_to_send == 2

    bench.put(f"{FIGURES}/1.png", png("green"))

    with pytest.raises(ConfigError, match="has changed since the experiment"):
        bench.plan("model-a")


def test_a_job_downloads_each_image_once_however_many_items_and_models_use_it(
    bench: Bench, figures: None
) -> None:
    bench.create(VISUAL_INPUTS)
    bench.storage.downloads.clear()

    bench.run("model-a", "model-b")

    assert sorted(bench.image_downloads()) == [
        "mathv/images/1.png",
        "mathv/images/2.png",
    ]


def test_a_run_downloads_only_the_images_of_calls_it_still_has_to_make(
    bench: Bench, figures: None
) -> None:
    """Retrying geometry-1 needs only its figure."""

    bench.create(VISUAL_INPUTS)
    bench.models.fail("geometry-1")
    bench.run("model-a")
    bench.models.recover("geometry-1")
    bench.storage.downloads.clear()

    bench.run("model-a", retry_errors=True)

    assert bench.image_downloads() == ["mathv/images/1.png"]


def test_images_copied_into_an_experiment_are_read_from_its_folder(
    bench: Bench,
) -> None:
    """Once copied, the original can go: jobs on GCS read the experiment's copy."""

    experiment = "gs://bucket/mathv/experiments/level1-v1"
    figure = bench.root / "figure.png"
    figure.write_bytes(png("red"))
    bench.create(
        [
            {
                "benchmark_item_id": "visual-1",
                "prompt": "Which shape is shown?",
                "images": [{"uri": "figure.png", "detail": "high"}],
            }
        ],
        dest=experiment,
        copy_images=True,
    )
    figure.unlink()

    bench.run("model-a", experiment=experiment)

    [downloaded] = bench.image_downloads()
    assert downloaded.startswith("mathv/experiments/level1-v1/images/")
    assert bench.models.called() == ["visual-1"]


def test_a_model_not_marked_as_taking_images_is_refused_before_any_call(
    bench: Bench, providers: FakeProviders, figures: None
) -> None:
    """FDLLM would raise on its first image; the run refuses before any call."""

    bench.create(VISUAL_INPUTS)
    providers.vision["model-b"] = False

    with pytest.raises(ConfigError, match="model-b"):
        bench.run("model-a", "model-b", through_fdllm=True)

    assert providers.requests == []

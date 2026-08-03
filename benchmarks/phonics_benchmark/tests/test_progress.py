"""Tests for benchmark run progress reporting."""

from collections.abc import Generator
from contextlib import contextmanager
from io import StringIO

from phonics_benchmark.progress import RichProgress
from rich.console import Console


def test_rich_progress_tracks_each_model_separately() -> None:
    """Concurrent models get their own bar, counted independently."""

    with _rich_progress() as (progress, buffer):
        progress.model_started("model-a", 2)
        progress.model_started("model-b", 3)
        progress.item_finished("model-a", error=False)
        progress.item_finished("model-b", error=False)
        progress.item_finished("model-b", error=False)

    output = buffer.getvalue()
    assert "model-a" in output
    assert "model-b" in output
    assert "1/2" in output
    assert "2/3" in output


def test_rich_progress_overall_total_grows_as_models_start() -> None:
    """The overall total is unknown up front, so each model adds its pending count."""

    with _rich_progress() as (progress, buffer):
        progress.model_started("model-a", 2)
        progress.model_started("model-b", 3)
        progress.item_finished("model-a", error=False)
        progress.item_finished("model-b", error=False)

    assert "2/5" in buffer.getvalue()


def test_rich_progress_labels_only_failing_models() -> None:
    """Error counts appear per model, and clean models stay unlabelled."""

    with _rich_progress() as (progress, buffer):
        progress.model_started("model-a", 2)
        progress.model_started("model-b", 1)
        progress.item_finished("model-a", error=True)
        progress.item_finished("model-a", error=True)
        progress.item_finished("model-b", error=False)

    output = buffer.getvalue()
    assert "2 err" in output
    assert output.count("err") == 1


@contextmanager
def _rich_progress() -> Generator[tuple[RichProgress, StringIO]]:
    """Render a RichProgress into a buffer instead of the terminal."""

    buffer = StringIO()
    progress = RichProgress(
        console=Console(file=buffer, force_terminal=True, width=120)
    )
    with progress:
        yield progress, buffer

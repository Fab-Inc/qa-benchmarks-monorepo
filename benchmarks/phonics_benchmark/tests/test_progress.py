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


def test_rich_progress_overall_counts_finished_models() -> None:
    """Models still queued for a worker count towards the overall denominator."""

    with _rich_progress(model_count=4) as (progress, buffer):
        progress.model_started("model-a", 2)
        progress.item_finished("model-a", error=False)
        progress.item_finished("model-a", error=False)
        progress.model_finished("model-a")

    assert "1/4" in buffer.getvalue()


def test_rich_progress_overall_ignores_items_within_a_model() -> None:
    """A part-finished model does not move the overall bar off zero."""

    with _rich_progress(model_count=4) as (progress, buffer):
        progress.model_started("model-a", 2)
        progress.item_finished("model-a", error=False)

    assert "0/4" in buffer.getvalue()


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


def test_rich_progress_display_height_does_not_grow_with_the_model_list() -> None:
    """Finished models free their row, so only running models hold the display."""

    with _rich_progress(model_count=50) as (progress, _buffer):
        for index in range(20):
            model_name = f"model-{index}"
            progress.model_started(model_name, 1)
            progress.item_finished(model_name, error=False)
            progress.model_finished(model_name)

        progress.model_started("model-last", 1)
        # The overall bar plus the single running model, after 20 completions.
        assert len(progress._progress.tasks) == 2


def test_rich_progress_logs_each_finished_model_above_the_display() -> None:
    """Removing a model's row keeps its result in scrollback, errors included."""

    with _rich_progress(model_count=2) as (progress, buffer):
        progress.model_started("model-a", 2)
        progress.item_finished("model-a", error=True)
        progress.item_finished("model-a", error=False)
        progress.model_finished("model-a")

    output = buffer.getvalue()
    assert "done model-a (2 items)" in output
    assert "1 err" in output


@contextmanager
def _rich_progress(
    model_count: int = 2,
) -> Generator[tuple[RichProgress, StringIO]]:
    """Render a RichProgress into a buffer instead of the terminal."""

    buffer = StringIO()
    progress = RichProgress(
        model_count,
        console=Console(file=buffer, force_terminal=True, width=120),
    )
    with progress:
        yield progress, buffer

"""Live progress reporting for phonics benchmark runs."""

from __future__ import annotations

from threading import Lock
from typing import Protocol, Self

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

_OVERALL_DESCRIPTION = "[bold]models[/bold]"

# A fixed bar keeps every row well short of the terminal. Full-width rows sit
# exactly on the wrap boundary, so narrowing the terminal by one column reflows
# all of them at once and the live display repaints over stale rows.
_BAR_WIDTH = 30


class ProgressReporter(Protocol):
    """Receives model and item lifecycle events from the benchmark runner.

    Reporters are context managers so implementations that own a terminal
    display can start and stop it around the run.
    """

    def __enter__(self) -> Self:
        """Start reporting."""
        ...

    def __exit__(self, *exc: object) -> None:
        """Stop reporting, restoring the terminal on success or failure."""
        ...

    def model_started(self, model_name: str, pending_count: int) -> None:
        """Announce that a model worker is about to evaluate pending_count items."""
        ...

    def item_finished(self, model_name: str, *, error: bool) -> None:
        """Record one evaluated item, whether or not the model call failed."""
        ...

    def model_finished(self, model_name: str) -> None:
        """Announce that a model worker evaluated every pending item."""
        ...


class NullProgress:
    """Discard every event so the runner needs no progress branches."""

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def model_started(self, model_name: str, pending_count: int) -> None:
        return None

    def item_finished(self, model_name: str, *, error: bool) -> None:
        return None

    def model_finished(self, model_name: str) -> None:
        return None


class RichProgress:
    """Render a bar per model, plus an overall bar, in one live display.

    Model bars are created as workers pick models up, so concurrent workers each
    get their own row. The overall bar counts finished models against the whole
    model list, so models still queued for a worker are in its denominator from
    the start.
    """

    def __init__(self, model_count: int, *, console: Console | None = None) -> None:
        self._progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=_BAR_WIDTH),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(compact=True),
            TextColumn("{task.fields[errors]}"),
            console=console if console is not None else Console(stderr=True),
        )
        self._lock = Lock()
        self._task_ids: dict[str, TaskID] = {}
        self._error_counts: dict[str, int] = {}
        self._overall_task_id = self._progress.add_task(
            _OVERALL_DESCRIPTION,
            total=model_count,
            errors="",
        )

    def __enter__(self) -> Self:
        self._progress.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._progress.stop()

    def model_started(self, model_name: str, pending_count: int) -> None:
        with self._lock:
            self._error_counts[model_name] = 0
            self._task_ids[model_name] = self._progress.add_task(
                model_name,
                total=pending_count,
                errors="",
            )

    def item_finished(self, model_name: str, *, error: bool) -> None:
        with self._lock:
            if error:
                self._error_counts[model_name] += 1
            error_count = self._error_counts[model_name]
            task_id = self._task_ids[model_name]
        self._progress.update(
            task_id,
            advance=1,
            errors=_error_label(error_count),
        )

    def model_finished(self, model_name: str) -> None:
        with self._lock:
            task_id = self._task_ids[model_name]
        self._progress.stop_task(task_id)
        self._progress.advance(self._overall_task_id)


def _error_label(error_count: int) -> str:
    if not error_count:
        return ""
    return f"[red]{error_count} err[/red]"

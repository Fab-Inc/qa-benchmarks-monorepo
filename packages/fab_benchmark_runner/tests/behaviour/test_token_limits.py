"""Token limits: ``max_tokens`` belongs to the experiment and never changes.

Decisions this file describes:

- ``max_tokens`` is set when an experiment is created, as a number or none (no
  limit, as the linguistic knowledge benchmark uses; FDLLM then sizes each call).
  Every call gets the experiment's limit, and every row records the limit it ran
  with.
- An experiment's limit never changes, so every result in it ran under the same
  limit. To give answers more room, create a new experiment with a higher limit, or
  none, and run the models on it; the old experiment stays as it was.
- Experiments that differ only in their limit are different experiments, so the
  default experiment ID depends on the limit.
- A response the limit left empty is an error row, with the stop reason when FDLLM
  reports one.
- A response cut off mid-answer counts as an answer and is scored as it is. FDLLM
  reports no stop reason for a response that isn't empty, so such responses can't
  be told apart yet (deferred).
"""

from __future__ import annotations

import pytest

from fab_benchmark_runner import ConfigError, create_experiment, read_experiment

from .conftest import Bench, FakeProviders


def test_every_call_gets_the_experiments_limit_and_every_row_records_it(
    bench: Bench,
) -> None:
    bench.create(max_tokens=1000)

    bench.run("model-a")

    assert {max_tokens for _, _, max_tokens in bench.models.calls} == {1000}
    assert {row["max_tokens"] for row in bench.rows()} == {1000}


def test_without_a_limit_fdllm_decides_how_long_each_answer_may_be(
    bench: Bench, providers: FakeProviders
) -> None:
    """FDLLM asks for what the model's context window leaves after the input,
    capped by the registry's Token_Limit_Completion."""

    bench.create(max_tokens=None)

    bench.run("model-a", through_fdllm=True)

    assert {request.max_tokens for request in providers.requests} == {None}
    assert {row["max_tokens"] for row in bench.rows()} == {None}


@pytest.mark.parametrize(
    ("empty_below", "stop_reason"),
    [("stop_reason", "length"), ("empty", None)],
    ids=["reported by FDLLM", "just empty"],
)
def test_a_response_the_limit_left_empty_is_an_error(
    bench: Bench, empty_below: str, stop_reason: str | None
) -> None:
    """FDLLM's dev branch reports the stop reason; its main branch returns empty
    text instead. Either way the row is an error."""

    bench.create(max_tokens=1000)
    bench.models.need_tokens("open-1", 3000, empty_below=empty_below)

    bench.run("model-a")

    cut = bench.row("open-1")
    assert (cut["call_status"], cut["stop_reason"]) == ("error", stop_reason)


def test_more_room_for_answers_means_a_new_experiment(bench: Bench) -> None:
    """The intended workflow: run, notice items hit the limit, create the
    experiment again with a higher limit elsewhere, and run there."""

    bench.create(dest="short", max_tokens=1000)
    bench.models.need_tokens("open-1", 3000)
    bench.run("model-a", experiment=str(bench.folder("short")))
    before = bench.rows(experiment="short")

    bench.create(dest="long", max_tokens=4000)
    bench.run("model-a", experiment=str(bench.folder("long")))

    longer = bench.row("open-1", experiment="long")
    assert (longer["call_status"], longer["max_tokens"]) == ("success", 4000)
    assert bench.rows(experiment="short") == before
    assert read_experiment(bench.folder("short")).max_tokens == 1000


def test_creating_the_experiment_again_never_changes_its_limit(bench: Bench) -> None:
    location = bench.create(max_tokens=1000)

    with pytest.raises(ConfigError, match="limit never changes"):
        create_experiment(
            bench.write_inputs(), dest=location, benchmark="lk", max_tokens=4000
        )
    assert read_experiment(location).max_tokens == 1000


def test_an_answer_cut_off_mid_sentence_counts_as_an_answer(bench: Bench) -> None:
    """It is kept, and scored, as it is: only a new experiment with a higher limit
    gets a longer answer."""

    bench.create(max_tokens=1000)
    bench.models.cut_off("open-1", 3000)
    bench.run("model-a")

    bench.models.forget_calls()
    bench.run("model-a", retry_errors=True)

    row = bench.row("open-1")
    assert bench.models.calls == []
    assert (row["call_status"], row["output"]) == ("success", "The answer is")


def test_experiments_that_differ_only_in_their_limit_have_different_ids(
    bench: Bench,
) -> None:
    bench.create(dest="short", max_tokens=1000)
    bench.create(dest="long", max_tokens=None)

    assert (
        read_experiment(bench.folder("short")).experiment_id
        != read_experiment(bench.folder("long")).experiment_id
    )

"""Token limits: ``max_tokens`` belongs to the experiment and can only be raised.

Decisions this file describes:

- ``max_tokens`` is set when an experiment is created, as a number or none (no
  limit). Every call gets the experiment's current limit, and every row records the
  limit it ran with.
- The limit is not part of call IDs. When some items hit it, raising the limit in
  place keeps every result that finished within the old one; a run with
  ``retry_errors`` then reruns only the error rows, which include responses the
  limit left empty.
- The limit can only be raised: an answer that finished above a lower limit is not
  one a model could give under it. The experiment keeps each limit it had.
- Creating an experiment again never changes its limit; only raising does.
- A response cut off mid-answer counts as an answer and is scored as it is. FDLLM
  reports no stop reason for a response that isn't empty, so such responses can't
  be told apart yet (deferred).
"""

from __future__ import annotations

import pytest

from fab_benchmark_runner import (
    ConfigError,
    create_experiment,
    raise_max_tokens,
    read_experiment,
)

from .conftest import Bench


def test_every_call_gets_the_experiments_limit_and_every_row_records_it(
    bench: Bench,
) -> None:
    bench.create(max_tokens=1000)

    bench.run("model-a")

    assert {max_tokens for _, _, max_tokens in bench.models.calls} == {1000}
    assert {row["max_tokens"] for row in bench.rows()} == {1000}


def test_raising_the_limit_reruns_only_the_answers_it_cut_off(bench: Bench) -> None:
    """The intended workflow: run, notice items hit the limit, raise it, retry."""

    bench.create(max_tokens=1000)
    bench.models.need_tokens("open-1", 3000)
    bench.run("model-a", "model-b")
    cut = bench.row("open-1")
    assert (cut["call_status"], cut["stop_reason"]) == ("error", "length")
    kept = bench.row("mcq-1")

    raise_max_tokens(bench.folder(), 4000)
    bench.models.forget_calls()
    bench.run("model-a", "model-b", retry_errors=True)

    assert bench.models.calls == [
        ("model-a", "open-1", 4000),
        ("model-b", "open-1", 4000),
    ]
    rerun = bench.row("open-1")
    assert (rerun["call_status"], rerun["max_tokens"]) == ("success", 4000)
    assert rerun["result_id"] == cut["result_id"]
    assert bench.row("mcq-1") == kept


def test_an_answer_the_limit_left_empty_is_retried_even_without_a_stop_reason(
    bench: Bench,
) -> None:
    """FDLLM's main branch returns empty text instead of reporting the stop reason;
    an empty answer is an error either way, so the same workflow applies."""

    bench.create(max_tokens=1000)
    bench.models.need_tokens("open-1", 3000, empty_below="empty")
    bench.run("model-a")
    assert bench.row("open-1")["stop_reason"] is None

    raise_max_tokens(bench.folder(), 4000)
    bench.models.forget_calls()
    bench.run("model-a", retry_errors=True)

    assert bench.models.called() == ["open-1"]
    assert bench.row("open-1")["call_status"] == "success"


def test_an_answer_cut_off_mid_sentence_counts_as_an_answer(bench: Bench) -> None:
    """It is kept, and scored, as it is: raising the limit doesn't rerun it."""

    bench.create(max_tokens=1000)
    bench.models.cut_off("open-1", 3000)
    bench.run("model-a")

    raise_max_tokens(bench.folder(), 4000)
    bench.models.forget_calls()
    bench.run("model-a", retry_errors=True)

    assert bench.models.calls == []
    assert bench.row("open-1")["output"] == "The answer is"


def test_the_limit_can_only_be_raised(bench: Bench) -> None:
    bench.create(max_tokens=1000)

    with pytest.raises(ConfigError, match="can only be raised"):
        raise_max_tokens(bench.folder(), 500)
    assert raise_max_tokens(bench.folder(), None).status == "raised"
    with pytest.raises(ConfigError, match="can only be raised"):
        raise_max_tokens(bench.folder(), 100_000)


def test_the_experiment_keeps_each_limit_it_had(bench: Bench) -> None:
    bench.create(max_tokens=1000)

    raise_max_tokens(bench.folder(), 4000)
    unchanged = raise_max_tokens(bench.folder(), 4000)

    record = read_experiment(bench.folder()).record
    assert unchanged.status == "unchanged"
    assert record["max_tokens"] == 4000
    assert [entry["max_tokens"] for entry in record["max_tokens_history"]] == [
        1000,
        4000,
    ]


def test_creating_the_experiment_again_never_changes_its_limit(bench: Bench) -> None:
    location = bench.create(max_tokens=1000)

    with pytest.raises(ConfigError, match="raise-max-tokens"):
        create_experiment(
            bench.write_inputs(), dest=location, benchmark="lk", max_tokens=4000
        )
    assert read_experiment(location).max_tokens == 1000


def test_the_default_experiment_id_does_not_depend_on_the_limit(bench: Bench) -> None:
    """Raising the limit keeps the experiment, so its ID can't depend on it."""

    bench.create(dest="short", max_tokens=1000)
    bench.create(dest="long", max_tokens=None)

    assert (
        read_experiment(bench.folder("short")).experiment_id
        == read_experiment(bench.folder("long")).experiment_id
    )

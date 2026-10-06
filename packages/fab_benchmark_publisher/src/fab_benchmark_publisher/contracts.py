from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class LeaderboardRecord:
    """Canonical record emitted by each benchmark before website merge."""

    benchmark_domain: str
    model: str
    accuracy: float
    provider: str | None = None
    subject: str | None = None
    level: str | None = None
    bad_format: float | None = None
    n: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkPair:
    """One column of the leaderboard: a benchmark x language pair.

    `benchmark_domain` is the id the website keys everything by ("LK-sw");
    `display_name` is the selector label a visitor reads ("LK Kiswahili").
    """

    benchmark_domain: str
    display_name: str


@dataclass(frozen=True)
class RecordsFile:
    """One benchmark's contribution to the leaderboard, as written to disk or GCS.

    The envelope, rather than a bare list of records, is what lets the cluster
    publisher discover contributions by globbing: each file names the column it
    fills, so two files claiming one benchmark_domain is a detectable error rather
    than a silent last-write-wins, and `source`/`generated_at` say what is about to
    be published before it is.
    """

    benchmark: BenchmarkPair
    records: list[LeaderboardRecord]
    source: str | None = None
    generated_at: str | None = None


@dataclass(frozen=True)
class LeaderboardBundle:
    """Everything one leaderboard publish consists of.

    `payloads` maps the <file-name> segment of the upload filenames
    (<timestamp>_<cluster-name>_<file-name>.json) to its payload. `default_models`
    is the benchmark_domain -> model ids audit record written alongside them; it is
    NOT an upload file.

    `dropped` is model id -> why it was left out, for every model that had a score but
    could not be published. A dropped model is invisible on the site, so the reason
    has to survive past the warning printed during the build -- the manifest is where
    it ends up, and that is what makes "why is this model not on the leaderboard?"
    answerable later.
    """

    payloads: dict[str, Any]
    default_models: dict[str, list[str]]
    dropped: dict[str, str] = field(default_factory=dict)

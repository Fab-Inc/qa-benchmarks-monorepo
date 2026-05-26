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

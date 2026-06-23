"""Result persistence for phonics benchmark runs."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from phonics_benchmark.config import ConfigError


def load_results(file_name: str | Path, *, force: bool = False) -> pd.DataFrame:
    """Load existing JSONL results, or return an empty frame."""

    path = Path(file_name)
    if force or not path.exists():
        return pd.DataFrame()

    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        return pd.DataFrame()

    results = pd.DataFrame(rows)
    if "result_id" not in results.columns:
        raise ConfigError(f"Existing results at {path} lack result_id")
    return results


def save_results(
    results: pd.DataFrame,
    file_name: str | Path,
    config_representation: dict[str, Any],
    run_id: str,
    custom_models_path: str | Path | None,
) -> Path:
    """Persist benchmark results.

    This is the persistence boundary for future result storage. For now it writes
    JSONL locally and accepts the run/config/model metadata that a richer
    backend will need.
    """

    path = Path(file_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = results.to_dict(orient="records")
    content = "\n".join(_json_dumps(row) for row in records)
    _write_text_atomic(path, content + ("\n" if content else ""))
    return path


def _write_text_atomic(path: Path, content: str) -> None:
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    tmp_path = Path(tmp_name)

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            tmp_file.write(content)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)

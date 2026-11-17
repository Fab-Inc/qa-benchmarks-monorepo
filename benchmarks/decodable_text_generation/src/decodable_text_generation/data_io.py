"""Small file IO helpers shared by the generation item builders."""

from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel


def write_jsonl(records: Iterable[BaseModel], path: Path) -> None:
    """Write Pydantic model instances to a JSONL file, one record per line."""
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(record.model_dump_json())
            file.write("\n")

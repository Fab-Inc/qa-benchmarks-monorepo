"""Loading of benchmark items for rcqg runs."""

from pathlib import Path
from typing import Any

from phonics_benchmark.items import load_benchmark_items

# What a run needs from an item. The shared loader requires ``benchmark_item_id``, names
# the file and line for malformed, blank, or duplicate lines, and drops any other key.
ITEM_REQUIRED_FIELDS: dict[str, type] = {
    "benchmark_item_id": str,
    "text_content": str,
    "language": str,
}


def load_items(items_path: str | Path) -> tuple[list[dict[str, Any]], str]:
    """Load benchmark items; the harness's ``item_loader``.

    Items have exactly the three required fields and no extras, so descriptive
    metadata can never reach a prompt through the item. Returns the items and the
    SHA-256 of the exact bytes read, which the harness records in the run's config.
    """
    return load_benchmark_items(items_path, required_fields=ITEM_REQUIRED_FIELDS)

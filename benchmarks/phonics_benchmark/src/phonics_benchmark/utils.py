from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def require_columns(columns: list[str]):
    """Decorator to require certain columns in the first DataFrame argument."""

    def decorator(func):
        def wrapper(df: pd.DataFrame, *args, **kwargs):
            missing = set(columns) - set(df.columns)
            if missing:
                raise ValueError(
                    f"DataFrame is missing required column(s): {sorted(missing)}"
                )
            return func(df, *args, **kwargs)

        return wrapper

    return decorator


def require_single_language_iso_code(
    codes: Iterable[str],
    *,
    source: str,
    expected: str | None = None,
) -> str:
    """Return the one language ISO code shared by ``codes``.

    Raises ValueError if there are no codes and no ``expected``, if the codes
    hold more than one language, or if they disagree with ``expected``.
    """
    unique = {code.strip() for code in codes if code and code.strip()}
    if len(unique) > 1:
        raise ValueError(
            f"Expected one language_iso_code across {source}, found: "
            f"{', '.join(sorted(unique))}"
        )
    if expected is None:
        if not unique:
            raise ValueError(f"Cannot resolve a language_iso_code: none in {source}")
        return unique.pop()
    if unique and expected not in unique:
        raise ValueError(
            f"Requested language '{expected}' does not match the "
            f"language_iso_code '{unique.pop()}' in {source}"
        )
    return expected

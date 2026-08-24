from __future__ import annotations

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

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


def warn_fully_decodable_items(threshold: float = 0.1):
    """Decorator to raise a warning if a significant fraction of items are fully
    decodable."""

    def decorator(func):
        def wrapper(benchmark_items: pd.DataFrame, *args, **kwargs):
            n_fully_decodable = (
                benchmark_items["nondecodable_words"].apply(len).eq(0).sum()
            )
            percent_fully_decodable = n_fully_decodable / len(benchmark_items)
            if percent_fully_decodable > threshold:
                import warnings

                warnings.warn(
                    f"{percent_fully_decodable:.0%} of the items are fully decodable.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                import logging

                logging.info(
                    f"{percent_fully_decodable:.0%} of the items are fully decodable."
                )
            return func(benchmark_items, *args, **kwargs)

        return wrapper

    return decorator

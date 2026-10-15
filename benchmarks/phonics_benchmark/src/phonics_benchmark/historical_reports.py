"""Read the saved September report layout for historical analysis consumers.

New reporting uses reporting.py; this adapter never scores or plots data.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from phonics_benchmark.reporting import read_csv


def load_plot_table(run: Path, metadata: Path) -> tuple[pd.DataFrame, dict]:
    summary = read_csv(run / "report/summary_by_model.csv").rename(
        columns={"model_name": "model"}
    )
    ci = read_csv(run / "report/bootstrap_ci.csv")
    coverage = read_csv(run / "coverage_by_model.csv")
    models = read_csv(metadata / "models.csv").set_index("model_id")
    providers = read_csv(metadata / "providers.csv").set_index("provider")
    for table in (models, providers):
        if not table.index.is_unique:
            raise ValueError("Metadata identifiers must be unique")
    overrides = json.loads((metadata / "display_overrides.json").read_text())
    for model, override in overrides.items():
        models.loc[model, "display_name"] = override["display_name"]
        models.loc[model, "provider"] = override["provider"]

    values = summary[["model", "pooled_f2", "n_items"]].merge(
        ci[
            [
                "model",
                "pooled_f2",
                "pooled_f2_ci_low",
                "pooled_f2_ci_high",
                "n_bootstrap",
                "ci_level",
            ]
        ],
        on="model",
        validate="one_to_one",
        how="outer",
        suffixes=("", "_bootstrap"),
    )
    np.testing.assert_allclose(
        values.pooled_f2, values.pooled_f2_bootstrap, rtol=0, atol=1e-14
    )
    values = values.drop(columns="pooled_f2_bootstrap").merge(
        coverage[
            [
                "model",
                "expected_items",
                "scorable_items",
                "call_failed",
                "parse_failed",
                "missing_responses",
            ]
        ],
        on="model",
        how="left",
        validate="one_to_one",
    )
    if not values.n_items.eq(values.scorable_items).all():
        raise ValueError("Scored item counts disagree with coverage")
    values = values.merge(
        models[["display_name", "provider"]],
        left_on="model",
        right_index=True,
        how="left",
        validate="one_to_one",
    )
    if values[["display_name", "provider"]].isna().any().any():
        raise ValueError("Some scored models lack display metadata")
    colors = providers.color.to_dict()
    values["color"] = values.provider.map(colors)
    if values.color.isna().any():
        raise ValueError("Some providers lack saved colours")
    values = values.rename(
        columns={
            "pooled_f2": "score",
            "pooled_f2_ci_low": "ci_low",
            "pooled_f2_ci_high": "ci_high",
        }
    )
    if (
        not values[["score", "ci_low", "ci_high"]]
        .apply(lambda c: c.between(0, 1))
        .all()
        .all()
    ):
        raise ValueError("F2 estimates and intervals must be defined within [0, 1]")
    if not values.ci_low.le(values.ci_high).all():
        raise ValueError("Reversed confidence interval")
    for column in ("ci_level", "n_bootstrap", "expected_items"):
        if values[column].nunique() != 1:
            raise ValueError(f"Inconsistent {column}")
    return values.sort_values(["score", "model"], ascending=[False, True]).reset_index(
        drop=True
    ), colors

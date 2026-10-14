"""Check worked metric examples and render a standalone HTML report, offline.

uv run --frozen --package phonics-benchmark python \
  benchmarks/phonics_benchmark/scripts/build_metric_examples.py

Expected answers below are handwritten. Production functions compute actual values.
The report is written even when a numeric check fails; the command then exits 1.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import math
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import analyze_balanced_accuracy as ba
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analyze_phonics_hardware_tiers import minimum_scorable_items
from matplotlib.container import ErrorbarContainer
from phonics_benchmark.bootstrap import bootstrap_metric_cis
from phonics_benchmark.scoring import (
    add_per_item_confusion_matrix,
    summarize_by_model,
)
from plot_adjusted_balanced_accuracy import plot_adjusted_leaderboard
from plot_report_f2_leaderboards import eligible_models, plot_leaderboard

# Counts are TP, FP (including out-of-text words), FN, TN.
# Scores are F1, F2, balanced accuracy, adjusted BA points.
# These constants must be reviewed, never regenerated from the implementation.
CASES = [
    (
        "Perfect",
        "cat dog",
        (2, 0, 0, 2),
        (1, 1, 1, 100),
        "Both gold words found; neither negative word flagged.",
    ),
    (
        "Flag everything",
        "cat dog sun moon",
        (2, 2, 0, 0),
        (2 / 3, 5 / 6, 1 / 2, 0),
        "Recall is 1, specificity is 0. F1 = 4/6; F2 = 10/12.",
    ),
    (
        "Flag nothing",
        "",
        (0, 0, 2, 2),
        (0, 0, 1 / 2, 0),
        "Recall is 0, specificity is 1. Neither gold word is found.",
    ),
    (
        "Miss one gold word",
        "cat",
        (1, 0, 1, 2),
        (2 / 3, 5 / 9, 3 / 4, 50),
        "F1 = 2/3; F2 = 5/(5+4). BA = (1/2 + 1)/2.",
    ),
    (
        "Flag one extra text word",
        "cat dog sun",
        (2, 1, 0, 1),
        (4 / 5, 10 / 11, 3 / 4, 50),
        "F1 = 4/5; F2 = 10/11. BA = (1 + 1/2)/2.",
    ),
    (
        "Add an out-of-text word",
        "cat dog fish",
        (2, 1, 0, 2),
        (4 / 5, 10 / 11, 1, 100),
        "Fish penalizes F1/F2. Fixed-vocabulary BA excludes it "
        "and audits it separately.",
    ),
    (
        "Invert every label",
        "sun moon",
        (0, 2, 2, 0),
        (0, 0, 0, -100),
        "Both recall and specificity are zero; adjusted BA retains the negative score.",
    ),
    (
        "Repeat and capitalize",
        "CAT cat DOG dog",
        (2, 0, 0, 2),
        (1, 1, 1, 100),
        "Vocabulary and predictions are lowercase sets, so repetition adds no weight.",
    ),
]


class Checks:
    def __init__(self):
        self.results = []

    def equal(self, name, actual, expected):
        try:
            np.testing.assert_allclose(
                np.asarray(actual, dtype=float),
                np.asarray(expected, dtype=float),
                rtol=0,
                atol=1e-12,
                equal_nan=True,
            )
        except AssertionError as error:
            self.results.append((name, False, str(error)))
        else:
            self.results.append((name, True, ""))

    def true(self, name, condition):
        self.results.append(
            (name, bool(condition), "" if condition else "Condition failed")
        )

    @property
    def failures(self):
        return [row for row in self.results if not row[1]]


def dataset(items):
    """Use the production vocabulary/gold loader, including its validation."""
    with tempfile.TemporaryDirectory(prefix="phonics-examples-") as tmp:
        path = Path(tmp) / "items.jsonl"
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in items), encoding="utf-8"
        )
        return ba.load_items(path, "en")


def item(key="four", content="cat dog sun moon", gold=("cat", "dog")):
    return {
        "benchmark_item_id": key,
        "content": content,
        "nondecodable_words": list(gold),
    }


def score(items, predictions):
    inputs = [
        {
            **row,
            "model": "example",
            "response_parsed": predictions[row["benchmark_item_id"]],
        }
        for row in items
        if row["benchmark_item_id"] in predictions
    ]
    scored = add_per_item_confusion_matrix(pd.DataFrame(inputs), "en")
    summary = summarize_by_model(scored).iloc[0]
    counts = ba.vocabulary_counts(scored, dataset(items))
    balanced = ba.metrics_from_totals(counts[ba.COUNTS].sum().to_numpy())
    return (
        scored,
        summary,
        [summary.pooled_f1, summary.pooled_f2, balanced[2], balanced[3]],
    )


def display(value):
    if isinstance(value, (float, np.floating)):
        return "undefined" if math.isnan(value) else f"{value:.6g}"
    return str(value)


def table(headers, rows):
    def cell(value, tag):
        return f"<{tag}>{html.escape(display(value))}</{tag}>"

    head = "".join(cell(v, "th") for v in headers)
    body = "".join(
        "<tr>" + "".join(cell(v, "td") for v in row) + "</tr>" for row in rows
    )
    return (
        f'<div class="table-scroll"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
    )


def worked_cases(checks):
    result, explanations = [], []
    for name, prediction, expected_counts, expected, explanation in CASES:
        content = (
            "cat cat DOG dog sun moon moon"
            if name == "Repeat and capitalize"
            else "cat dog sun moon"
        )
        scored, _, actual = score([item(content=content)], {"four": prediction.split()})
        counts = (
            scored[["tp_count", "fp_count", "fn_count", "tn_count"]].iloc[0].tolist()
        )
        checks.equal(f"{name}: confusion counts", counts, expected_counts)
        checks.equal(f"{name}: four metrics", actual, expected)
        result.extend([(name, "Expected", *expected), (name, "Computed", *actual)])
        explanations.append(
            (
                name,
                prediction or "(empty)",
                "/".join(map(str, expected_counts)),
                explanation,
            )
        )
    for name, gold, prediction, counts, expected in [
        ("No positive class, empty prediction", [], [], (0, 0, 0, 4), [math.nan] * 4),
        (
            "No negative class, perfect prediction",
            ["cat", "dog", "sun", "moon"],
            ["cat", "dog", "sun", "moon"],
            (4, 0, 0, 0),
            [1, 1, math.nan, math.nan],
        ),
    ]:
        scored, _, actual = score([item(gold=gold)], {"four": prediction})
        checks.equal(
            name + ": confusion counts",
            scored[["tp_count", "fp_count", "fn_count", "tn_count"]].iloc[0],
            counts,
        )
        checks.equal(name + ": undefined denominators", actual, expected)
        result.extend([(name, "Expected", *expected), (name, "Computed", *actual)])
    return (
        "<h2>Single item examples</h2><p>The text is <code>cat dog sun moon</code> and "
        "the gold nondecodable words are <code>cat dog</code>, except in the "
        "explicitly "
        "named repetition and missing-class cases. A positive means a word flagged as "
        "nondecodable. Counts are TP / FP / FN / TN.</p>"
        + table(
            ["Example", "Prediction", "Hand-counted totals", "Reasoning"], explanations
        )
        + table(["Example", "Result", "F1", "F2", "BA", "Adjusted BA points"], result)
        + "<p>F1 = 2TP / (2TP + FP + FN). F2 = 5TP / (5TP + FP + 4FN). "
        "BA = (TP/(TP+FN) + TN/(TN+FP<sub>in text</sub>))/2. "
        "Adjusted BA = 100 × (2BA − 1). Undefined denominators stay undefined; "
        "an empty correct answer can have perfect-match accuracy 1 "
        "while F1 is undefined.</p>"
    )


def pooled_and_missing(checks):
    items = [
        item("small", "cat dog", ["cat"]),
        item("large", "sun moon rain wind snow ice", ["sun", "moon", "rain", "wind"]),
    ]
    scored, summary, actual = score(items, {"small": ["cat"], "large": []})
    checks.equal("Unequal items: pooled metrics", actual, [1 / 3, 5 / 21, 3 / 5, 20])
    checks.equal("Unequal items: item-averaged F1", summary.macro_f1, 1 / 2)
    bootstrap = bootstrap_metric_cis(scored, n_bootstrap=1000, seed=0).iloc[0]
    checks.equal(
        "Bootstrap: F1 and F2 points and bounds",
        bootstrap[
            [
                "pooled_f1",
                "pooled_f1_ci_low",
                "pooled_f1_ci_high",
                "pooled_f2",
                "pooled_f2_ci_low",
                "pooled_f2_ci_high",
            ]
        ],
        [1 / 3, 0, 1, 5 / 21, 0, 1],
    )
    balanced, _ = ba.summarize(scored, dataset(items), n_bootstrap=1000, seed=0)
    checks.equal(
        "Bootstrap: BA and adjusted bounds",
        balanced.iloc[0][
            [
                "balanced_accuracy_ci_low",
                "balanced_accuracy_ci_high",
                "adjusted_balanced_accuracy_ci_low",
                "adjusted_balanced_accuracy_ci_high",
            ]
        ],
        [1 / 2, 1, 0, 100],
    )
    missing, _, missing_scores = score(items, {"small": ["cat"]})
    missing_summary, _ = ba.summarize(missing, dataset(items), n_bootstrap=1000, seed=0)
    checks.equal("Missing item: scores", missing_scores, [1, 1, 1, 100])
    checks.equal(
        "Missing item: full denominator",
        missing_summary.iloc[0][["n_scorable", "expected_items", "coverage"]],
        [1, 2, 1 / 2],
    )
    return (
        "<h2>Pooling, missing responses and bootstrap intervals</h2>"
        "<p>The small item has text <code>cat dog</code>, gold <code>cat</code>, and a "
        "perfect prediction. The large item has six unique words, four gold words, "
        "and an empty prediction. Their totals are TP=1, FP=0, FN=4, TN=3.</p>"
        + table(
            ["Quantity", "Expected", "Computed"],
            [
                ("Pooled F1: 2/(2+4)", 1 / 3, actual[0]),
                ("Item-averaged F1: (1+0)/2", 1 / 2, summary.macro_f1),
                ("Pooled F2: 5/(5+16)", 5 / 21, actual[1]),
                ("BA: (1/5+1)/2", 3 / 5, actual[2]),
                ("Adjusted BA", 20, actual[3]),
                ("F1 if the large response is missing", 1, missing_scores[0]),
                (
                    "Coverage if the large response is missing",
                    1 / 2,
                    missing_summary.coverage.iloc[0],
                ),
            ],
        )
        + "<p>Missing responses contribute no confusion counts: the scored subset can "
        "look perfect while coverage is only 50%. Coverage keeps "
        "the full item denominator.</p>"
        "<p>For an item bootstrap of size two, the possible draws are small/small, "
        "small/large, large/small and large/large, with probabilities ¼ each. "
        "F1 therefore "
        "takes 1, ⅓, ⅓, 0; F2 takes 1, 5/21, 5/21, 0; BA takes 1, ⅗, ⅗, ½. "
        "The 1,000-draw seed-0 production bootstrap gives 95% bounds [0,1] for F1/F2 "
        "and [½,1] for BA, hence [0,100] adjusted points. These deliberately tiny "
        "examples explain behaviour; they are not realistic uncertainty estimates.</p>"
    )


def figure_html(fig, description):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return (
        f'<figure><img alt="{html.escape(description)}" '
        f'src="data:image/png;base64,{encoded}">'
        f"<figcaption>{html.escape(description)}</figcaption></figure>"
    )


def check_artists(checks, fig, name, scores, lows, highs, labels):
    ax = fig.axes[0]
    checks.equal(name + ": bar values", [bar.get_width() for bar in ax.patches], scores)
    checks.equal(
        name + ": bar positions",
        [bar.get_y() + bar.get_height() / 2 for bar in ax.patches],
        [0, 1, 2],
    )
    checks.true(
        name + ": model order",
        [label.get_text() for label in ax.get_yticklabels()] == labels,
    )
    errors = [
        container
        for container in ax.containers
        if isinstance(container, ErrorbarContainer)
    ]
    segments = errors[0].lines[2][0].get_segments()
    checks.equal(name + ": lower interval endpoints", [s[0, 0] for s in segments], lows)
    checks.equal(
        name + ": upper interval endpoints", [s[1, 0] for s in segments], highs
    )
    checks.equal(
        name + ": interval model positions",
        [[s[0, 1], s[1, 1]] for s in segments],
        [[0, 0], [1, 1], [2, 2]],
    )
    # These intervals are supplied plotting fixtures, not bootstrap estimates.
    for text in ax.texts:
        if "target items" in text.get_text():
            text.set_text(
                "Synthetic plotting inputs · supplied interval endpoints "
                "(not computed CIs)"
            )


def plotting_examples(checks):
    colors = {"Synthetic": "#376b94"}
    frame = pd.DataFrame(
        {
            "model": ["excluded", "b", "top", "a"],
            "display_name": ["Exactly 90%", "Tied B", "Top", "Tied A"],
            "score": [0.95, 0.4, 0.8, 0.4],
            "ci_low": [0.9, 0.2, 0.6, 0.45],
            "ci_high": [1, 0.5, 0.95, 0.65],
            "n_items": [900, 901, 1000, 901],
            "scorable_items": [900, 901, 1000, 901],
            "expected_items": 1000,
            "provider": "Synthetic",
            "color": colors["Synthetic"],
            "n_bootstrap": 1000,
            "ci_level": 0.95,
        }
    )
    required = minimum_scorable_items(1000, 700, 0.9)
    checks.equal("Eligibility: >90% of 1,000", required, 901)
    checks.equal("Eligibility: >90% of 991", minimum_scorable_items(991, 700, 0.9), 892)
    checks.equal(
        "Eligibility: 700 item floor", minimum_scorable_items(500, 700, 0.9), 700
    )
    eligible = eligible_models(
        frame.sort_values(["score", "model"], ascending=[False, True]), required
    )
    checks.true(
        "Eligibility: excluded model absent",
        eligible.model.tolist() == ["top", "a", "b"],
    )
    original = eligible.copy(deep=True)
    fig = plot_leaderboard(eligible, colors, "synthetic")
    check_artists(
        checks,
        fig,
        "F2 plot",
        [0.8, 0.4, 0.4],
        [0.6, 0.45, 0.2],
        [0.95, 0.65, 0.5],
        ["Top", "Tied A", "Tied B"],
    )
    ax = fig.axes[0]
    checks.true("F2 plot: axis label", ax.get_xlabel() == "Pooled F2")
    checks.true("F2 plot: percent units", ax.xaxis.get_major_formatter()(0.8) == "80%")
    checks.true(
        "F2 plot: score annotations",
        [t.get_text() for t in ax.texts if t.get_position()[0] == 1.02]
        == ["80.0%", "40.0%", "40.0%", "F2"],
    )
    checks.true("F2 plot: input unchanged", eligible.equals(original))
    f2_html = figure_html(
        fig,
        "Production F2 renderer: ties, exclusion at exactly 90%, "
        "and intervals aligned with their models.",
    )
    adjusted = eligible.copy()
    adjusted["score"] = [50.0, 0.0, -40.0]
    adjusted["ci_low"] = [20.0, -20.0, -80.0]
    adjusted["ci_high"] = [80.0, 30.0, -10.0]
    before = adjusted.copy(deep=True)
    fig = plot_adjusted_leaderboard(
        adjusted,
        colors,
        "synthetic",
        required,
        "Synthetic models and supplied intervals.",
    )
    check_artists(
        checks,
        fig,
        "Adjusted BA plot",
        [0.5, 0, -0.4],
        [0.2, -0.2, -0.8],
        [0.8, 0.3, -0.1],
        ["Top", "Tied A", "Tied B"],
    )
    ax = fig.axes[0]
    checks.true("Adjusted BA plot: negative interval visible", ax.get_xlim()[0] <= -0.8)
    checks.true(
        "Adjusted BA plot: point units", ax.xaxis.get_major_formatter()(0.5) == "50"
    )
    checks.true(
        "Adjusted BA plot: axis label",
        ax.get_xlabel() == "Adjusted balanced accuracy (0 = chance; 100 = perfect)",
    )
    checks.true(
        "Adjusted BA plot: score annotations",
        [t.get_text() for t in ax.texts if t.get_position()[0] == 1.02]
        == ["50.0", "0.0", "-40.0", "Adj. BA"],
    )
    checks.true("Adjusted BA plot: input unchanged", adjusted.equals(before))
    adjusted_html = figure_html(
        fig,
        "Production adjusted BA renderer: negative scores and interval tails "
        "remain visible; annotations use points.",
    )
    return (
        "<h2>Plotting and eligibility</h2><p>The following are supplied plot inputs, "
        "independent of the scoring examples above. They test the same F2 and adjusted "
        "BA renderers used for real reports. The intervals are illustrative, "
        "not fitted "
        "to synthetic responses. Tied A deliberately has an interval that does not "
        "contain its F2 estimate: the renderer must draw the supplied endpoints.</p>"
        + table(
            ["Model", "Scorable / target", "Eligible", "F2", "Supplied interval"],
            [
                ("Exactly 90%", "900 / 1000", "No", 0.95, "[.90, 1.00]"),
                ("Top", "1000 / 1000", "Yes", 0.8, "[.60, .95]"),
                ("Tied A", "901 / 1000", "Yes", 0.4, "[.45, .65]"),
                ("Tied B", "901 / 1000", "Yes", 0.4, "[.20, .50]"),
            ],
        )
        + f2_html
        + adjusted_html
        + "<p>Checks inspect Matplotlib bar widths, model labels, interval endpoints "
        "and positions, units, and negative-axis limits. They do not compare image "
        "pixels or assess whether every label is visually readable; review the figures "
        "for presentation. The shared renderer is also used by F1/BA reports, but "
        "their full report drivers and hardware-tier figures "
        "are outside these examples.</p>"
    )


def build(output):
    checks = Checks()
    sections = (
        worked_cases(checks) + pooled_and_missing(checks) + plotting_examples(checks)
    )
    status = "PASS" if not checks.failures else "FAIL"
    results = table(
        ["Check", "Result", "Details"],
        [
            (name, "PASS" if passed else "FAIL", detail)
            for name, passed, detail in checks.results
        ],
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Phonics metric worked examples</title><style>
body{{max-width:1100px;margin:40px auto;padding:0 24px;
font:16px/1.6 system-ui,sans-serif;color:#243342;background:#fafbfc}}
h1,h2{{line-height:1.25;color:#162b40}}h2{{margin-top:2.5em}}
code{{background:#e9eef3;padding:2px 5px}}
table{{border-collapse:collapse;width:100%;font-size:14px;background:white}}
th,td{{padding:9px 12px;border:1px solid #d6dfe7;text-align:left;vertical-align:top}}
th{{background:#e9eef3}}.table-scroll{{overflow-x:auto;margin:20px 0}}
img{{max-width:100%;height:auto}}figure{{margin:24px 0}}figcaption{{color:#536171}}
.status{{padding:14px;border:2px solid {"#23734b" if status == "PASS" else "#b32d2d"};
background:white}}details{{margin:28px 0}}a{{color:#245d91}}
</style></head><body><h1>Phonics metric worked examples</h1>
<p>How F1, F2, balanced accuracy and adjusted balanced accuracy behave on small,
hand-checkable inputs. Expected answers are handwritten; computed answers use the
production scorer. Figures use the production leaderboard renderers.</p>
<p class="status"><strong>{status}</strong> ·
{len(checks.results) - len(checks.failures)} / {len(checks.results)} checks passed.</p>
{sections}<details><summary>All numerical and plotting checks</summary>
{results}</details>
<h2>Reproduce</h2><pre><code>uv run --frozen --package phonics-benchmark python \\
  benchmarks/phonics_benchmark/scripts/build_metric_examples.py \\
  --output outputs/metric-examples.html</code></pre>
<p>The command runs offline, writes this self-contained file, and exits nonzero if
any check fails. No external images, JavaScript, documentation framework or hosting
are required. Edit the examples and their explanations in the Python script;
expected values must be reviewed independently of computed outputs.</p></body></html>
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    print(f"{status}: {len(checks.results)} checks; {output}")
    for name, _, details in checks.failures:
        print(f"FAIL: {name}\n{details}")
    return not checks.failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/metric-examples.html")
    )
    args = parser.parse_args()
    raise SystemExit(0 if build(args.output) else 1)

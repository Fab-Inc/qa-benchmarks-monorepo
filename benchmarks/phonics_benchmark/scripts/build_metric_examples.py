"""Check worked metric examples and render a standalone HTML report, offline.

uv run --frozen --package phonics-benchmark python \
  benchmarks/phonics_benchmark/scripts/build_metric_examples.py

Expected answers below are handwritten. Production functions compute actual values.
The report is written even when a numeric check fails; the command then exits 1.
"""

from __future__ import annotations

# BEGIN EXAMPLE: imports
import argparse
import base64
import html
import io
import json
import math
import tempfile
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.container import ErrorbarContainer
from phonics_benchmark import balanced_accuracy as ba
from phonics_benchmark.bootstrap import (
    bootstrap_metric_cis,
    summarize_balanced_accuracy,
)
from phonics_benchmark.hardware import minimum_scorable_items
from phonics_benchmark.plots import (
    eligible_models,
    plot_adjusted_leaderboard,
    plot_leaderboard,
)
from phonics_benchmark.scoring import (
    add_per_item_confusion_matrix,
    summarize_by_model,
)

# END EXAMPLE: imports

# BEGIN EXAMPLE: cases
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
        "Finds cat and dog, but also flags sun and moon. "
        "Recall is 1 and specificity is 0, giving chance-level BA.",
    ),
    (
        "Flag nothing",
        "",
        (0, 0, 2, 2),
        (0, 0, 1 / 2, 0),
        "Misses both gold words. Recall is 0 and specificity is 1: "
        "BA is at chance, while F1 and F2 are zero.",
    ),
    (
        "Miss one gold word",
        "cat",
        (1, 0, 1, 2),
        (2 / 3, 5 / 9, 3 / 4, 50),
        "Finds cat but misses dog. Recall halves; specificity stays at 1. "
        "F2 falls further than F1 because it emphasizes recall.",
    ),
    (
        "Flag one extra text word",
        "cat dog sun",
        (2, 1, 0, 1),
        (4 / 5, 10 / 11, 3 / 4, 50),
        "Incorrectly flags sun. Recall stays at 1; specificity halves. "
        "F2 stays closer to 1 than F1 does.",
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
# Handwritten substitutions, independent of the scoring functions.
# Each entry is (in-text FP, out-of-text FP, F1, F2, BA, adjusted BA workings).
WORKINGS = {
    "Perfect": (
        0,
        0,
        "4 / (4 + 0 + 0) = 1",
        "10 / (10 + 0 + 0) = 1",
        "½ × (2/2 + 2/2) = 1",
        "100 × (2 × 1 − 1) = 100",
    ),
    "Miss one gold word": (
        0,
        0,
        "2 / (2 + 0 + 1) = 2/3",
        "5 / (5 + 0 + 4) = 5/9",
        "½ × (1/2 + 2/2) = 3/4",
        "100 × (2 × 3/4 − 1) = 50",
    ),
    "Flag one extra text word": (
        1,
        0,
        "4 / (4 + 1 + 0) = 4/5",
        "10 / (10 + 1 + 0) = 10/11",
        "½ × (2/2 + 1/2) = 3/4",
        "100 × (2 × 3/4 − 1) = 50",
    ),
    "Add an out-of-text word": (
        0,
        1,
        "4 / (4 + 1 + 0) = 4/5",
        "10 / (10 + 1 + 0) = 10/11",
        "½ × (2/2 + 2/2) = 1",
        "100 × (2 × 1 − 1) = 100",
    ),
    "Flag everything": (
        2,
        0,
        "4 / (4 + 2 + 0) = 2/3",
        "10 / (10 + 2 + 0) = 5/6",
        "½ × (2/2 + 0/2) = 1/2",
        "100 × (2 × 1/2 − 1) = 0",
    ),
    "Flag nothing": (
        0,
        0,
        "0 / (0 + 0 + 2) = 0",
        "0 / (0 + 0 + 8) = 0",
        "½ × (0/2 + 2/2) = 1/2",
        "100 × (2 × 1/2 − 1) = 0",
    ),
    "Invert every label": (
        2,
        0,
        "0 / (0 + 2 + 2) = 0",
        "0 / (0 + 2 + 8) = 0",
        "½ × (0/2 + 0/2) = 0",
        "100 × (2 × 0 − 1) = −100",
    ),
    "Repeat and capitalize": (
        0,
        0,
        "4 / (4 + 0 + 0) = 1",
        "10 / (10 + 0 + 0) = 1",
        "½ × (2/2 + 2/2) = 1",
        "100 × (2 × 1 − 1) = 100",
    ),
}
REVIEW_ORDER = [
    "Perfect",
    "Miss one gold word",
    "Flag one extra text word",
    "Add an out-of-text word",
    "Flag everything",
    "Flag nothing",
    "Invert every label",
    "Repeat and capitalize",
]
METRIC_LABELS = ["F1", "F2", "Balanced accuracy", "Adjusted BA (points)"]
# END EXAMPLE: cases


# BEGIN EXAMPLE: checks
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


# END EXAMPLE: checks


# BEGIN EXAMPLE: scoring_helpers
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


# END EXAMPLE: scoring_helpers


def code_block(name, title, *, collapsed=True):
    """Show an excerpt of the executed source, with its original line numbers."""
    source = Path(__file__).read_text(encoding="utf-8")
    marker = f"# BEGIN EXAMPLE: {name}\n"
    start = source.index(marker) + len(marker)
    end = source.index(f"# END EXAMPLE: {name}\n", start)
    excerpt = textwrap.dedent(source[start:end]).strip()
    first_line = source[:start].count("\n") + 1
    last_line = source[:end].count("\n")
    return (
        f'<details class="code-example"{"" if collapsed else " open"}>'
        f"<summary>{html.escape(title)}</summary>"
        f'<p class="source-location">build_metric_examples.py · '
        f"lines {first_line}–{last_line}</p>"
        f'<pre><code class="language-python">{html.escape(excerpt)}</code></pre>'
        "</details>"
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


def fraction(numerator, denominator):
    """Typeset trusted formula markup without an external math renderer."""
    return (
        f'<span class="fraction"><span>{numerator}</span>'
        f"<span>{denominator}</span></span>"
    )


def metric_equations(checks):
    # BEGIN EXAMPLE: count_table
    inputs = [item(content="cat dog sun moon", gold=["cat", "dog"])]
    scored, _, actual = score(inputs, {"four": ["cat", "sun", "fish"]})
    row = scored.iloc[0]
    groups = {
        "TP": row.tp,
        "FN": row.fn,
        "FP_in": row.fp & row.vocab_set,
        "FP_out": row.fp - row.vocab_set,
        "TN": row.tn,
    }
    meanings = {
        "TP": "Gold word correctly flagged",
        "FN": "Gold word missed",
        "FP_in": "Word in the text incorrectly flagged",
        "FP_out": "Predicted word absent from the text",
        "TN": "Other word in the text correctly left unflagged",
    }
    count_rows = [
        (name, meanings[name], ", ".join(sorted(words)) or "∅", len(words))
        for name, words in groups.items()
    ]
    count_table = table(
        ["Count", "Meaning", "Computed words", "Computed count"], count_rows
    )
    # END EXAMPLE: count_table
    checks.equal(
        "Count walkthrough: all five groups",
        [len(words) for words in groups.values()],
        [1, 1, 1, 1, 1],
    )
    checks.equal("Count walkthrough: four metrics", actual, [2 / 5, 5 / 11, 1 / 2, 0])
    # BEGIN EXAMPLE: equation_results
    # These expected answers and substitutions are handwritten.
    expected = [2 / 5, 5 / 11, 1 / 2, 0]
    substitutions = [
        "2 / (2 + 2 + 1) = 2/5",
        "5 / (5 + 2 + 4) = 5/11",
        "½ × (1/2 + 1/2) = 1/2",
        "100 × (2 × 1/2 − 1) = 0",
    ]
    equation_rows = list(zip(METRIC_LABELS, substitutions, expected, actual))
    equation_table = table(
        ["Metric", "Substitution", "Expected", "Computed"], equation_rows
    )
    # END EXAMPLE: equation_results
    equations = [
        (
            "F1",
            "F<sub>1</sub> = " + fraction("2TP", "2TP + FP + FN"),
            "Uses both kinds of false positives. "
            "True negatives do not enter the score.",
        ),
        (
            "F2",
            "F<sub>2</sub> = " + fraction("5TP", "5TP + FP + 4FN"),
            "Emphasizes recall: a missed gold word contributes four times the "
            "denominator penalty of an extra prediction, holding TP fixed.",
        ),
        (
            "Balanced accuracy",
            "BA = ½ × ("
            + fraction("TP", "TP + FN")
            + " + "
            + fraction("TN", "TN + FP<sub>in</sub>")
            + ")",
            "Averages recall and specificity, giving the two classes equal weight. "
            "Only words in the text enter these counts.",
        ),
        (
            "Adjusted balanced accuracy",
            "Adjusted BA = 100 × (2BA − 1)",
            "A change of scale: BA 0, ½ and 1 become −100, 0 and 100 points. "
            "Ranking is unchanged. Zero is the chance baseline; "
            "negative values remain.",
        ),
    ]
    return (
        '<section id="equations"><h2>1. Counts and equations</h2>'
        "<p>A positive prediction flags a word as nondecodable. We compare unique, "
        "lowercase words. All examples use already-parsed word lists.</p>"
        "<p>For a concrete count walkthrough, take text <code>cat dog sun moon</code>, "
        "gold <code>cat dog</code>, and prediction <code>cat sun fish</code>. "
        "The production scorer identifies the words in each group below.</p>"
        + code_block(
            "scoring_helpers",
            "Helpers used below: item construction and production scoring",
        )
        + code_block(
            "count_table",
            "Code: score the prediction and populate the count table",
            collapsed=False,
        )
        + count_table
        + '<p class="takeaway">F1 and F2 use <strong>FP = FP<sub>in</sub> + '
        "FP<sub>out</sub></strong>. Balanced accuracy uses only FP<sub>in</sub>. "
        "This distinction explains why an out-of-text prediction affects the metrics "
        "differently.</p>"
        + '<div class="equation-grid">'
        + "".join(
            f'<article class="equation-card"><h3>{title}</h3>'
            f'<div class="equation">{formula}</div><p>{explanation}</p></article>'
            for title, formula, explanation in equations
        )
        + "</div><p>In BA, the first fraction is recall and the second is specificity. "
        "F1, F2 and BA range from 0 to 1 when defined. A zero denominator remains "
        "undefined; it is not silently replaced by zero.</p>"
        "<p>Here TP = FN = FP<sub>in</sub> = FP<sub>out</sub> = TN = 1, so "
        "total FP = 2. The <code>actual</code> values below come from the "
        "<code>score(...)</code> call above.</p>"
        + code_block(
            "equation_results",
            "Code: populate the metric comparison table",
            collapsed=False,
        )
        + equation_table
        + "</section>"
    )


def case_card(name, prediction, counts, expected, actual, explanation):
    # BEGIN EXAMPLE: case_tables
    # Arguments contain this case's handwritten counts and production scores.
    fp_in, fp_out, *workings = WORKINGS[name]
    tp, _, fn, tn = counts
    count_table = table(
        ["Hand counts", "TP", "FP_in", "FP_out", "FN", "TN"],
        [("Expected", tp, fp_in, fp_out, fn, tn)],
    )
    metric_rows = list(zip(METRIC_LABELS, workings, expected, actual))
    metric_table = table(
        ["Metric", "Substitute the counts", "Expected", "Computed"], metric_rows
    )
    # END EXAMPLE: case_tables
    expanded = name in REVIEW_ORDER[:4]
    if name == "Repeat and capitalize":
        explanation += " The text is also repeated: cat cat DOG dog sun moon moon."
    return (
        f'<details class="worked-case"{" open" if expanded else ""}>'
        f"<summary>{html.escape(name)}</summary>"
        f"<p><strong>Prediction:</strong> <code>{html.escape(prediction or '∅')}</code>"
        f"<br>{html.escape(explanation)}</p>"
        + code_block(
            "case_tables",
            "Code: populate this case’s tables",
            collapsed=name != "Perfect",
        )
        + count_table
        + metric_table
        + "</details>"
    )


def worked_cases(checks):
    # BEGIN EXAMPLE: single_items
    results = {}
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
        vocab_counts = ba.vocabulary_counts(scored, dataset([item(content=content)]))
        checks.equal(
            f"{name}: in-text and out-of-text false positives",
            vocab_counts[["fp_in_vocabulary", "extra_predictions"]].iloc[0],
            WORKINGS[name][:2],
        )
        results[name] = (prediction, expected_counts, expected, actual, explanation)
    overview_rows = [
        (name, results[name][0] or "∅", *results[name][3]) for name in REVIEW_ORDER
    ]
    overview_table = table(
        ["Prediction change", "Prediction", *METRIC_LABELS], overview_rows
    )
    # END EXAMPLE: single_items
    # BEGIN EXAMPLE: undefined_metrics
    edge_results = []
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
        edge_results.extend(
            [(name, "Expected", *expected), (name, "Computed", *actual)]
        )
    edge_table = table(["Case", "Result", *METRIC_LABELS], edge_results)
    # END EXAMPLE: undefined_metrics
    return (
        '<section id="examples"><h2>2. Change one prediction at a time</h2>'
        '<div class="fixture"><p><strong>Text:</strong> <code>cat dog sun moon</code>'
        "<br><strong>Gold nondecodable words:</strong> <code>cat dog</code>"
        "<br><strong>Other words:</strong> <code>sun moon</code></p></div>"
        "<p>Start with the perfect prediction. Compare a missed gold word, an extra "
        "text word, and an out-of-text word. The overview shows computed scores; "
        "each worked example below compares them with independent "
        "handwritten answers.</p>"
        + code_block("cases", "Inputs: predictions and handwritten expected answers")
        + code_block(
            "single_items",
            "Code: score each prediction and populate the overview",
            collapsed=False,
        )
        + overview_table
        + '<p class="takeaway"><strong>What to compare:</strong> missing '
        "<code>dog</code> and adding <code>sun</code> both give BA = ¾, but F2 is "
        "5/9 versus 10/11. Adding <code>fish</code> instead of <code>sun</code> "
        "leaves the F scores unchanged and restores BA to 1. F scores penalize "
        "both extra predictions; BA evaluates only the text vocabulary.</p>"
        + "".join(case_card(name, *results[name]) for name in REVIEW_ORDER)
        + "<h3>When a metric is undefined</h3>"
        "<p>With no gold words and an empty prediction, TP = FP = FN = 0 and TN = 4. "
        "F1 and F2 are 0/0; BA also contains recall = 0/0. With every text word gold "
        "and correctly predicted, TP = 4 and FP = FN = TN = 0: F1 = 8/8 = 1 and "
        "F2 = 20/20 = 1, but BA contains specificity = 0/0. Adjusted BA is undefined "
        "whenever BA is undefined.</p>"
        + code_block(
            "undefined_metrics",
            "Code: score missing-class cases and populate the table",
            collapsed=False,
        )
        + edge_table
        + "</section>"
    )


def pooled_and_missing(checks):
    # BEGIN EXAMPLE: pooled_inputs
    items = [
        item("small", "cat dog", ["cat"]),
        item("large", "sun moon rain wind snow ice", ["sun", "moon", "rain", "wind"]),
    ]
    scored, summary, actual = score(items, {"small": ["cat"], "large": []})
    count_columns = ["tp_count", "fp_count", "fn_count", "tn_count"]
    item_rows = [
        (
            row.benchmark_item_id,
            row.content,
            " ".join(row.nondecodable_words),
            " ".join(row.response_parsed) or "∅",
            " / ".join(str(getattr(row, column)) for column in count_columns),
            row.f1,
        )
        for row in scored.itertuples()
    ]
    totals = " / ".join(str(value) for value in scored[count_columns].sum())
    item_rows.append(("Pooled counts", "", "", "", totals, ""))
    item_table = table(
        ["Item", "Text", "Gold", "Prediction", "TP / FP / FN / TN", "F1"], item_rows
    )
    # END EXAMPLE: pooled_inputs
    checks.equal("Unequal items: pooled metrics", actual, [1 / 3, 5 / 21, 3 / 5, 20])
    checks.equal("Unequal items: item-averaged F1", summary.macro_f1, 1 / 2)
    checks.equal(
        "Unequal items: per-item counts",
        scored[count_columns],
        [[1, 0, 0, 1], [0, 0, 4, 2]],
    )
    # BEGIN EXAMPLE: pooled_examples
    pooled_rows = [
        ("Pooled F1: 2/(2+4)", 1 / 3, actual[0]),
        ("Item-averaged F1: (1+0)/2", 1 / 2, summary.macro_f1),
        ("Pooled F2: 5/(5+16)", 5 / 21, actual[1]),
        ("BA: (1/5+1)/2", 3 / 5, actual[2]),
        ("Adjusted BA: 100 × (2 × 3/5 − 1)", 20, actual[3]),
    ]
    pooled_table = table(
        ["Metric and substitution", "Expected", "Computed"], pooled_rows
    )
    # END EXAMPLE: pooled_examples
    # BEGIN EXAMPLE: missing_examples
    _, missing_summary, missing_scores = score(items, {"small": ["cat"]})
    coverage = missing_summary.n_items / len(items)
    missing_rows = [
        (
            "Empty (scored)",
            *actual,
            f"{summary.n_items}/{len(items)} = {summary.n_items / len(items):.0%}",
        ),
        (
            "Missing",
            *missing_scores,
            f"{missing_summary.n_items}/{len(items)} = {coverage:.0%}",
        ),
    ]
    missing_table = table(
        ["Response on large item", *METRIC_LABELS, "Coverage"], missing_rows
    )
    # END EXAMPLE: missing_examples
    checks.equal("Missing item: scores", missing_scores, [1, 1, 1, 100])
    checks.equal(
        "Missing item: full denominator",
        [missing_summary.n_items, len(items), coverage],
        [1, 2, 1 / 2],
    )
    return (
        '<section id="pooling"><h2>3. Pooling and missing responses</h2>'
        "<p>With multiple items, the report adds their confusion counts before "
        "applying the metric equation. This differs from averaging item scores.</p>"
        + code_block(
            "pooled_inputs",
            "Code: score both items and populate their counts",
            collapsed=False,
        )
        + item_table
        + '<div class="equation">Pooled F<sub>1</sub> = '
        + fraction("2ΣTP", "2ΣTP + ΣFP + ΣFN")
        + " = "
        + fraction("2 × 1", "2 × 1 + 0 + 4")
        + " = ⅓</div>"
        + '<div class="equation">Item-averaged F<sub>1</sub> = '
        + fraction("1 + 0", "2")
        + " = ½</div>"
        + "<p>Both items have gold words, so both enter this item average. "
        "In the production scorer, macro F1 averages over items with at least "
        "one gold word. The large item's four missed words weigh more in the pooled "
        "score; an item average gives these two items equal weight.</p>"
        + code_block(
            "pooled_examples",
            "Code: compare pooled scores with expected answers",
            collapsed=False,
        )
        + pooled_table
        + "<h3>An empty prediction and a missing response are different</h3>"
        "<p>The empty prediction above is a scored response: it misses four gold "
        "words. If that response is absent instead, the large item contributes "
        "no confusion counts. Only the perfect small item remains.</p>"
        + code_block(
            "missing_examples",
            "Code: omit the large response and populate the comparison",
            collapsed=False,
        )
        + missing_table
        + '<p class="takeaway">A perfect score on the observed subset can coexist '
        "with only 50% coverage. Always read the score together with coverage.</p>"
        + '<div class="equation">Coverage = '
        + fraction("Number of scorable responses", "Number of target items")
        + "</div>"
        + "</section>"
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


# BEGIN EXAMPLE: plot_checks
def check_artists(
    checks,
    fig,
    name,
    scores,
    lows,
    highs,
    labels,
    *,
    note="Synthetic plotting inputs · supplied interval endpoints (not computed CIs)",
):
    ax = fig.axes[0]
    checks.equal(name + ": bar values", [bar.get_width() for bar in ax.patches], scores)
    checks.equal(
        name + ": bar positions",
        [bar.get_y() + bar.get_height() / 2 for bar in ax.patches],
        list(range(len(scores))),
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
        [[i, i] for i in range(len(scores))],
    )
    # Compare rendered coordinates too: an artist transform could offset a line
    # even when its data coordinates match the bar's row.
    interval_transform = errors[0].lines[2][0].get_transform()
    checks.equal(
        name + ": intervals align with bars on screen",
        [interval_transform.transform(segment)[:, 1] for segment in segments],
        [[bar.get_transform().transform((0.5, 0.5))[1]] * 2 for bar in ax.patches],
    )
    for text in ax.texts:
        if "target items" in text.get_text():
            text.set_text(note)


# END EXAMPLE: plot_checks


def bootstrap_plot_examples(checks, *, render_figures=False):
    # BEGIN EXAMPLE: bootstrap_intervals
    items = [
        item("small", "cat dog", ["cat"]),
        item("large", "sun moon rain wind snow ice", ["sun", "moon", "rain", "wind"]),
    ]
    scored, summary, _ = score(items, {"small": ["cat"], "large": []})
    bootstrap = bootstrap_metric_cis(
        scored, n_bootstrap=1000, seed=0, item_ids=dataset(items).index
    ).iloc[0]
    balanced, _ = summarize_balanced_accuracy(
        scored, dataset(items), n_bootstrap=1000, seed=0
    )
    balanced = balanced.iloc[0]
    # END EXAMPLE: bootstrap_intervals
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
    checks.equal(
        "Bootstrap: BA and adjusted bounds",
        balanced[
            [
                "balanced_accuracy_ci_low",
                "balanced_accuracy_ci_high",
                "adjusted_balanced_accuracy_ci_low",
                "adjusted_balanced_accuracy_ci_high",
            ]
        ],
        [1 / 2, 1, 0, 100],
    )
    missing, _, _ = score(items, {"small": ["cat"]})
    missing_report, _ = summarize_balanced_accuracy(
        missing, dataset(items), n_bootstrap=1000, seed=0
    )
    checks.equal(
        "Missing item: BA report full denominator",
        missing_report.iloc[0][["n_scorable", "expected_items", "coverage"]],
        [1, 2, 1 / 2],
    )
    # BEGIN EXAMPLE: bootstrap_plots
    colors = {"Synthetic": "#376b94"}
    frame = pd.DataFrame(
        [
            {
                "model": "example",
                "display_name": "Two-item example",
                "score": bootstrap.pooled_f2,
                "ci_low": bootstrap.pooled_f2_ci_low,
                "ci_high": bootstrap.pooled_f2_ci_high,
                "n_items": summary.n_items,
                "scorable_items": summary.n_items,
                "expected_items": len(items),
                "provider": "Synthetic",
                "color": colors["Synthetic"],
                "n_bootstrap": 1000,
                "ci_level": 0.95,
            }
        ]
    )
    f2_fig = plot_leaderboard(frame, colors, "synthetic")
    adjusted = frame.assign(
        score=balanced.adjusted_balanced_accuracy,
        ci_low=balanced.adjusted_balanced_accuracy_ci_low,
        ci_high=balanced.adjusted_balanced_accuracy_ci_high,
    )
    adjusted_fig = plot_adjusted_leaderboard(
        adjusted,
        colors,
        "synthetic",
        0,
        "Two synthetic items; no leaderboard coverage filter applied.",
    )
    f2_fig.axes[0].set_title(
        "Two-item bootstrap example · Pooled F2", loc="left", pad=35
    )
    adjusted_fig.axes[0].set_title(
        "Two-item bootstrap example · Adjusted balanced accuracy", loc="left", pad=35
    )
    # END EXAMPLE: bootstrap_plots
    note = "Two synthetic items · 1,000 item resamples · 95% percentile intervals"
    check_artists(
        checks,
        f2_fig,
        "Bootstrap F2 plot",
        [5 / 21],
        [0],
        [1],
        ["Two-item example"],
        note=note,
    )
    check_artists(
        checks,
        adjusted_fig,
        "Bootstrap adjusted BA plot",
        [0.2],
        [0],
        [1],
        ["Two-item example"],
        note=note,
    )
    figures = ""
    for fig, description in [
        (
            f2_fig,
            "F2 = 5/21 (23.8%); the item bootstrap gives a 95% interval of [0%, 100%].",
        ),
        (
            adjusted_fig,
            "Adjusted BA = 20 points; the item bootstrap gives a 95% "
            "interval of [0, 100] points.",
        ),
    ]:
        if render_figures:
            figures += figure_html(fig, description)
        else:
            plt.close(fig)
    # BEGIN EXAMPLE: bootstrap_draw_table
    # Handwritten possible draws and expected scores, independent of the bootstrap.
    draw_rows = [
        ("Small + small", "¼", "2 / 0 / 0 / 2", "10/10 = 1", "1", "100"),
        ("Small + large", "¼", "1 / 0 / 4 / 3", "5/21", "3/5", "20"),
        ("Large + small", "¼", "1 / 0 / 4 / 3", "5/21", "3/5", "20"),
        ("Large + large", "¼", "0 / 0 / 8 / 4", "0/32 = 0", "1/2", "0"),
    ]
    draw_table = table(
        ["Draw", "Probability", "TP / FP / FN / TN", "F2", "BA", "Adjusted BA"],
        draw_rows,
    )
    # END EXAMPLE: bootstrap_draw_table
    # BEGIN EXAMPLE: interval_table
    interval_rows = [
        (
            "F2",
            5 / 21,
            bootstrap.pooled_f2,
            "[0, 1]",
            f"[{display(bootstrap.pooled_f2_ci_low)}, "
            f"{display(bootstrap.pooled_f2_ci_high)}]",
        ),
        (
            "Adjusted BA",
            20,
            balanced.adjusted_balanced_accuracy,
            "[0, 100]",
            f"[{display(balanced.adjusted_balanced_accuracy_ci_low)}, "
            f"{display(balanced.adjusted_balanced_accuracy_ci_high)}]",
        ),
    ]
    interval_table = table(
        [
            "Plot",
            "Expected point",
            "Computed point",
            "Expected interval",
            "Computed interval",
        ],
        interval_rows,
    )
    # END EXAMPLE: interval_table
    return (
        '<section id="plots"><h2>4. Plots and bootstrap intervals</h2>'
        "<p>The bar is the pooled score from the original items. Its interval comes "
        "from resampling whole items with replacement and recomputing that score. "
        "Resampling does not change the metric definition "
        "or the original point estimate. Intervals show their lower and upper "
        "bounds; they need not be symmetric around the end of the bar.</p>"
        "<p>Reuse the small and large items above. Each resample draws two items. "
        "These are all possible draws, with their handwritten scores:</p>"
        + code_block(
            "bootstrap_draw_table",
            "Code: define the handwritten resampling table",
            collapsed=False,
        )
        + draw_table
        + '<div class="equation">95% percentile interval = '
        "[Q<sub>2.5%</sub>(resampled scores), "
        "Q<sub>97.5%</sub>(resampled scores)]</div>"
        "<p>The production routines draw 1,000 resamples with seed 0. The expected "
        "endpoints are 0 and 1 for F2, and 0 and 100 points for adjusted BA. "
        "The plots below use the computed endpoints directly.</p>"
        + code_block(
            "bootstrap_intervals",
            "Code: compute the bootstrap intervals",
            collapsed=False,
        )
        + code_block(
            "interval_table", "Code: populate the interval comparison", collapsed=False
        )
        + interval_table
        + code_block(
            "bootstrap_plots",
            "Code: plot the computed scores and intervals",
            collapsed=False,
        )
        + figures
        + '<p class="takeaway">Two very different items produce very wide intervals. '
        "This is an illustration of the resampling mechanics, not a realistic "
        "benchmark uncertainty estimate. The two-item illustration bypasses the "
        "real leaderboard coverage threshold.</p>" + "</section>"
    )


def plotting_examples(checks, *, render_figures=False):
    # BEGIN EXAMPLE: plotting_inputs
    colors = {"Synthetic": "#376b94"}
    frame = pd.DataFrame(
        {
            "model": ["excluded", "b", "top", "a"],
            "display_name": ["Exactly 90%", "Tied B", "Top", "Tied A"],
            "score": [0.95, 0.4, 0.8, 0.4],
            "ci_low": [0.9, 0.2, 0.6, 0.25],
            "ci_high": [1, 0.5, 0.95, 0.55],
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
    eligible_ids = set(eligible.model)
    eligibility_rows = [
        (
            row.display_name,
            f"{row.scorable_items} / {row.expected_items}",
            "Yes" if row.model in eligible_ids else "No",
            row.score,
            f"[{row.ci_low:g}, {row.ci_high:g}]",
        )
        for row in frame.sort_values(
            ["score", "model"], ascending=[False, True]
        ).itertuples()
    ]
    eligibility_table = table(
        ["Model", "Scorable / target", "Eligible", "F2", "Supplied interval"],
        eligibility_rows,
    )
    # END EXAMPLE: plotting_inputs
    # BEGIN EXAMPLE: f2_plot
    original = eligible.copy(deep=True)
    fig = plot_leaderboard(eligible, colors, "synthetic")
    check_artists(
        checks,
        fig,
        "F2 plot",
        [0.8, 0.4, 0.4],
        [0.6, 0.25, 0.2],
        [0.95, 0.55, 0.5],
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
    # END EXAMPLE: f2_plot
    f2_html = ""
    if render_figures:
        f2_html = figure_html(
            fig,
            "Production F2 renderer: ties, exclusion at exactly 90%, "
            "and intervals aligned with their models.",
        )
    else:
        plt.close(fig)
    # BEGIN EXAMPLE: adjusted_plot
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
    # END EXAMPLE: adjusted_plot
    adjusted_html = ""
    if render_figures:
        adjusted_html = figure_html(
            fig,
            "Production adjusted BA renderer: negative scores and interval tails "
            "remain visible; annotations use points.",
        )
    else:
        plt.close(fig)
    return (
        '<section id="plot-checks"><h3>Additional plot and eligibility checks</h3>'
        "<p>The following are supplied plot inputs, "
        "independent of the scoring examples above. They test the same F2 and adjusted "
        "BA renderers used for real reports. The intervals are illustrative, "
        "not fitted to synthetic responses. Each interval here contains its score. "
        "For example, Tied A has F2 = 40% and a supplied interval of [25%, 55%]. "
        "A separate regression test checks that the renderer also preserves bounds "
        "when an interval does not contain its score.</p>"
        + code_block(
            "plotting_inputs", "Code: populate the eligibility table", collapsed=False
        )
        + eligibility_table
        + "<h3>F2 leaderboard</h3>"
        + code_block("f2_plot", "Code: F2 plot and checks", collapsed=False)
        + f2_html
        + "<h3>Adjusted balanced accuracy leaderboard</h3>"
        + code_block(
            "adjusted_plot", "Code: adjusted BA plot and checks", collapsed=False
        )
        + adjusted_html
        + code_block("plot_checks", "Shared plot inspection helper", collapsed=True)
        + "<p>Checks inspect Matplotlib bar widths, model labels, interval endpoints "
        "and positions, alignment with bars in rendered coordinates, units, and "
        "negative-axis limits. They do not compare image "
        "pixels or assess whether every label is visually readable; review the figures "
        "for presentation. The shared renderer is also used by F1/BA reports, but "
        "their full report drivers and hardware-tier figures "
        "are outside these examples.</p></section>"
    )


def build(output):
    checks = Checks()
    setup = (
        '<section id="source"><h2>5. Shared setup and source</h2>'
        "<p>Each section includes the code used to produce its results. "
        "Code excerpts are read from this script when the report is generated. "
        "They run inside the example functions with a shared "
        "<code>checks = Checks()</code> collector. Expand these definitions "
        "to inspect the imports, scoring helpers and checks they use.</p>"
        + code_block("imports", "Imports", collapsed=True)
        + code_block(
            "scoring_helpers",
            "Item construction and production scoring calls",
            collapsed=True,
        )
        + code_block("checks", "Numerical check collector", collapsed=True)
        + "</section>"
    )
    sections = (
        metric_equations(checks)
        + worked_cases(checks)
        + pooled_and_missing(checks)
        + bootstrap_plot_examples(checks, render_figures=True)
        + plotting_examples(checks, render_figures=True)
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
h3{{color:#162b40}}section{{scroll-margin-top:24px}}
nav{{display:flex;gap:18px;flex-wrap:wrap;margin:24px 0}}
.equation-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}}
.equation-card,.fixture,.worked-case{{padding:16px 20px;border:1px solid #d6dfe7;
border-radius:6px;background:white}}
.equation-card h3{{margin-top:0}}
.equation{{font-family:Georgia,serif;font-size:20px;line-height:1.8;
padding:12px 0;overflow-x:auto;white-space:nowrap}}
.fraction{{display:inline-flex;flex-direction:column;text-align:center;
vertical-align:middle;line-height:1.3;margin:0 4px}}
.fraction>span:first-child{{border-bottom:1px solid currentColor;padding:0 5px 3px}}
.fraction>span:last-child{{padding:3px 5px 0}}
.takeaway{{padding:14px 18px;border-left:4px solid #376b94;background:#edf2f6}}
.worked-case>summary{{font-weight:600;cursor:pointer;color:#162b40;font-size:18px}}
.fixture p{{margin:0}}
@media(max-width:720px){{.equation-grid{{grid-template-columns:1fr}}
body{{padding:0 14px}}.worked-case{{padding:12px}}}}
code{{background:#e9eef3;padding:2px 5px}}
pre{{overflow-x:auto;padding:16px;background:#edf2f6;border:1px solid #d6dfe7;
border-radius:6px;font-size:13px;line-height:1.55;tab-size:4}}
pre code{{background:none;padding:0}}
.code-example summary{{cursor:pointer;font-weight:600;color:#245d91}}
.source-location{{font-size:12px;color:#536171;margin-bottom:0}}
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
<nav aria-label="Report sections"><a href="#equations">Counts &amp; equations</a>
<a href="#examples">Prediction changes</a><a href="#pooling">Pooling &amp; coverage</a>
<a href="#plots">Plots &amp; intervals</a><a href="#source">Shared code</a></nav>
<p class="status"><strong>{status}</strong> ·
{len(checks.results) - len(checks.failures)} / {len(checks.results)} checks passed.</p>
{sections}{setup}<details><summary>All numerical and plotting checks</summary>
{results}</details>
<h2>Reproduce</h2><pre><code>uv run --frozen --package phonics-benchmark python \\
  benchmarks/phonics_benchmark/scripts/build_metric_examples.py \\
  --output benchmarks/phonics_benchmark/outputs/metric-examples.html</code></pre>
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
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "outputs"
        / "metric-examples.html",
    )
    args = parser.parse_args()
    raise SystemExit(0 if build(args.output) else 1)

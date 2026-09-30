#!/usr/bin/env python3
"""
Quick EDA of a final benchmark dataset JSON (item schema, e.g.
data/linguistic_knowledge_eval_kiswahili_clean.json).

Prints headline distributions and writes a 6-panel figure + a markdown report:
    eda_clean_<label>.png     domain, grade, domain x grade, answer-type/letters,
                              SVR/GEEAP, source-tier panels
    eda_clean_<label>.md      the same numbers as tables

Run:
    python benchmarks/linguistic_knowledge_benchmark/scripts/workflow/eda_clean_dataset.py
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

DOMAINS = [f"D{i}" for i in range(1, 10)]
GRADES = ["G1", "G2", "G3", "G4", "G5", "beyond_primary"]
GLABEL = {
    "G1": "G1",
    "G2": "G2",
    "G3": "G3",
    "G4": "G4",
    "G5": "G5",
    "beyond_primary": "BP",
}


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--json",
        default=os.path.join(
            root, "data", "linguistic_knowledge_eval_kiswahili_clean.json"
        ),
    )
    ap.add_argument("--outdir", default=os.path.join(root, "results", "analysis"))
    ap.add_argument("--label", default="kiswahili_clean")
    args = ap.parse_args()

    d = json.load(open(args.json, encoding="utf-8"))
    n = len(d)
    os.makedirs(args.outdir, exist_ok=True)

    dom = Counter(x["domain"] for x in d)
    grade = Counter(x["grade"] for x in d)
    atype = Counter(x["answer_type"] for x in d)
    mcq = [x for x in d if x["answer_type"] == "multiple_choice"]
    sf = [x for x in d if x["answer_type"] == "short_form"]
    letters = Counter(x["expected_answers"][0] for x in mcq)
    svr = Counter(x["svr_strand"] for x in d)
    geeap = Counter(x["geeap_component"] for x in d)
    gpe = Counter(x["grade_placement_evidence"] for x in d)
    tiers = Counter(s.get("tier") for x in d for s in x["sources"])
    grid = np.zeros((len(DOMAINS), len(GRADES)), dtype=int)
    for x in d:
        if x["domain"] in DOMAINS and x["grade"] in GRADES:
            grid[DOMAINS.index(x["domain"]), GRADES.index(x["grade"])] += 1

    # ---- figure ----
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    c1 = "#4e79a7"

    ax = axes[0, 0]
    ax.bar([d_ for d_ in DOMAINS], [dom.get(d_, 0) for d_ in DOMAINS], color=c1)
    ax.set_title("Items per domain", fontweight="bold")
    for i, d_ in enumerate(DOMAINS):
        ax.text(i, dom.get(d_, 0) + 0.3, dom.get(d_, 0), ha="center", fontsize=8)

    ax = axes[0, 1]
    ax.bar([GLABEL[g] for g in GRADES], [grade.get(g, 0) for g in GRADES], color=c1)
    ax.set_title("Items per grade", fontweight="bold")
    for i, g in enumerate(GRADES):
        ax.text(i, grade.get(g, 0) + 0.3, grade.get(g, 0), ha="center", fontsize=8)

    ax = axes[0, 2]
    ax.imshow(grid, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(GRADES)), [GLABEL[g] for g in GRADES])
    ax.set_yticks(range(len(DOMAINS)), DOMAINS)
    ax.set_title("Domain x Grade", fontweight="bold")
    vmax = grid.max()
    for i in range(len(DOMAINS)):
        for j in range(len(GRADES)):
            if grid[i, j]:
                ax.text(
                    j,
                    i,
                    grid[i, j],
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if grid[i, j] > vmax * 0.55 else "#1f2d3d",
                )

    ax = axes[1, 0]
    bars = ax.bar(
        ["A", "B", "C", "D"], [letters.get(x, 0) for x in "ABCD"], color="#59a14f"
    )
    exp = len(mcq) / 4
    ax.axhline(exp, color="#b30000", ls="--", lw=1)
    ax.set_title(f"MCQ correct-letter balance (n={len(mcq)})", fontweight="bold")
    for b, x in zip(bars, "ABCD"):
        ax.text(
            b.get_x() + b.get_width() / 2,
            letters.get(x, 0) + 0.2,
            letters.get(x, 0),
            ha="center",
            fontsize=9,
        )

    ax = axes[1, 1]
    items = list(geeap.items())
    items.sort(key=lambda kv: -kv[1])
    ax.barh([k for k, _ in items][::-1], [v for _, v in items][::-1], color="#f28e2b")
    ax.set_title("GEEAP component", fontweight="bold")

    ax = axes[1, 2]
    titems = list(tiers.items())
    order = {"curriculum": 0, "teacher_material": 1, "student_material": 2, "other": 3}
    titems.sort(key=lambda kv: order.get(kv[0], 9))
    ax.bar([k for k, _ in titems], [v for _, v in titems], color="#af7aa1")
    ax.set_title("Source tier (all source rows)", fontweight="bold")
    ax.tick_params(axis="x", rotation=20)

    fig.suptitle(
        f"EDA — {os.path.basename(args.json)}  (n={n}: "
        f"{atype.get('multiple_choice', 0)} MCQ, {atype.get('short_form', 0)} short-form)",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    figpath = os.path.join(args.outdir, f"eda_clean_{args.label}.png")
    fig.savefig(figpath, dpi=150)
    plt.close(fig)

    # ---- report ----
    nacc = [len(x["expected_answers"]) for x in sf]
    nsrc = [len(x["sources"]) for x in d]
    lines = [
        f"# EDA — {os.path.basename(args.json)}\n",
        f"- **{n} items**: {atype.get('multiple_choice', 0)} MCQ, "
        f"{atype.get('short_form', 0)} short-form",
        f"- Domains covered: {sum(1 for x in DOMAINS if dom.get(x))}/9  ·  "
        f"Grades covered: {sum(1 for g in GRADES if grade.get(g))}/6",
        f"- Unique subdomains: {len(set(x['subdomain'] for x in d))}",
        "- MCQ letter balance: "
        + ", ".join(f"{k}={letters.get(k, 0)}" for k in "ABCD"),
        f"- Short-form accepted answers/item: mean {np.mean(nacc):.2f} "
        f"(1={nacc.count(1)}, 2={nacc.count(2)}, 3={nacc.count(3)})",
        f"- Multi-source items: {sum(1 for v in nsrc if v > 1)}/{n}\n",
    ]

    def table(title, counter, keys=None):
        keys = keys or [k for k, _ in counter.most_common()]
        rows = [f"## {title}\n", "| value | count | share |", "|---|---|---|"]
        for k in keys:
            v = counter.get(k, 0)
            rows.append(f"| {k} | {v} | {v / n * 100:.0f}% |")
        rows.append("")
        return rows

    lines += table("Domain", dom, DOMAINS)
    lines += table("Grade", grade, GRADES)
    lines += table("SVR strand", svr)
    lines += table("GEEAP component", geeap)
    lines += table("Grade-placement evidence", gpe)
    lines += table(
        "Source tier (all rows)",
        tiers,
        ["curriculum", "teacher_material", "student_material", "other"],
    )
    lines += ["## Plot\n", f"- `eda_clean_{args.label}.png`"]

    mdpath = os.path.join(args.outdir, f"eda_clean_{args.label}.md")
    with open(mdpath, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"n={n}  MCQ={len(mcq)}  short_form={len(sf)}")
    print("domain:", dict(sorted(dom.items())))
    print("grade:", dict(sorted(grade.items())))
    print("letters:", dict(sorted(letters.items())))
    print("wrote", figpath)
    print("wrote", mdpath)


if __name__ == "__main__":
    main()

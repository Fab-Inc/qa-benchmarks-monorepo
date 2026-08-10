#!/usr/bin/env python3
"""
EDA of the v9 Linguistic Knowledge Benchmark authoring pipeline (chat stages + 3b script).

The pipeline:
    1a  digest      -> resource_digest.by_document[*].by_domain[*].by_grade[*]
                       (raw supply: what each SOURCE offers, per domain x grade;
                        each testable fact now carries a verbatim example)
    1b  merge       -> feature_map.by_domain[*].features[*]
                       (consolidated map: one feature list per domain, grade-ordered)
    2   generate    -> [ candidate, ... ]  (sharded, oversampled; shard_id + candidate_id)
    3a  pool & QC   -> [ candidate + qc_verdict / dup_group_id / quality_rank, ... ]
    3b  assemble    -> stage_3b_dataset.json (final ~150 items) + backups + gap_report
                       (produced by stage_3b_assemble_v10.py)

This script produces ONE multi-panel SVG figure per stage, plus a cross-stage funnel.
Each figure answers two questions:
    (a) is the stage working / internally consistent?
    (b) where are the gaps (domains, grades, sources, QC rules) and any surprises?

Stages whose input JSON is absent are skipped (the v9 candidate/pool/3b artifacts do
not exist until the pipeline has been run); the final-dataset figure also works against
a plain v8 dataset with no gap_report.

Run:
    python benchmarks/linguistic_knowledge_benchmark/scripts/eda_pipeline.py --language english
    python benchmarks/linguistic_knowledge_benchmark/scripts/eda_pipeline.py --language kiswahili
    python benchmarks/linguistic_knowledge_benchmark/scripts/eda_pipeline.py --language english \
        --digest ... --merge ... --candidates ... --pool ... \
        --dataset ... --gap-report ... --backups ... --outdir ...

Output: <outdir>/eda_1a_digest.svg, eda_1b_merge.svg, eda_2_candidates.svg,
        eda_3a_qc.svg, eda_3b_final.svg, and (when possible) eda_funnel.svg
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# --------------------------------------------------------------------------- #
# Canonical axes — fixed so that *absent* domains/grades still show as gaps.
# --------------------------------------------------------------------------- #
DOMAINS = [f"D{i}" for i in range(1, 10)]
DOMAIN_LABEL = {
    "D1": "D1 Phonics/Ortho",
    "D2": "D2 Phonology",
    "D3": "D3 Nominal morph",
    "D4": "D4 Verbal morph",
    "D5": "D5 Syntax",
    "D6": "D6 Semantics/Lex",
    "D7": "D7 Pragmatics/Disc",
    "D8": "D8 Socioling/Reg",
    "D9": "D9 Metalinguistic",
}
GRADES = ["G0", "G1", "G2", "G3", "G4", "G5", "beyond_primary"]
GRADE_LABEL = {g: ("BP" if g == "beyond_primary" else g) for g in GRADES}

# coverage categories (digest) ordered worst -> best
COVERAGE_ORDER = ["absent", "thin", "some", "rich"]
COVERAGE_COLOR = {
    "absent": "#d9d9d9",
    "thin": "#fdbe85",
    "some": "#fd8d3c",
    "rich": "#d94701",
}

# GEEAP components ordered as the tagging prompt explains them
GEEAP_ORDER = [
    "systematic_phonics",
    "phonological_awareness",
    "writing",
    "oral_language",
    "reading_comprehension",
    "reading_fluency",
]

# tier hierarchy ordered best -> worst (source quality)
TIER_ORDER = ["curriculum", "teacher_material", "student_material", "other"]
TIER_COLOR = {
    "curriculum": "#1a9850",
    "teacher_material": "#91cf60",
    "student_material": "#fee08b",
    "other": "#fc8d59",
}

# qualitative palette for "by source document"
SRC_PALETTE = [
    "#4e79a7",
    "#f28e2b",
    "#59a14f",
    "#e15759",
    "#76b7b2",
    "#edc948",
    "#b07aa1",
    "#ff9da7",
    "#9c755f",
    "#bab0ac",
]

HEAT = LinearSegmentedColormap.from_list(
    "supply", ["#ffffff", "#fee0b6", "#fdae61", "#d7301f"]
)

# Stage 3a anti-gaming / shape QC rules (numbered as in process_prompts_chat_ready_v9.md).
# Rules 3-7 apply to multiple_choice items only; 1-2 and 8-12 apply to every item.
QC_RULES = {
    1: "no-answer-in-stem",
    2: "rule-not-stated",
    3: "length-parity",
    4: "uniform-glosses",
    5: "parallel-options",
    6: "distractors-function",
    7: "shortcut-check",
    8: "construct-check",
    9: "well-formed",
    10: "english-medium",
    11: "text-only-impersonal",
    12: "translation-answer-neutral",
}
QC_RULE_ORDER = list(range(1, 13))
QC_MCQ_ONLY = {3, 4, 5, 6, 7}  # rules that apply to multiple_choice items only

# answer_type colours, shared across the Stage-2 and Stage-3b figures
ATYPE_COLOR = {"multiple_choice": "#4e79a7", "short_form": "#f28e2b"}

# --------------------------------------------------------------------------- #
# STAGE 1a SOURCE SELECTION
# Every source document currently present in the digest JSON is listed below.
# Comment out (prepend #) a row to EXCLUDE that document from the STAGE 1a EDA
# only — the 1b merge and Stage 2 figures always use every source.
# When you add new documents to the digest JSON, add their `source_name` here.
# Names must match the digest's `source_name` exactly (the file name).
# --------------------------------------------------------------------------- #
DIGEST_SOURCES = [
    # comment out any source to exclude it from the Stage 1a EDA;
    # leave the list empty to include all sources found in the digest JSON.
    # "Swahili_Grammar_for_Introductory_and_Intermediate.pdf",
    # "Swahili_Learners__Reference_Grammar",
    # "Kiswahili_1_-_Mwongozo_wa_Mwalimu",
    # "Kiswahili_2_-_Mwongozo_wa_Mwalimu",
    # "Kiswahili_3_-_Mazoezi_ya_Lugha_-_Mwongozo_wa_Mwalimu_-_Darasa_la_3",
    # "Kiswahili_1_-_Kitabu_cha_Mwanafunzi",
    # "Kiswahili_2_-_Kitabu_cha_Mwanafunzi",
    # "Kiswahili_3_-_Mazoezi_ya_Lugha_-_Kitabu_cha_Mwanafunzi_-_Darasa_la_3",
    # "tie_early_childhood_education_curriculum.pdf",
    # "tie_primary_education_curriculum.pdf",
]

# --------------------------------------------------------------------------- #
# DEFAULT INPUT / OUTPUT PATHS
# Derived from the mandatory --language flag (each language has its own
# data/<language>_workflow/ tree with the same stage file layout). Override
# any individual path on the CLI with the matching --flag. `candidates`,
# `pool` and `backups` are lists so several per-shard files can be listed.
# Stage-2/3a/3b files may not exist yet — a stage with a missing input is
# simply skipped.
# --------------------------------------------------------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)  # benchmarks/linguistic_knowledge_benchmark

LANGUAGE_WORKFLOW_DIR = {
    "english": "english_workflow",
    "kiswahili": "kiswahili_workflow",
}


def default_paths_for(language):
    data = os.path.join(_ROOT, "data", LANGUAGE_WORKFLOW_DIR[language])
    proc = os.path.join(data, "process")
    return {
        "digest": os.path.join(proc, "stage_1a_digest_all_files.json"),
        "merge": os.path.join(proc, "stage_1b_merge_all_files.json"),
        "candidates": [os.path.join(proc, "stage_2_merged.json")],
        "pool": [os.path.join(proc, "stage_3a.json")],
        "dataset": os.path.join(proc, "stage_3b_dataset.json"),
        "gap_report": os.path.join(proc, "stage_3b_gap_report.json"),
        "backups": [os.path.join(proc, "stage_3b_backups.json")],
        "outdir": os.path.join(data, "eda"),
    }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def source_of(locator: str) -> str:
    """First segment of a locator '<file>:<page>:<heading>' -> the file name."""
    if not locator:
        return "(unknown)"
    name = locator.split(":")[0].strip()
    return name.replace(".pdf", "").replace("_REDUCED_FILE", "")


def short_src(name: str, n: int = 22) -> str:
    return name if len(name) <= n else name[: n - 1] + "…"


def iter_example_facts(cell):
    """Yield (fact_text, example_or_None) for a 1a by_grade cell, tolerating both
    shapes of `example_facts`:
        old  -> ["<fact string>", ...]
        v9   -> [{"fact": "...", "example": "<verbatim form + gloss>"}, ...]
    Old string data yields example=None (so capture counts degrade to 0/N)."""
    for ef in cell.get("example_facts") or []:
        if isinstance(ef, dict):
            yield ef.get("fact", ""), ef.get("example")
        else:
            yield ef, None


def load_many(paths):
    """Load and concatenate one or more bare-array JSON files into a single list.
    Accepts a single path (str) or a list of paths."""
    if isinstance(paths, str):
        paths = [paths]
    items = []
    for p in paths:
        data = load(p)
        items.extend(data if isinstance(data, list) else [data])
    return items


def heatmap(ax, mat, title, *, fmt="{:.0f}", cmap=HEAT, zero_blank=True):
    """Domain (rows) x Grade (cols) heatmap with annotations and gap markers."""
    mat = np.asarray(mat, dtype=float)
    vmax = mat.max() if mat.max() > 0 else 1
    ax.imshow(mat, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(GRADES)))
    ax.set_xticklabels([GRADE_LABEL[g] for g in GRADES], fontsize=8)
    ax.set_yticks(range(len(DOMAINS)))
    ax.set_yticklabels([DOMAIN_LABEL[d] for d in DOMAINS], fontsize=8)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            v = mat[r, c]
            if v == 0:
                if zero_blank:
                    ax.text(
                        c,
                        r,
                        "·",
                        ha="center",
                        va="center",
                        color="#bbbbbb",
                        fontsize=11,
                    )
                continue
            ax.text(
                c,
                r,
                fmt.format(v),
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if v > 0.6 * vmax else "#222222",
            )
    ax.set_xticks(np.arange(-0.5, len(GRADES), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(DOMAINS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)


def barh_counter(ax, counter, title, order=None, color="#4e79a7", colormap=None):
    items = order if order else sorted(counter, key=lambda k: -counter[k])
    items = [k for k in items if counter.get(k, 0) > 0 or order]
    vals = [counter.get(k, 0) for k in items]
    colors = [colormap[k] for k in items] if colormap else color
    y = range(len(items))
    ax.barh(list(y), vals, color=colors)
    ax.set_yticks(list(y))
    ax.set_yticklabels(items, fontsize=8)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    for i, v in zip(y, vals):
        ax.text(v + max(vals + [1]) * 0.01, i, str(v), va="center", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    return items


def stamp(fig, stage, n_note):
    fig.text(
        0.01,
        0.005,
        f"Linguistic Knowledge Benchmark — EDA · {stage}",
        fontsize=8,
        color="#888888",
    )
    fig.text(0.99, 0.005, n_note, fontsize=8, color="#888888", ha="right")


# --------------------------------------------------------------------------- #
# STAGE 1a — DIGEST (raw supply)
# --------------------------------------------------------------------------- #
def eda_digest(path, outpath):
    data = load(path)
    all_docs = data["resource_digest"]["by_document"]

    # empty DIGEST_SOURCES = use all sources; non-empty = filter to listed sources only
    if DIGEST_SOURCES:
        active = set(DIGEST_SOURCES)
        docs = [d for d in all_docs if d.get("source_name") in active]
        excluded = [
            d.get("source_name") for d in all_docs if d.get("source_name") not in active
        ]
        unseen = [
            s
            for s in DIGEST_SOURCES
            if s not in {d.get("source_name") for d in all_docs}
        ]
        if excluded:
            print(f"  [1a] excluded {len(excluded)} source(s): {', '.join(excluded)}")
        if unseen:
            print(f"  [1a] listed but NOT found in digest JSON: {', '.join(unseen)}")
        print(f"  [1a] using {len(docs)}/{len(all_docs)} source(s)")
    else:
        docs = all_docs
        print(f"  [1a] using all {len(docs)} source(s)")

    facts = np.zeros((len(DOMAINS), len(GRADES)))  # summed testable facts
    facts_by_src = defaultdict(
        lambda: np.zeros(len(DOMAINS))
    )  # src -> per-domain facts
    cov_counter = Counter()
    grade_facts = Counter()
    src_total = Counter()
    tier_counter = Counter()
    ex_total = 0  # example_facts entries seen
    ex_with_example = 0  # ... that carry a verbatim example (v9 capture)

    for doc in docs:
        src = source_of(doc.get("source_name", "(unknown)"))
        tier_counter[doc.get("tier") or "(unrecorded)"] += 1
        for dom in doc.get("by_domain", []):
            d = dom.get("domain")
            if d not in DOMAINS:
                continue
            di = DOMAINS.index(d)
            for cell in dom.get("by_grade", []):
                g = cell.get("grade")
                n = cell.get("n_testable_facts_est") or 0
                cov_counter[cell.get("coverage", "absent")] += 1
                if g in GRADES:
                    facts[di, GRADES.index(g)] += n
                    grade_facts[g] += n
                facts_by_src[src][di] += n
                src_total[src] += n
                for _fact, example in iter_example_facts(cell):
                    ex_total += 1
                    if example:
                        ex_with_example += 1

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(
        "Stage 1a · Digest — raw supply per source (testable-fact estimates)",
        fontsize=14,
        fontweight="bold",
    )
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.3, 1.0],
        hspace=0.45,
        wspace=0.32,
        left=0.07,
        right=0.97,
        top=0.9,
        bottom=0.08,
    )

    # (1) headline supply heatmap
    heatmap(
        fig.add_subplot(gs[0, 0:2]),
        facts,
        "Supply map — Σ testable facts by domain × grade  (· = gap)",
    )

    # (2) facts per domain, stacked by source document (provenance of supply)
    ax = fig.add_subplot(gs[0, 2])
    srcs = sorted(facts_by_src, key=lambda s: -src_total[s])
    bottom = np.zeros(len(DOMAINS))
    yy = np.arange(len(DOMAINS))
    for i, s in enumerate(srcs):
        vals = facts_by_src[s]
        ax.barh(
            yy,
            vals,
            left=bottom,
            color=SRC_PALETTE[i % len(SRC_PALETTE)],
            label=short_src(s),
        )
        bottom += vals
    ax.set_yticks(yy)
    ax.set_yticklabels([DOMAIN_LABEL[d] for d in DOMAINS], fontsize=8)
    ax.invert_yaxis()
    ax.set_title("Facts per domain, by source", fontsize=11, fontweight="bold", pad=8)
    ax.legend(fontsize=7, loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    # (3) supply across grades
    ax = fig.add_subplot(gs[1, 0])
    vals = [grade_facts.get(g, 0) for g in GRADES]
    ax.bar([GRADE_LABEL[g] for g in GRADES], vals, color="#fd8d3c")
    ax.set_title(
        "Supply across grades (Σ facts)", fontsize=11, fontweight="bold", pad=8
    )
    for i, v in enumerate(vals):
        if v:
            ax.text(i, v, str(int(v)), ha="center", va="bottom", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    # (4) coverage-quality of the (doc,domain,grade) cells
    ax = fig.add_subplot(gs[1, 1])
    vals = [cov_counter.get(c, 0) for c in COVERAGE_ORDER]
    ax.bar(COVERAGE_ORDER, vals, color=[COVERAGE_COLOR[c] for c in COVERAGE_ORDER])
    ax.set_title("Coverage quality of cells", fontsize=11, fontweight="bold", pad=8)
    for i, v in enumerate(vals):
        if v:
            ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    # (5) per-document yield + tier annotation
    ax = fig.add_subplot(gs[1, 2])
    barh_counter(
        ax,
        src_total,
        "Total facts per document",
        order=sorted(src_total, key=lambda s: -src_total[s]),
    )
    ax.set_yticklabels(
        [short_src(s) for s in sorted(src_total, key=lambda s: -src_total[s])],
        fontsize=8,
    )
    tier_note = "tier: " + ", ".join(f"{k}×{v}" for k, v in tier_counter.items())
    ax.set_xlabel(tier_note, fontsize=8, color="#666666")

    ex_note = (
        f" · {ex_with_example}/{ex_total} facts carry a verbatim example"
        if ex_total
        else ""
    )
    stamp(fig, "1a digest", f"{len(docs)} documents{ex_note}")
    fig.savefig(outpath, format="svg", bbox_inches="tight")
    plt.close(fig)
    return facts, grade_facts


# --------------------------------------------------------------------------- #
# STAGE 1b — MERGE (consolidated feature map)
# --------------------------------------------------------------------------- #
def eda_merge(path, outpath):
    data = load(path)
    by_domain = data["feature_map"]["by_domain"]

    fcount = np.zeros((len(DOMAINS), len(GRADES)))
    gpe_counter = Counter()
    tier_counter = Counter()
    nsrc_counter = Counter()  # #sources per feature
    subdoms = defaultdict(set)
    src_features = Counter()

    total_feats = 0
    feats_with_example = 0  # features carrying an `example` (v9 carry-through)
    for block in by_domain:
        d = block.get("domain")
        if d not in DOMAINS:
            continue
        di = DOMAINS.index(d)
        for feat in block.get("features", []):
            total_feats += 1
            if feat.get("example"):
                feats_with_example += 1
            g = feat.get("grade")
            if g in GRADES:
                fcount[di, GRADES.index(g)] += 1
            gpe_counter[feat.get("grade_placement_evidence", "(none)")] += 1
            sources = feat.get("sources", []) or []
            nsrc_counter[min(len(sources), 3)] += 1  # bucket 3+ together
            if feat.get("subdomain"):
                subdoms[d].add(feat["subdomain"])
            seen_src = set()
            for s in sources:
                tier_counter[s.get("tier", "(none)")] += 1
                sn = source_of(s.get("locator", ""))
                if sn not in seen_src:  # count a source once per feature
                    src_features[sn] += 1
                    seen_src.add(sn)

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(
        "Stage 1b · Merge — consolidated feature map (coverage & grounding)",
        fontsize=14,
        fontweight="bold",
    )
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.3, 1.0],
        hspace=0.5,
        wspace=0.34,
        left=0.07,
        right=0.97,
        top=0.9,
        bottom=0.08,
    )

    # (1) headline feature-count heatmap
    heatmap(
        fig.add_subplot(gs[0, 0:2]),
        fcount,
        "Feature map — # features by domain × grade  (· = gap)",
    )

    # (2) subdomain breadth per domain
    ax = fig.add_subplot(gs[0, 2])
    vals = [len(subdoms.get(d, set())) for d in DOMAINS]
    yy = np.arange(len(DOMAINS))
    ax.barh(yy, vals, color="#6a51a3")
    ax.set_yticks(yy)
    ax.set_yticklabels([DOMAIN_LABEL[d] for d in DOMAINS], fontsize=8)
    ax.invert_yaxis()
    ax.set_title(
        "Distinct subdomains per domain", fontsize=11, fontweight="bold", pad=8
    )
    for i, v in enumerate(vals):
        if v:
            ax.text(v, i, f" {v}", va="center", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    # (3) grade-placement evidence basis
    ax = fig.add_subplot(gs[1, 0])
    barh_counter(ax, gpe_counter, "Grade-placement evidence")

    # (4) source tier backing the features
    ax = fig.add_subplot(gs[1, 1])
    barh_counter(
        ax,
        tier_counter,
        "Source tier (feature citations)",
        order=TIER_ORDER,
        colormap=TIER_COLOR,
    )

    # (5) corroboration: #sources per feature  +  source contribution
    ax = fig.add_subplot(gs[1, 2])
    labels = {1: "1 source", 2: "2 sources", 3: "3+ sources"}
    order = [1, 2, 3]
    vals = [nsrc_counter.get(k, 0) for k in order]
    ax.bar([labels[k] for k in order], vals, color="#2c7fb8")
    ax.set_title("Corroboration per feature", fontsize=11, fontweight="bold", pad=8)
    for i, v in enumerate(vals):
        if v:
            ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    multi = sum(nsrc_counter.get(k, 0) for k in (2, 3))
    pct = 100 * multi / total_feats if total_feats else 0
    ax.set_xlabel(
        f"{pct:.0f}% of features cite ≥2 sources", fontsize=8, color="#666666"
    )

    stamp(
        fig,
        "1b merge",
        f"{total_feats} features · {feats_with_example} with example · "
        + " ".join(f"{short_src(s)}:{c}" for s, c in src_features.most_common(3)),
    )
    fig.savefig(outpath, format="svg", bbox_inches="tight")
    plt.close(fig)
    return fcount


# --------------------------------------------------------------------------- #
# STAGE 3b — ASSEMBLE (the final dataset)
# (was eda_create/eda_2_create.svg — the final dataset is Stage 3b's output in v9)
# --------------------------------------------------------------------------- #
def _feature_counts_by_cell(feature_map_path, exclude_grades=()):
    """Count 1b features per (domain, grade) cell, skipping excluded grades.

    Returns a len(DOMAINS) x len(GRADES) matrix — the raw supply the feature_map
    allocation divides into proportions (mirrors stage_3b_assemble's derivation)."""
    data = load(feature_map_path)
    mat = np.zeros((len(DOMAINS), len(GRADES)))
    excluded = set(exclude_grades)
    for block in data.get("feature_map", {}).get("by_domain", []):
        d = block.get("domain")
        if d not in DOMAINS:
            continue
        di = DOMAINS.index(d)
        for feat in block.get("features", []):
            g = feat.get("grade")
            if g in GRADES and g not in excluded:
                mat[di, GRADES.index(g)] += 1
    return mat


def eda_final(
    path, outpath, gap_report_path=None, backups_path=None, feature_map_path=None
):
    items = load(path)

    # Stage-3b extras (optional): only present once the assembler has been run.
    gap_report = (
        load(gap_report_path)
        if gap_report_path and os.path.exists(gap_report_path)
        else None
    )
    backups = (
        load_many(backups_path)
        if backups_path
        and os.path.exists(
            backups_path if isinstance(backups_path, str) else backups_path[0]
        )
        else None
    )

    icount = np.zeros((len(DOMAINS), len(GRADES)))
    atype_by_domain = defaultdict(Counter)
    atype = Counter()
    grade_counter = Counter()
    tier_counter = Counter()
    gpe_counter = Counter()
    src_items = Counter()
    qid_seen = Counter()
    svr_counter = Counter()
    geeap_counter = Counter()
    svr_geeap = Counter()  # (svr_strand, geeap_component) joint counts
    issues = Counter()

    for it in items:
        d, g = it.get("domain"), it.get("grade")
        if d in DOMAINS and g in GRADES:
            icount[DOMAINS.index(d), GRADES.index(g)] += 1
        at = it.get("answer_type", "(none)")
        atype[at] += 1
        if d in DOMAINS:
            atype_by_domain[d][at] += 1
        grade_counter[g] += 1
        gpe_counter[it.get("grade_placement_evidence", "(none)")] += 1
        svr = it.get("svr_strand", "(none)")
        geeap = it.get("geeap_component", "(none)")
        svr_counter[svr] += 1
        geeap_counter[geeap] += 1
        svr_geeap[(svr, geeap)] += 1
        for s in it.get("sources", []) or []:
            tier_counter[s.get("tier", "(none)")] += 1
            src_items[source_of(s.get("locator", ""))] += 1
        qid_seen[it.get("question_id")] += 1

        # --- data-quality checks ---
        exp = it.get("expected_answers", []) or []
        if not exp:
            issues["no expected answer"] += 1
        if at == "multiple_choice":
            opts = it.get("options", {}) or {}
            if len(opts) < 2:
                issues["MCQ <2 options"] += 1
            # expected answer should map to an option (key or value)
            ok = any(e in opts or e in opts.values() for e in exp)
            if exp and not ok:
                issues["MCQ answer not in options"] += 1
        if not (it.get("sources")):
            issues["no source / ungrounded"] += 1
    dup_ids = sum(1 for q, c in qid_seen.items() if c > 1)
    if dup_ids:
        issues["duplicate question_id"] += dup_ids

    has_extra = gap_report is not None
    quota_source = gap_report.get("quota_source") if has_extra else None
    nrows = 7 if has_extra else 4
    fig = plt.figure(figsize=(16, 21 if has_extra else 13))
    alloc_note = f"  ·  allocation: {quota_source}" if quota_source else ""
    fig.suptitle(
        f"Stage 3b · Assemble — the final dataset ({len(items)} items): coverage, balance, provenance{alloc_note}",  # noqa: E501
        fontsize=14,
        fontweight="bold",
    )
    gs = fig.add_gridspec(
        nrows,
        3,
        height_ratios=[1.25, 0.95, 0.85, 0.85]
        + ([1.25, 0.85, 1.15] if has_extra else []),
        hspace=0.6,
        wspace=0.34,
        left=0.07,
        right=0.97,
        top=0.93,
        bottom=0.05,
    )

    # (1) headline item-count heatmap
    heatmap(
        fig.add_subplot(gs[0, 0:2]),
        icount,
        "Dataset coverage — # items by domain × grade  (· = gap)",
    )

    # (2) answer_type balance per domain (stacked)
    ax = fig.add_subplot(gs[0, 2])
    yy = np.arange(len(DOMAINS))
    types = sorted(atype, key=lambda t: -atype[t])
    tcol = {"multiple_choice": "#4e79a7", "short_form": "#f28e2b"}
    bottom = np.zeros(len(DOMAINS))
    for t in types:
        vals = np.array([atype_by_domain[d].get(t, 0) for d in DOMAINS])
        ax.barh(yy, vals, left=bottom, color=tcol.get(t, "#9c755f"), label=t)
        bottom += vals
    ax.set_yticks(yy)
    ax.set_yticklabels([DOMAIN_LABEL[d] for d in DOMAINS], fontsize=8)
    ax.invert_yaxis()
    ax.set_title("Answer type per domain", fontsize=11, fontweight="bold", pad=8)
    ax.legend(fontsize=7, loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    mc = atype.get("multiple_choice", 0)
    ax.set_xlabel(
        f"overall {mc}/{len(items) - mc} MCQ/short ({100 * mc / len(items):.0f}% MCQ)",
        fontsize=8,
        color="#666666",
    )

    # (3) grade distribution
    ax = fig.add_subplot(gs[1, 0])
    vals = [grade_counter.get(g, 0) for g in GRADES]
    ax.bar([GRADE_LABEL[g] for g in GRADES], vals, color="#fd8d3c")
    ax.set_title("Items per grade", fontsize=11, fontweight="bold", pad=8)
    for i, v in enumerate(vals):
        if v:
            ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    # (4) tier hierarchy used
    ax = fig.add_subplot(gs[1, 1])
    barh_counter(
        ax,
        tier_counter,
        "Source tier (item citations)",
        order=TIER_ORDER,
        colormap=TIER_COLOR,
    )

    # (5) evidence basis
    ax = fig.add_subplot(gs[1, 2])
    barh_counter(ax, gpe_counter, "Grade-placement evidence")

    # (6) provenance: items per source
    ax = fig.add_subplot(gs[2, 0:2])
    top = dict(src_items.most_common(8))
    barh_counter(ax, top, "Items per source")
    ax.set_yticklabels(
        [short_src(s, 40) for s in sorted(top, key=lambda s: -top[s])], fontsize=8
    )

    # (7) data-quality panel
    ax = fig.add_subplot(gs[2, 2])
    ax.axis("off")
    ax.set_title("Data-quality checks", fontsize=11, fontweight="bold", pad=8)
    lines = [
        f"items: {len(items)}",
        f"distinct question_id: {len(qid_seen)}",
        f"distinct subdomains: {len(set(i.get('subdomain') for i in items))}",
        f"domains covered: {int((icount.sum(axis=1) > 0).sum())}/9",
        f"grades covered: {int((icount.sum(axis=0) > 0).sum())}/{len(GRADES)}",
        "",
    ]
    if issues:
        lines.append("issues:")
        lines += [f"  ⚠ {k}: {v}" for k, v in issues.items()]
    else:
        lines.append("✓ no structural issues found")
    ax.text(
        0.0,
        0.95,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
        transform=ax.transAxes,
        color="#b30000" if issues else "#1a7a1a",
    )

    # (8) SVR strand distribution across the dataset
    ax = fig.add_subplot(gs[3, 0])
    barh_counter(ax, svr_counter, "SVR strand (items)", color="#2c7fb8")

    # (9) GEEAP component distribution across the dataset
    # ordered as the tagging prompt explains the components
    ax = fig.add_subplot(gs[3, 1])
    barh_counter(
        ax,
        geeap_counter,
        "GEEAP component (items)",
        order=GEEAP_ORDER,
        color="#6a51a3",
    )

    # (10) SVR strand × GEEAP component crosstab (are the two tags consistent?)
    ax = fig.add_subplot(gs[3, 2])
    svr_cols = ["decoding", "language_comprehension", "outside_flr_construct"]
    svr_cols += [s for s in svr_counter if s not in svr_cols]
    geeap_rows = GEEAP_ORDER + [g for g in geeap_counter if g not in GEEAP_ORDER]
    mat = np.array(
        [[svr_geeap.get((s, g), 0) for s in svr_cols] for g in geeap_rows],
        dtype=float,
    )
    vmax = mat.max() if mat.max() > 0 else 1
    ax.imshow(mat, cmap=HEAT, vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(svr_cols)))
    ax.set_xticklabels(
        [
            s.replace("language_comprehension", "lang_comp").replace(
                "outside_flr_construct", "outside"
            )
            for s in svr_cols
        ],  # noqa: E501
        fontsize=7,
        rotation=25,
        ha="right",
    )
    ax.set_yticks(range(len(geeap_rows)))
    ax.set_yticklabels(geeap_rows, fontsize=7)
    ax.set_title("SVR strand × GEEAP crosstab", fontsize=11, fontweight="bold", pad=8)
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            v = mat[r, c]
            ax.text(
                c,
                r,
                f"{int(v)}" if v else "·",
                ha="center",
                va="center",
                fontsize=8,
                color="white"
                if v > 0.6 * vmax
                else ("#bbbbbb" if not v else "#222222"),
            )
    ax.set_xticks(np.arange(-0.5, len(svr_cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(geeap_rows), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)

    # ---- Stage-3b assembly panels (only when the gap_report is supplied) ---- #
    counts = None
    if has_extra:
        counts = gap_report.get("counts", {}) or {}
        cells = gap_report.get("cells", []) or []
        total_selected = gap_report.get("total_selected", len(items))

        # Reconstruct the per-cell quota (the allocation). The gap_report lists only
        # deficit≠0 cells, so start from the dataset's own D×G counts (icount == filled)
        # and override the listed cells with their quota. Exact, because a filled==quota
        # cell has deficit 0 and is omitted. Reused by panels (11) and (16).
        quota = icount.copy()
        for c in cells:
            d, g = c.get("domain"), c.get("grade")
            if d in DOMAINS and g in GRADES:
                quota[DOMAINS.index(d), GRADES.index(g)] = c.get("quota", 0)

        # (11) quota vs filled — the deficit grid.
        deficit = quota - icount
        heatmap(
            fig.add_subplot(gs[4, 0:2]),
            deficit,
            "Quota deficit — quota − filled  (+ = under-filled, · = on target)",
            fmt="{:+.0f}",
        )

        # (12) QC → dataset funnel: pool → qc_passed → dedup reps → selected
        ax = fig.add_subplot(gs[4, 2])
        labels = ["candidates", "QC pass", "deduped", "shipped"]
        vals = [
            counts.get("pool", 0),
            counts.get("qc_passed", 0),
            counts.get("dup_groups", 0),
            total_selected,
        ]
        ax.bar(labels, vals, color=["#bab0ac", "#76b7b2", "#59a14f", "#1a7a1a"])
        ax.set_title(
            "Candidate → dataset funnel", fontsize=11, fontweight="bold", pad=8
        )
        for i, v in enumerate(vals):
            if v:
                ax.text(i, v, str(int(v)), ha="center", va="bottom", fontsize=8)
        ax.tick_params(axis="x", labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)

        # (13) answer-type mix vs the 0.5 target
        ax = fig.add_subplot(gs[5, 0])
        atd = gap_report.get("answer_type_distribution", {}) or {}
        at_keys = sorted(atd, key=lambda k: -atd[k])
        av = [atd[k] for k in at_keys]
        ax.bar(
            [
                k.replace("multiple_choice", "MCQ").replace("short_form", "short")
                for k in at_keys
            ],
            av,
            color=[ATYPE_COLOR.get(k, "#9c755f") for k in at_keys],
        )
        ax.axhline(0.5 * total_selected, ls="--", lw=1, color="#666666")
        ax.set_title("Answer type vs 50% target", fontsize=11, fontweight="bold", pad=8)
        for i, v in enumerate(av):
            if v:
                ax.text(i, v, str(int(v)), ha="center", va="bottom", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

        # (14) correct-answer letter balance across A–D
        ax = fig.add_subplot(gs[5, 1])
        ld = gap_report.get("letter_distribution", {}) or {}
        letters = ["A", "B", "C", "D"]
        lv = [ld.get(k, 0) for k in letters]
        n_mcq = sum(lv)
        ax.bar(letters, lv, color="#4e79a7")
        if n_mcq:
            ax.axhline(n_mcq / 4, ls="--", lw=1, color="#666666")
        ax.set_title("MCQ answer-letter balance", fontsize=11, fontweight="bold", pad=8)
        for i, v in enumerate(lv):
            if v:
                ax.text(i, v, str(int(v)), ha="center", va="bottom", fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)

        # (15) selection pool sizes — bench depth for the human gate
        ax = fig.add_subplot(gs[5, 2])
        n_backups = len(backups) if backups is not None else counts.get("backups", 0)
        pools = Counter(
            {
                "selected": total_selected,
                "backups": n_backups,
                "rejected": counts.get("qc_rejected", 0),
            }
        )
        barh_counter(
            ax,
            pools,
            "Selection pools",
            order=["selected", "backups", "rejected"],
            colormap={
                "selected": "#1a7a1a",
                "backups": "#f28e2b",
                "rejected": "#b30000",
            },
        )

        # (16) ALLOCATION — how the per-cell quota was decided (type + derivation).
        # feature_map: show the 3 derivation steps (counts → proportions → quota).
        # hard-coded (or feature map unavailable): show the final quota table + a note.
        fm_ok = bool(feature_map_path) and os.path.exists(
            feature_map_path
            if isinstance(feature_map_path, str)
            else (feature_map_path or [None])[0]
        )
        if quota_source == "feature_map" and fm_ok:
            excluded = gap_report.get("excluded_grades", []) or []
            fm_path = (
                feature_map_path
                if isinstance(feature_map_path, str)
                else feature_map_path[0]
            )
            fcounts = _feature_counts_by_cell(fm_path, excluded)
            ftotal = gap_report.get("feature_total") or int(fcounts.sum())
            target = gap_report.get("quota_target_total") or int(quota.sum())
            props = (fcounts / ftotal * 100.0) if ftotal else fcounts
            heatmap(
                fig.add_subplot(gs[6, 0]),
                fcounts,
                f"① Feature supply (1b) — Σ={int(fcounts.sum())}",
            )
            axp = fig.add_subplot(gs[6, 1])
            heatmap(axp, props, "② Proportion of features (%)", fmt="{:.1f}")
            excl_txt = ", ".join(GRADE_LABEL.get(g, g) for g in excluded) or "none"
            axp.set_xlabel(
                f"allocation = feature_map · target N={target} · "
                f"features={int(ftotal)} · excluded: {excl_txt}",
                fontsize=8,
                color="#666666",
            )
            heatmap(
                fig.add_subplot(gs[6, 2]),
                quota,
                f"③ Allocation quota — round(② × {target})",
            )
        else:
            label = (
                "hard-coded DEFAULT_QUOTA"
                if quota_source == "hard-coded"
                else "derived"
                if quota_source == "feature_map"
                else "quota"
            )
            heatmap(
                fig.add_subplot(gs[6, 0:2]),
                quota,
                f"Allocation quota — {label} table",
            )
            ax = fig.add_subplot(gs[6, 2])
            ax.axis("off")
            ax.set_title("Allocation", fontsize=11, fontweight="bold", pad=8)
            note = [
                f"type: {quota_source or 'unknown'}",
                f"Σ quota: {int(quota.sum())}",
                "",
            ]
            if quota_source == "hard-coded":
                note += ["fixed table —", "no feature-map derivation"]
            elif quota_source == "feature_map":
                note += ["feature map not supplied;", "showing final quota only"]
            ax.text(
                0.0,
                0.9,
                "\n".join(note),
                va="top",
                ha="left",
                fontsize=9,
                family="monospace",
                transform=ax.transAxes,
                color="#333333",
            )

    stamp(
        fig,
        "3b assemble",
        f"{len(items)} items"
        + (
            f" · pool {counts.get('pool', '?')} → shipped {len(items)}"
            if has_extra
            else ""
        ),
    )
    fig.savefig(outpath, format="svg", bbox_inches="tight")
    plt.close(fig)
    return counts


# --------------------------------------------------------------------------- #
# STAGE 2 — GENERATE (sharded candidate pool)
# --------------------------------------------------------------------------- #
def _feature_key(obj, name_field):
    """Structural identity of a feature/candidate: (domain, subdomain, grade, name)."""
    return (
        obj.get("domain"),
        obj.get("subdomain"),
        obj.get("grade"),
        obj.get(name_field),
    )


def eda_candidates(paths, outpath, merge_path=None):
    cands = load_many(paths)

    ccount = np.zeros((len(DOMAINS), len(GRADES)))
    shard_counter = Counter()
    atype = Counter()
    atype_by_domain = defaultdict(Counter)
    per_feature = Counter()  # structural key -> #candidates (oversampling)
    id_seen = Counter()  # (shard_id, candidate_id) uniqueness
    issues = Counter()

    for c in cands:
        d, g = c.get("domain"), c.get("grade")
        if d in DOMAINS and g in GRADES:
            ccount[DOMAINS.index(d), GRADES.index(g)] += 1
        shard_counter[c.get("shard_id", "(none)")] += 1
        at = c.get("answer_type", "(none)")
        atype[at] += 1
        if d in DOMAINS:
            atype_by_domain[d][at] += 1
        per_feature[_feature_key(c, "linguistic_feature")] += 1
        id_seen[(c.get("shard_id"), c.get("candidate_id"))] += 1
        if at == "multiple_choice":
            opts = c.get("options", {}) or {}
            if len(opts) != 4:
                issues["MCQ options≠4"] += 1
            exp = c.get("expected_answers", []) or []
            if exp and not any(e in opts for e in exp):
                issues["MCQ answer not an option key"] += 1
        if not c.get("sources"):
            issues["no source / ungrounded"] += 1

    # feature_map coverage cross-reference (optional)
    covered = uncovered = total_map_feats = None
    map_by_domain = cov_by_domain = None
    if merge_path and os.path.exists(
        merge_path if isinstance(merge_path, str) else merge_path[0]
    ):
        fmap = load(merge_path if isinstance(merge_path, str) else merge_path[0])
        cand_keys = set(per_feature)
        map_by_domain = Counter()
        cov_by_domain = Counter()
        for block in fmap.get("feature_map", {}).get("by_domain", []):
            dblk = block.get("domain")
            for feat in block.get("features", []):
                key = (
                    dblk,
                    feat.get("subdomain"),
                    feat.get("grade"),
                    feat.get("feature"),
                )
                map_by_domain[dblk] += 1
                if key in cand_keys:
                    cov_by_domain[dblk] += 1
        total_map_feats = sum(map_by_domain.values())
        covered = sum(cov_by_domain.values())
        uncovered = total_map_feats - covered

    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(
        f"Stage 2 · Generate — candidate pool ({len(cands)} candidates, sharded, oversampled)",  # noqa: E501
        fontsize=14,
        fontweight="bold",
    )
    gs = fig.add_gridspec(
        3,
        3,
        height_ratios=[1.3, 1.0, 0.5],
        hspace=0.55,
        wspace=0.34,
        left=0.07,
        right=0.97,
        top=0.91,
        bottom=0.06,
    )

    # (1) candidate coverage heatmap
    heatmap(
        fig.add_subplot(gs[0, 0:2]),
        ccount,
        "Candidates by domain × grade  (· = gap)",
    )

    # (2) candidates per shard
    ax = fig.add_subplot(gs[0, 2])
    order = sorted(shard_counter, key=lambda s: -shard_counter[s])
    barh_counter(ax, shard_counter, "Candidates per shard", order=order)
    for i, patch in enumerate(ax.patches):  # recolour bars by shard palette
        patch.set_color(SRC_PALETTE[i % len(SRC_PALETTE)])

    # (3) oversampling — candidates per feature
    ax = fig.add_subplot(gs[1, 0])
    buckets = Counter()
    for n in per_feature.values():
        buckets["1" if n == 1 else "2" if n == 2 else "3+"] += 1
    keys = ["1", "2", "3+"]
    vals = [buckets.get(k, 0) for k in keys]
    ax.bar(keys, vals, color="#59a14f")
    ax.set_title(
        "Oversampling — candidates/feature", fontsize=11, fontweight="bold", pad=8
    )
    for i, v in enumerate(vals):
        if v:
            ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    ax.set_xlabel(
        f"{len(per_feature)} distinct features hit", fontsize=8, color="#666666"
    )
    ax.spines[["top", "right"]].set_visible(False)

    # (4) feature_map coverage per domain (overlay: covered over in-map total)
    ax = fig.add_subplot(gs[1, 1])
    yy = np.arange(len(DOMAINS))
    if map_by_domain is not None:
        tot = [map_by_domain.get(d, 0) for d in DOMAINS]
        cov = [cov_by_domain.get(d, 0) for d in DOMAINS]
        ax.barh(yy, tot, color="#d9d9d9", label="in map")
        ax.barh(yy, cov, color="#4e79a7", label="≥1 candidate")
        ax.set_yticks(yy)
        ax.set_yticklabels([DOMAIN_LABEL[d] for d in DOMAINS], fontsize=8)
        ax.invert_yaxis()
        ax.legend(fontsize=7, loc="lower right", frameon=False)
        ax.set_xlabel(
            f"{covered}/{total_map_feats} map features covered",
            fontsize=8,
            color="#666666",
        )
    else:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "feature-map coverage:\nmerge not supplied",
            ha="center",
            va="center",
            fontsize=9,
            color="#888888",
            transform=ax.transAxes,
        )
    ax.set_title("feature_map coverage", fontsize=11, fontweight="bold", pad=8)
    ax.spines[["top", "right"]].set_visible(False)

    # (5) answer-type mix, stacked per domain
    ax = fig.add_subplot(gs[1, 2])
    types = sorted(atype, key=lambda t: -atype[t])
    bottom = np.zeros(len(DOMAINS))
    for t in types:
        v = np.array([atype_by_domain[d].get(t, 0) for d in DOMAINS])
        ax.barh(yy, v, left=bottom, color=ATYPE_COLOR.get(t, "#9c755f"), label=t)
        bottom += v
    ax.set_yticks(yy)
    ax.set_yticklabels([DOMAIN_LABEL[d] for d in DOMAINS], fontsize=8)
    ax.invert_yaxis()
    ax.set_title("Answer type per domain", fontsize=11, fontweight="bold", pad=8)
    ax.legend(fontsize=7, loc="lower right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    mc = atype.get("multiple_choice", 0)
    if cands:
        ax.set_xlabel(
            f"{100 * mc / len(cands):.0f}% MCQ (pre-3b)", fontsize=8, color="#666666"
        )

    # (6) counts / data-quality text panel
    ax = fig.add_subplot(gs[2, 0:3])
    ax.axis("off")
    dup_ids = sum(1 for c in id_seen.values() if c > 1)
    lines = [
        f"candidates: {len(cands)}    shards: {len(shard_counter)}    "
        f"distinct (shard,candidate) ids: {len(id_seen)}"
        + (f"  ⚠ {dup_ids} duplicated" if dup_ids else ""),
        f"distinct features hit: {len(per_feature)}"
        + (
            f"    feature_map coverage: {covered}/{total_map_feats} "
            f"({uncovered} uncovered)"
            if map_by_domain is not None
            else ""
        ),
    ]
    if issues:
        lines.append("issues: " + "   ".join(f"⚠ {k}: {v}" for k, v in issues.items()))
    else:
        lines.append("✓ no structural issues found")
    ax.text(
        0.0,
        0.9,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
        transform=ax.transAxes,
        color="#b30000" if issues or dup_ids else "#1a7a1a",
    )

    stamp(fig, "2 candidates", f"{len(cands)} candidates · {len(shard_counter)} shards")
    fig.savefig(outpath, format="svg", bbox_inches="tight")
    plt.close(fig)
    return len(cands), ccount


# --------------------------------------------------------------------------- #
# STAGE 3a — POOL & QC (annotation audit)
# --------------------------------------------------------------------------- #
def eda_qc(paths, outpath):
    pool = load_many(paths)

    pass_fail = Counter()
    rule_fails = Counter()
    pass_by_domain = Counter()
    total_by_domain = Counter()
    group_size = Counter()
    rank_dist = Counter()
    pass_by_domain_vec = np.zeros(len(DOMAINS))
    total_by_domain_vec = np.zeros(len(DOMAINS))

    for c in pool:
        verdict = c.get("qc_verdict") or {}
        passed = bool(verdict.get("pass", False))
        pass_fail["pass" if passed else "fail"] += 1
        for r in verdict.get("failed_rules", []) or []:
            rule_fails[r] += 1
        d = c.get("domain")
        if d in DOMAINS:
            di = DOMAINS.index(d)
            total_by_domain_vec[di] += 1
            if passed:
                pass_by_domain_vec[di] += 1
        total_by_domain[d] += 1
        if passed:
            pass_by_domain[d] += 1
        group_size[c.get("dup_group_id", c.get("candidate_id"))] += 1
        rank_dist[int(c.get("quality_rank", 1))] += 1

    n_groups = len(group_size)
    collapsed = sum(sz - 1 for sz in group_size.values())
    n_passed = pass_fail.get("pass", 0)
    n_rejected = pass_fail.get("fail", 0)

    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(
        f"Stage 3a · Pool & QC — annotation audit ({len(pool)} candidates)",
        fontsize=14,
        fontweight="bold",
    )
    gs = fig.add_gridspec(
        3,
        3,
        height_ratios=[1.3, 1.0, 0.5],
        hspace=0.55,
        wspace=0.34,
        left=0.09,
        right=0.97,
        top=0.91,
        bottom=0.06,
    )

    # (1) per-rule failure histogram
    ax = fig.add_subplot(gs[0, 0:2])
    yy = np.arange(len(QC_RULE_ORDER))
    vals = [rule_fails.get(r, 0) for r in QC_RULE_ORDER]
    ax.barh(
        yy,
        vals,
        color=["#c0504d" if r in QC_MCQ_ONLY else "#4e79a7" for r in QC_RULE_ORDER],
    )
    ax.set_yticks(yy)
    ax.set_yticklabels(
        [
            f"{r} {QC_RULES[r]}" + (" (MCQ)" if r in QC_MCQ_ONLY else "")
            for r in QC_RULE_ORDER
        ],
        fontsize=8,
    )
    ax.invert_yaxis()
    ax.set_title(
        "QC failures per rule  (red = MCQ-only)", fontsize=11, fontweight="bold", pad=8
    )
    for i, v in enumerate(vals):
        if v:
            ax.text(v, i, f" {v}", va="center", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    # (2) QC pass / fail
    ax = fig.add_subplot(gs[0, 2])
    pv = [pass_fail.get("pass", 0), pass_fail.get("fail", 0)]
    ax.bar(["pass", "fail"], pv, color=["#1a7a1a", "#b30000"])
    ax.set_title("QC pass / fail", fontsize=11, fontweight="bold", pad=8)
    for i, v in enumerate(pv):
        if v:
            ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    if pool:
        ax.set_xlabel(
            f"{100 * n_passed / len(pool):.0f}% pass", fontsize=8, color="#666666"
        )
    ax.spines[["top", "right"]].set_visible(False)

    # (3) per-domain pass rate
    ax = fig.add_subplot(gs[1, 0])
    yy = np.arange(len(DOMAINS))
    rate = np.divide(
        pass_by_domain_vec,
        total_by_domain_vec,
        out=np.zeros(len(DOMAINS)),
        where=total_by_domain_vec > 0,
    )
    ax.barh(yy, rate, color="#2c7fb8")
    ax.set_yticks(yy)
    ax.set_yticklabels([DOMAIN_LABEL[d] for d in DOMAINS], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_title("Per-domain QC pass rate", fontsize=11, fontweight="bold", pad=8)
    for i in range(len(DOMAINS)):
        t = int(total_by_domain_vec[i])
        if t:
            ax.text(
                rate[i],
                i,
                f" {int(pass_by_domain_vec[i])}/{t}",
                va="center",
                fontsize=7,
            )
    ax.spines[["top", "right"]].set_visible(False)

    # (4) dup_group size distribution
    ax = fig.add_subplot(gs[1, 1])
    gb = Counter()
    for sz in group_size.values():
        gb["1" if sz == 1 else "2" if sz == 2 else "3" if sz == 3 else "4+"] += 1
    keys = ["1", "2", "3", "4+"]
    vals = [gb.get(k, 0) for k in keys]
    ax.bar(keys, vals, color="#6a51a3")
    ax.set_title("dup_group size", fontsize=11, fontweight="bold", pad=8)
    for i, v in enumerate(vals):
        if v:
            ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    ax.set_xlabel(
        f"{n_groups} groups · {collapsed} dups collapse", fontsize=8, color="#666666"
    )
    ax.spines[["top", "right"]].set_visible(False)

    # (5) quality_rank distribution
    ax = fig.add_subplot(gs[1, 2])
    rb = Counter()
    for r, c in rank_dist.items():
        rb["1" if r == 1 else "2" if r == 2 else "3+"] += c
    keys = ["1", "2", "3+"]
    vals = [rb.get(k, 0) for k in keys]
    ax.bar(keys, vals, color="#fd8d3c")
    ax.set_title("quality_rank within group", fontsize=11, fontweight="bold", pad=8)
    for i, v in enumerate(vals):
        if v:
            ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    # (6) funnel / counts text panel
    ax = fig.add_subplot(gs[2, 0:3])
    ax.axis("off")
    lines = [
        f"pool: {len(pool)}   →   qc_passed: {n_passed}   →   "
        f"dup_groups (reps): {n_groups}   →   collapsed duplicates: {collapsed}",
        f"qc_rejected: {n_rejected}     "
        f"(Stage 3b keeps the rank-1 member of each passed dup_group; the rest become backups)",
    ]
    ax.text(
        0.0,
        0.9,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
        transform=ax.transAxes,
        color="#333333",
    )

    stamp(fig, "3a qc", f"{len(pool)} candidates · {n_passed} pass · {n_groups} groups")
    fig.savefig(outpath, format="svg", bbox_inches="tight")
    plt.close(fig)
    return {"pool": len(pool), "qc_passed": n_passed, "dup_groups": n_groups}


# --------------------------------------------------------------------------- #
# CROSS-STAGE FUNNEL (overview)
# --------------------------------------------------------------------------- #
def eda_funnel(outpath, totals, per_domain_mat=None, stage_labels=None):
    """totals: list of (label, value) descending through the pipeline.
    per_domain_mat: optional (len(DOMAINS) x len(stages)) matrix for the D-panel."""
    labels = [t[0] for t in totals]
    vals = [t[1] for t in totals]

    two = per_domain_mat is not None
    fig = plt.figure(figsize=(16, 8 if two else 5))
    fig.suptitle(
        "Cross-stage funnel — supply → features → candidates → qc → deduped → selected",
        fontsize=14,
        fontweight="bold",
    )
    gs = fig.add_gridspec(
        2 if two else 1,
        1,
        height_ratios=[1.0, 1.4] if two else [1.0],
        hspace=0.4,
        left=0.1,
        right=0.96,
        top=0.9,
        bottom=0.08,
    )

    # (1) global staircase
    ax = fig.add_subplot(gs[0, 0])
    yy = np.arange(len(labels))
    ax.barh(yy, vals, color=SRC_PALETTE[: len(labels)])
    ax.set_yticks(yy)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_title(
        "Item count through the pipeline", fontsize=11, fontweight="bold", pad=8
    )
    for i, v in enumerate(vals):
        ax.text(v, i, f" {int(v)}", va="center", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    # (2) per-domain funnel (inline imshow — heatmap() is hard-wired to GRADES x-axis)
    if two:
        ax = fig.add_subplot(gs[1, 0])
        mat = np.asarray(per_domain_mat, dtype=float)
        cols = stage_labels or labels
        vmax = mat.max() if mat.max() > 0 else 1
        ax.imshow(mat, cmap=HEAT, vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, fontsize=8, rotation=20, ha="right")
        ax.set_yticks(range(len(DOMAINS)))
        ax.set_yticklabels([DOMAIN_LABEL[d] for d in DOMAINS], fontsize=8)
        ax.set_title("Per-domain funnel", fontsize=11, fontweight="bold", pad=8)
        for r in range(mat.shape[0]):
            for c in range(mat.shape[1]):
                v = mat[r, c]
                ax.text(
                    c,
                    r,
                    f"{int(v)}" if v else "·",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white"
                    if v > 0.6 * vmax
                    else ("#bbbbbb" if not v else "#222222"),
                )
        ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(DOMAINS), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)
        ax.tick_params(which="minor", length=0)

    stamp(fig, "funnel", " → ".join(f"{lbl}:{int(v)}" for lbl, v in totals))
    fig.savefig(outpath, format="svg", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
def _domain_vec_from_list(items, field="domain"):
    """Count list items per canonical domain -> length-9 vector."""
    v = np.zeros(len(DOMAINS))
    for it in items:
        d = it.get(field)
        if d in DOMAINS:
            v[DOMAINS.index(d)] += 1
    return v


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--language",
        required=True,
        choices=sorted(LANGUAGE_WORKFLOW_DIR),
        help="which data/<language>_workflow/ tree to analyse",
    )
    # Path defaults are derived from --language (see default_paths_for above);
    # any of them can still be overridden individually on the CLI.
    ap.add_argument("--digest", default=None)
    ap.add_argument("--merge", default=None)
    # Stage 2 / 3a inputs may be one file or several per-shard files.
    ap.add_argument("--candidates", nargs="+", default=None)
    ap.add_argument("--pool", nargs="+", default=None)
    # final dataset (Stage 3b output; falls back to the v8 dataset for back-compat)
    ap.add_argument("--dataset", "--create", dest="dataset", default=None)
    ap.add_argument("--gap-report", dest="gap_report", default=None)
    ap.add_argument("--backups", nargs="+", default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument(
        "--no-funnel",
        action="store_true",
        help="skip the cross-stage funnel overview figure",
    )
    args = ap.parse_args()

    defaults = default_paths_for(args.language)
    for key in (
        "digest",
        "merge",
        "candidates",
        "pool",
        "dataset",
        "gap_report",
        "backups",
        "outdir",
    ):
        if getattr(args, key) is None:
            setattr(args, key, defaults[key])

    os.makedirs(args.outdir, exist_ok=True)

    def out(n):
        return os.path.join(args.outdir, n)

    def exists(path):
        if not path:
            return False
        first = path if isinstance(path, str) else (path[0] if path else None)
        return bool(first) and os.path.exists(first)

    written = 0
    # stage-total accumulators for the funnel (None = stage not run)
    supply = feats = n_cands = qc = None
    ccount = np.zeros((len(DOMAINS), len(GRADES)))
    fcount = np.zeros((len(DOMAINS), len(GRADES)))

    # --- 1a digest ---
    if exists(args.digest):
        print("Stage 1a (digest):", args.digest)
        facts, _ = eda_digest(args.digest, out("eda_1a_digest.svg"))
        supply = facts
        written += 1
        print("  ->", out("eda_1a_digest.svg"))
    else:
        print(f"  [1a digest] input not found: {args.digest} — skipping")

    # --- 1b merge ---
    if exists(args.merge):
        print("Stage 1b (merge): ", args.merge)
        fcount = eda_merge(args.merge, out("eda_1b_merge.svg"))
        feats = fcount
        written += 1
        print("  ->", out("eda_1b_merge.svg"))
    else:
        print(f"  [1b merge] input not found: {args.merge} — skipping")

    # --- 2 candidates ---
    if exists(args.candidates):
        print("Stage 2 (candidates):", args.candidates)
        n_cands, ccount = eda_candidates(
            args.candidates,
            out("eda_2_candidates.svg"),
            args.merge if exists(args.merge) else None,
        )
        written += 1
        print("  ->", out("eda_2_candidates.svg"))
    else:
        print(f"  [2 candidates] input not found: {args.candidates[0]} — skipping")

    # --- 3a pool & QC ---
    pool_items = None
    if exists(args.pool):
        print("Stage 3a (pool & QC):", args.pool)
        pool_items = load_many(args.pool)
        qc = eda_qc(args.pool, out("eda_3a_qc.svg"))
        written += 1
        print("  ->", out("eda_3a_qc.svg"))
    else:
        print(f"  [3a qc] input not found: {args.pool[0]} — skipping")

    # --- 3b final dataset (dataset required; gap_report/backups optional) ---
    dataset_items = None
    if exists(args.dataset):
        print("Stage 3b (final dataset):", args.dataset)
        dataset_items = load(args.dataset)
        eda_final(
            args.dataset,
            out("eda_3b_final.svg"),
            args.gap_report if exists(args.gap_report) else None,
            args.backups if exists(args.backups) else None,
            args.merge if exists(args.merge) else None,
        )
        written += 1
        print("  ->", out("eda_3b_final.svg"))
    else:
        print(f"  [3b final] input not found: {args.dataset} — skipping")

    # --- cross-stage funnel (needs at least the two ends) ---
    if not args.no_funnel and supply is not None and dataset_items is not None:
        selected = len(dataset_items)
        totals = [("supply (1a)", float(supply.sum()))]
        if feats is not None:
            totals.append(("features (1b)", float(feats.sum())))
        if n_cands is not None:
            totals.append(("candidates (2)", float(n_cands)))
        if qc is not None:
            totals.append(("qc_passed (3a)", float(qc["qc_passed"])))
            totals.append(("deduped (3a)", float(qc["dup_groups"])))
        totals.append(("selected (3b)", float(selected)))

        # per-domain matrix (best effort; column present only if that stage ran)
        cols, dom_cols = [], []
        cols.append("supply")
        dom_cols.append(supply.sum(axis=1))
        if feats is not None:
            cols.append("features")
            dom_cols.append(feats.sum(axis=1))
        if n_cands is not None:
            cols.append("candidates")
            dom_cols.append(ccount.sum(axis=1))
        if pool_items is not None:
            passed = [c for c in pool_items if (c.get("qc_verdict") or {}).get("pass")]
            cols.append("qc_passed")
            dom_cols.append(_domain_vec_from_list(passed))
        cols.append("selected")
        dom_cols.append(_domain_vec_from_list(dataset_items))
        per_domain = np.array(dom_cols).T if len(dom_cols) > 1 else None

        eda_funnel(out("eda_funnel.svg"), totals, per_domain, cols)
        written += 1
        print("  ->", out("eda_funnel.svg"))
    elif not args.no_funnel:
        print("  [funnel] need at least the 1a digest and the final dataset — skipping")

    print(f"\nDone. {written} SVG figure(s) written to", args.outdir)


if __name__ == "__main__":
    main()

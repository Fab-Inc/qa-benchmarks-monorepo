#!/usr/bin/env python3
"""
Three-step pipeline for pushing a small SAMPLE of the Linguistic Knowledge Benchmark
to the Hugging Face Hub, as a PRIVATE dataset. The full item set
(data/linguistic_knowledge_eval_<language>.json) stays private too -- publishing
everything would let it leak into future models' training data -- so this selects and
pushes a curated N-item spread instead.

Privacy: push ALWAYS creates/updates the repo as private (hardcoded, no flag to
override) -- this script never makes anything public. Going public is a deliberate,
separate, manual step you do afterward on huggingface.co once you're happy with what
got pushed.

Step 1 -- select-subset: pick --n-items items from one language's full eval dataset,
spread across domain, answer_type (format), and difficulty (from real model results,
when available). Writes data/hf_subset_<language>_<n-items>.json. See
select_subset()'s docstring below for exactly how the spread is chosen.

Step 2 -- write-card (optional, for review): write the dataset card (README) to a
local .md file, so its wording can be reviewed/edited before it's pushed. No
huggingface_hub dependency needed for this step.

Step 3 -- push: push one language's subset JSON (from step 1) to its own HF Hub
dataset repo (fixed per language, see REPO_IDS below -- not a CLI parameter), plus the
same dataset card -- unless --no-card. Always private (see "Privacy" above). SAFE BY
DEFAULT beyond that too: without --yes, push always dry-runs (prints what it would do;
touches nothing on the Hub, creates no repo, needs no HF deps installed) -- --yes is
what actually pushes (and is the only thing that needs huggingface_hub/datasets
installed).

Run:
    # Step 1, once per language:
    uv run python benchmarks/linguistic_knowledge_benchmark/scripts/push_to_hub.py select-subset --language english
    uv run python .../push_to_hub.py select-subset --language kiswahili

    # Step 2 (optional), review the card text before it's pushed:
    uv run python .../push_to_hub.py write-card --language english
    uv run python .../push_to_hub.py write-card --language kiswahili

    # Step 3, once per language -- the repo is fixed per language (REPO_IDS below),
    # not a CLI parameter. Dry-run first (no --yes, the default):
    uv run python .../push_to_hub.py push --language english
    # Only once you're happy with the dry-run output, actually publish:
    uv run python .../push_to_hub.py push --language english --yes
    uv run python .../push_to_hub.py push --language kiswahili --yes

What you need to specify:
    REPO_IDS below      Already set to the fixed repo for each language
                         (AI-for-Education/English-Linguistic-Knowledge-Benchmark,
                         AI-for-Education/Kiswahili-Linguistic-Knowledge-Benchmark) --
                         edit that dict directly if either should change; there is no
                         --repo-id override.
    HF auth             `huggingface-cli login` once, or set HF_TOKEN (or
                         HUGGINGFACE_API_KEY) in your .env -- see --token below.
    Dataset card wording  _card_content() below describes the benchmark and its
                         D1-D9/G0-G5 spine (from scripts/workflow/pipeline_overview_
                         v10.md and process_prompts_chat_ready_v10_english.md) --
                         edit DOMAIN_DEFINITIONS/GRADE_DEFINITIONS/CARD_TAGS to change
                         that wording.

Requires (declared in pyproject.toml; imported lazily so the dry-run path and
select-subset/write-card work without them installed):
    huggingface_hub
    datasets  (pulls in pyarrow, its Arrow storage backend)
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict

import pandas as pd
from dotenv import load_dotenv

LANGUAGES = ["english", "kiswahili"]
DEFAULT_N_ITEMS = 20
DEFAULT_SEED = 0
REPO_IDS = {
    "english": "AI-for-Education/English-Linguistic-Knowledge-Benchmark",
    "kiswahili": "AI-for-Education/Kiswahili-Linguistic-Knowledge-Benchmark",
}
# The tracked "final" scored experiment per language -- see analyze_results.py/
# validate_results.py's own use of these same experiment-ids elsewhere in scripts/.
DEFAULT_EXPERIMENT_ID = {"english": "final-137-items", "kiswahili": "final-135-items"}
CORRECT_MAP = {True: 1, False: 0, "True": 1, "False": 0}
DIFFICULTY_TIERS = ["hard", "medium", "easy"]

CARD_TAGS = [
    "linguistics",
    "education",
    "benchmark",
    "language-knowledge",
]
# The fixed, language-agnostic D1-D9 spine -- from
# scripts/workflow/pipeline_overview_v10.md ("9 domains (D1-D9): ..."). Order matters:
# this is the D1->D9 order, not alphabetical.
DOMAIN_DEFINITIONS = {
    "D1": "Phonics/Orthography",
    "D2": "Phonology",
    "D3": "Nominal morphology",
    "D4": "Verbal morphology",
    "D5": "Syntax",
    "D6": "Semantics/Lexicon",
    "D7": "Pragmatics/Discourse",
    "D8": "Sociolinguistic/Register",
    "D9": "Metalinguistic reasoning",
}
# Grade = the (UK-curriculum-anchored) level content is first introduced at, not the
# question's difficulty -- from scripts/workflow/process_prompts_chat_ready_v10_english.md
# ("G0 = Reception / below Year 1, G1 = Year 1 ... G5 = Year 5, beyond_primary = above
# Year 5"). Non-English-source languages follow the same G0-G5/beyond_primary spine.
GRADE_DEFINITIONS = {
    "G0": "Reception / below Year 1",
    "G1": "Year 1",
    "G2": "Year 2",
    "G3": "Year 3",
    "G4": "Year 4",
    "G5": "Year 5",
    "beyond_primary": "Above Year 5",
}


# --------------------------------------------------------------------------- #
# Step 1: select-subset
# --------------------------------------------------------------------------- #
def load_dataset(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_difficulty(detailed_csv: str) -> dict[object, str] | None:
    """question_id -> difficulty tier ('easy'/'medium'/'hard'), from a judge-bench
    DETAILED csv's <model>_correct columns. None if the file doesn't exist."""
    if not os.path.exists(detailed_csv):
        return None
    df = pd.read_csv(detailed_csv)
    correct_cols = [c for c in df.columns if c.endswith("_correct")]
    if not correct_cols:
        return None
    scored = df[correct_cols].map(lambda v: CORRECT_MAP.get(v, pd.NA))
    accuracy = scored.mean(axis=1, skipna=True)
    tiers = pd.qcut(accuracy, 3, labels=DIFFICULTY_TIERS, duplicates="drop")
    return dict(zip(df["question_id"], tiers.astype(str)))


def largest_remainder_quotas(counts: dict[str, int], total: int) -> dict[str, int]:
    """Apportion `total` slots across `counts` (domain -> item count), proportional
    to each domain's share, summing to exactly `total` (largest-remainder method --
    avoids the under/over-count that naive rounding gives)."""
    grand_total = sum(counts.values())
    raw = {k: total * v / grand_total for k, v in counts.items()}
    quotas = {k: int(v) for k, v in raw.items()}
    remainder = total - sum(quotas.values())
    order = sorted(counts, key=lambda k: raw[k] - quotas[k], reverse=True)
    for k in order[:remainder]:
        quotas[k] += 1
    return quotas


def select_subset(
    items: list[dict],
    difficulty: dict[object, str] | None,
    n_items: int,
    seed: int,
) -> tuple[list[dict], dict[str, int]]:
    """Spread selection across domain (quota'd by share of the full dataset), then
    within each domain's quota, round-robin over whichever (answer_type, difficulty
    tier) combinations that domain actually has -- so a domain's picks don't land on
    a single format/difficulty by chance. Returns (selected items, domain -> quota)."""
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        by_domain[item["domain"]].append(item)

    domain_counts = {d: len(v) for d, v in by_domain.items()}
    quotas = largest_remainder_quotas(domain_counts, n_items)

    rng = random.Random(seed)
    selected: list[dict] = []
    for domain, quota in quotas.items():
        if quota == 0:
            continue
        candidates = list(by_domain[domain])
        rng.shuffle(candidates)

        def strata_key(item: dict) -> tuple:
            tier = (
                difficulty.get(item["question_id"], "unknown")
                if difficulty
                else "unknown"
            )
            return (item["answer_type"], tier)

        buckets: dict[tuple, list[dict]] = defaultdict(list)
        for item in candidates:
            buckets[strata_key(item)].append(item)
        bucket_cycle = list(buckets.values())
        picked: list[dict] = []
        i = 0
        while len(picked) < quota and any(bucket_cycle):
            bucket = bucket_cycle[i % len(bucket_cycle)]
            if bucket:
                picked.append(bucket.pop())
            i += 1
        selected.extend(picked)

    selected.sort(key=lambda i: i["question_id"])
    return selected, quotas


def _breakdown(items: list[dict], difficulty: dict[object, str] | None) -> pd.DataFrame:
    rows = [
        {
            "domain": i["domain"],
            "answer_type": i["answer_type"],
            "difficulty": (
                difficulty.get(i["question_id"], "unknown") if difficulty else "n/a"
            ),
        }
        for i in items
    ]
    return pd.DataFrame(rows)


def print_selection_report(
    full_items: list[dict],
    selected: list[dict],
    difficulty: dict[object, str] | None,
    quotas: dict[str, int],
) -> None:
    full_df = _breakdown(full_items, difficulty)
    sel_df = _breakdown(selected, difficulty)

    print("\nDomain quotas (selected / full):")
    for domain in sorted(quotas, key=lambda d: -quotas[d]):
        full_n = int((full_df["domain"] == domain).sum())
        print(f"  {domain}: {quotas[domain]} / {full_n}")
    zero = [d for d, q in quotas.items() if q == 0]
    if zero:
        print(
            f"  (0 slots -- too small a share of {len(selected)} items: {', '.join(zero)})"
        )

    print("\nanswer_type (selected / full):")
    for atype, count in sel_df["answer_type"].value_counts().items():
        full_n = int((full_df["answer_type"] == atype).sum())
        print(f"  {atype}: {count} / {full_n}")

    if difficulty is not None:
        print("\ndifficulty tier (selected / full):")
        for tier in DIFFICULTY_TIERS:
            count = int((sel_df["difficulty"] == tier).sum())
            full_n = int((full_df["difficulty"] == tier).sum())
            print(f"  {tier}: {count} / {full_n}")
    else:
        print(
            "\nNo detailed CSV found -- difficulty tiers skipped (see --detailed-csv)."
        )


def cmd_select_subset(args: argparse.Namespace, root: str) -> None:
    outdir = args.outdir or os.path.join(root, "data")
    dataset_path = os.path.join(
        root, "data", f"linguistic_knowledge_eval_{args.language}.json"
    )
    items = load_dataset(dataset_path)

    detailed_csv = args.detailed_csv or os.path.join(
        root,
        "results",
        f"summary_linguistic-knowledge_{args.language}_"
        f"{DEFAULT_EXPERIMENT_ID[args.language]}_fallback_detailed.csv",
    )
    difficulty = load_difficulty(detailed_csv)
    if difficulty is None:
        print(f"NOTE: no detailed CSV at {detailed_csv} -- skipping difficulty tiers.")

    selected, quotas = select_subset(items, difficulty, args.n_items, args.seed)

    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"hf_subset_{args.language}_{args.n_items}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(selected, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Selected {len(selected)}/{len(items)} items from {dataset_path}")
    print_selection_report(items, selected, difficulty, quotas)
    print(f"\nWrote {out_path}")


# --------------------------------------------------------------------------- #
# Step 2: push
# --------------------------------------------------------------------------- #
def _definitions_table() -> str:
    """Markdown table with the fixed D1-D9 domain spine (DOMAIN_DEFINITIONS) and the
    fixed G0-G5/beyond_primary grade spine (GRADE_DEFINITIONS) side by side, separated
    by a blank spacer column -- both static, not derived from any particular subset
    (the two spines are the same regardless of which items got selected)."""
    domain_rows = list(DOMAIN_DEFINITIONS.items())
    grade_rows = list(GRADE_DEFINITIONS.items())
    lines = ["| Domain | Covers | | Grade | Content level |", "|---|---|---|---|---|"]
    for i in range(max(len(domain_rows), len(grade_rows))):
        d_code, d_name = domain_rows[i] if i < len(domain_rows) else ("", "")
        g_code, g_level = grade_rows[i] if i < len(grade_rows) else ("", "")
        g_code_fmt = f"`{g_code}`" if g_code else ""
        lines.append(f"| {d_code} | {d_name} | | {g_code_fmt} | {g_level} |")
    return "\n".join(lines)


def _answer_type_counts(items: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for item in items:
        counts[item["answer_type"]] += 1
    return dict(counts)


def _card_content(language: str, records: list[dict], full_items: list[dict]) -> str:
    """The dataset card's raw YAML-frontmatter + Markdown text. Deliberately has no
    huggingface_hub dependency, unlike _build_card() below, so the card can be
    generated and reviewed (see cmd_write_card) without installing it."""
    import yaml

    n_items = len(records)
    lang_display = "English" if language == "english" else "Kiswahili"
    english_note = (
        f" Questions are written in English; only the linguistic *subject* of the "
        f"questions is {lang_display}."
        if language != "english"
        else ""
    )
    n_full = len(full_items)
    full_counts = _answer_type_counts(full_items)
    metadata: dict = {
        "language": ["en"],
        "size_categories": ["n<1K"],
        "tags": CARD_TAGS,
        "pretty_name": f"{lang_display} Linguistic Knowledge Benchmark",
    }
    body = f"""# {lang_display} Linguistic Knowledge Benchmark

A {n_items}-item public sample of the {lang_display} Linguistic Knowledge Benchmark,
measuring how much an LLM *knows about the linguistics of {lang_display}* (phonics/
orthography, morphology, syntax, vocabulary, and phonological awareness),
rather than its ability to converse in the language.{english_note}

This is a curated sample, not the full item set: the full {n_full}-item benchmark
({full_counts.get("multiple_choice", 0)} multiple_choice, {full_counts.get("short_form", 0)} short_form)
is kept private to avoid leaking into future models' training data. The full dataset
is available on request. Items here were selected to spread across question domain,
answer format, and difficulty.

## Domain and grade definitions

Domain (fixed, language-agnostic -- every language's items are filed against the same
9 domains) and grade (the content level a fact is *first introduced* at, not the
question's difficulty) definitions, side by side:

{_definitions_table()}

## Columns

| Column | Description |
|---|---|
| `question_id` | Question identifier in the full benchmark |
| `domain` | Domain code (`D1`-`D9`) -- see Domain and grade definitions above |
| `subdomain` | Free-text sub-topic within the domain |
| `grade` | Grade code (`G0`-`G5`, or `beyond_primary`) -- see Domain and grade definitions above |
| `answer_type` | `multiple_choice` (answer is a single letter A-D) or `short_form` (1-3 words) |
| `question` | Question text (MCQ options are a separate `options` field) |
| `options` | `{{"A": ..., "B": ..., "C": ..., "D": ...}}`, present for `multiple_choice` only |
| `expected_answers` | List of acceptable answers (short_form can have more than one) |
| `linguistic_feature` | The specific linguistic point the question targets |
| `svr_strand` | Simple View of Reading strand the item serves: `decoding`, `language_comprehension`, or `outside_flr_construct` |
| `geeap_component` | GEEAP core instructional skill an error in this item would corrupt: `systematic_phonics`, `phonological_awareness`, `oral_language`, `reading_fluency`, `reading_comprehension`, `writing`, or `none` |
| `sources` | Source materials the question/answer is grounded in |
| `grade_placement_evidence` | How the `grade` assignment was justified in the pipeline |

## Source & methodology

Built by [The Quality Assurance Facility for AI in Education](https://www.fab-ai.org/initiatives/ai-for-education/edtech-quality/quality-assurance).

"""
    yaml_str = yaml.safe_dump(metadata, sort_keys=False, default_flow_style=False)
    return f"---\n{yaml_str}---\n{body}"


def _build_card(language: str, records: list[dict], full_items: list[dict]):
    """The pushable DatasetCard object -- thin huggingface_hub wrapper around
    _card_content()'s text. Only needed for the real (--yes) push."""
    from huggingface_hub import DatasetCard

    return DatasetCard(_card_content(language, records, full_items))


def cmd_write_card(args: argparse.Namespace, root: str) -> None:
    """Write the dataset card to a local .md file for review -- no huggingface_hub
    dependency needed, unlike `push` (which needs it only once you pass --yes)."""
    subset_path = args.subset_json or os.path.join(
        root, "data", f"hf_subset_{args.language}_{args.n_items}.json"
    )
    if not os.path.exists(subset_path):
        raise SystemExit(
            f"{subset_path} not found -- run 'select-subset --language {args.language}' "
            "first (or pass --subset-json)."
        )
    with open(subset_path, encoding="utf-8") as f:
        records = json.load(f)
    full_items = load_dataset(
        os.path.join(root, "data", f"linguistic_knowledge_eval_{args.language}.json")
    )

    outdir = args.outdir or os.path.join(root, "data")
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, f"hf_card_{args.language}.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(_card_content(args.language, records, full_items))
    print(f"Wrote {out_path} ({len(records)} items)")


def cmd_push(args: argparse.Namespace, root: str) -> None:
    repo_id = REPO_IDS[args.language]
    subset_path = args.subset_json or os.path.join(
        root, "data", f"hf_subset_{args.language}_{args.n_items}.json"
    )
    if not os.path.exists(subset_path):
        raise SystemExit(
            f"{subset_path} not found -- run 'select-subset --language {args.language}' "
            "first (or pass --subset-json)."
        )

    with open(subset_path, encoding="utf-8") as f:
        records = json.load(f)
    full_items = load_dataset(
        os.path.join(root, "data", f"linguistic_knowledge_eval_{args.language}.json")
    )
    columns = sorted(records[0].keys()) if records else []
    print(
        f"Loaded {len(records)} rows from {subset_path} (columns: {columns}) "
        f"-> {repo_id}"
    )

    if not args.yes:
        print(
            f"[DRY-RUN] would push data to {repo_id} as a PRIVATE dataset "
            "(pass --yes to actually push)"
        )
        if not args.no_card:
            content = _card_content(args.language, records, full_items)
            print(f"[DRY-RUN] would push dataset card ({len(content)} chars)")
        return

    # Only imported here -- neither is needed for select-subset, write-card, or a
    # push dry-run, so a fresh checkout doesn't need them installed for those.
    from datasets import Dataset
    from huggingface_hub import HfApi

    ds = Dataset.from_list(records)
    # Always private -- see the module docstring's "Privacy" note. This script never
    # makes anything public; going public is a manual, separate step on the Hub.
    HfApi(token=args.token).create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=True,
        exist_ok=True,
    )
    ds.push_to_hub(repo_id=repo_id, split="train", token=args.token)
    print(f"[OK] pushed data to {repo_id}")

    if not args.no_card:
        card = _build_card(args.language, records, full_items)
        card.push_to_hub(repo_id, token=args.token)
        print(f"[OK] pushed dataset card to {repo_id}")


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    select = sub.add_parser("select-subset", help="Pick a spread-out N-item subset.")
    select.add_argument("--language", required=True, choices=LANGUAGES)
    select.add_argument("--n-items", type=int, default=DEFAULT_N_ITEMS)
    select.add_argument("--seed", type=int, default=DEFAULT_SEED)
    select.add_argument(
        "--detailed-csv",
        default=None,
        help="Judge-bench DETAILED csv for difficulty tiering. Defaults to the "
        "fallback-pass detailed CSV for this language's tracked 'final' experiment.",
    )
    select.add_argument("--outdir", default=None)

    card = sub.add_parser(
        "write-card",
        help="Write the dataset card to a local .md file for review (no HF deps "
        "needed -- see 'push', which pushes this same card for real).",
    )
    card.add_argument("--language", required=True, choices=LANGUAGES)
    card.add_argument("--n-items", type=int, default=DEFAULT_N_ITEMS)
    card.add_argument(
        "--subset-json",
        default=None,
        help="Defaults to data/hf_subset_<language>_<n-items>.json (select-subset's "
        "output) -- only its item count is used (for the card's item-count line).",
    )
    card.add_argument("--outdir", default=None)

    push = sub.add_parser("push", help="Push a subset JSON to its own HF Hub repo.")
    push.add_argument("--language", required=True, choices=LANGUAGES)
    push.add_argument("--n-items", type=int, default=DEFAULT_N_ITEMS)
    push.add_argument(
        "--subset-json",
        default=None,
        help="Defaults to data/hf_subset_<language>_<n-items>.json (select-subset's "
        "output).",
    )
    push.add_argument(
        "--yes",
        action="store_true",
        help="Actually push to the Hub. Without this, push always dry-runs (prints "
        "what it would do, touches nothing on the Hub) -- this is the safe default, "
        "not opt-in dry-running.",
    )
    push.add_argument("--no-card", action="store_true")
    push.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY"),
        help="Defaults to env var HF_TOKEN or HUGGINGFACE_API_KEY, or the cached "
        "token from `huggingface-cli login`.",
    )

    return ap


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    # Load the repo-root .env (HF_TOKEN/HUGGINGFACE_API_KEY) BEFORE build_parser(),
    # since --token's default is captured once, at add_argument() time.
    load_dotenv(os.path.join(root, "..", "..", ".env"))

    ap = build_parser()
    args = ap.parse_args()
    if args.cmd == "select-subset":
        cmd_select_subset(args, root)
    elif args.cmd == "write-card":
        cmd_write_card(args, root)
    elif args.cmd == "push":
        cmd_push(args, root)


if __name__ == "__main__":
    main()

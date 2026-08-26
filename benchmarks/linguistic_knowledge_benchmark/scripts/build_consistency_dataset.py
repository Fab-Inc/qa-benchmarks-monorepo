#!/usr/bin/env python3
"""
Build a repeated-question dataset for the "consistency" validation check: N copies of
every question in a language's eval dataset, grouped by item (all N copies of item 1,
then all N copies of item 2, ...) -- not N full passes of the whole dataset back to
back. Running run-bench/judge-bench against this file gives N independent samples per
question, in one single run/cache folder -- the raw material for measuring run-to-run
accuracy variance (see validate_results.py's "consistency" check, which reads the
resulting summary/detailed/judge_agreement CSVs).

question_id encoding: "<original_id>.<replicate>", replicate ZERO-PADDED to at least 2
digits ("1.01" .. "1.20", not "1.1" .. "1.20"). This isn't cosmetic -- it's required for
correctness. responses_<model>.csv round-trips through pandas.read_csv() inside
judge_bench.judge_results(), which auto-infers a numeric-looking string column as
float64. Floats drop trailing zeros, so unpadded "1.10" and "1.1" both become the SAME
float (1.1) -- replicate 10 would silently overwrite/merge with replicate 1, and
replicate 20 with replicate 2. Zero-padding (so every replicate suffix has the same
digit width) makes every (item, replicate) pair a numerically distinct float, so the
encoding survives that round-trip intact. validate_results.py's consistency check
decodes it back with the matching --n-replicates value (int(question_id) for the
original item id, round((question_id - int(question_id)) * 10**width) for the
replicate).

Scoring against this file: `linguistic-knowledge-benchmark run-bench`/`judge-bench` both
take a --dataset override (added specifically for this workflow) -- no need to touch
data/linguistic_knowledge_eval_<language>.json at all. --dataset is FILENAME ONLY, no
path (it's always resolved as DATA_DIR/<dataset>, i.e. the file must live directly in
data/ -- which is exactly where this script writes its output, so just pass the bare
filename this script printed). Pass the SAME --dataset to both commands, since
judge-bench's own "detailed" CSV is built by merging judged results onto whatever
--dataset it was given; passing it consistently means the detailed CSV comes out
correct on the first try, no separate rebuild step needed:
    uv run linguistic-knowledge-benchmark run-bench --language english --run-id <run-id> \\
        --dataset linguistic_knowledge_eval_english_20x.json
    uv run linguistic-knowledge-benchmark judge-bench --language english --run-id <run-id> \\
        --dataset linguistic_knowledge_eval_english_20x.json --num-judges 3
Then validate_results.py's "consistency" check reads the resulting
summary/detailed/judge_agreement CSVs for that --run-id directly.

Run (build the repeated dataset itself -- a separate, one-off step from the above):
    uv run python benchmarks/linguistic_knowledge_benchmark/scripts/build_consistency_dataset.py --language english
    uv run python .../build_consistency_dataset.py --language kiswahili --n-replicates 20
"""

from __future__ import annotations

import argparse
import json
import os

from analyze_results import LANGUAGES

DEFAULT_N_REPLICATES = 20


def build_repeated_dataset(items: list[dict], n_replicates: int) -> list[dict]:
    """N copies of every item, grouped by item (item1 x N, item2 x N, ...). Each copy's
    question_id becomes "<original_id>.<replicate>", replicate zero-padded -- see the
    module docstring for why the padding is required, not cosmetic."""
    width = max(len(str(n_replicates)), 2)
    out = []
    for item in items:
        original_id = item["question_id"]
        for rep in range(1, n_replicates + 1):
            copy = dict(item)
            copy["question_id"] = f"{original_id}.{rep:0{width}d}"
            out.append(copy)
    return out


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--language",
        required=True,
        choices=LANGUAGES,
        help="which language's data/linguistic_knowledge_eval_<language>.json to repeat",
    )
    ap.add_argument(
        "--n-replicates",
        type=int,
        default=DEFAULT_N_REPLICATES,
        help=f"copies per question (default {DEFAULT_N_REPLICATES})",
    )
    args = ap.parse_args()

    out_path = os.path.join(
        root,
        "data",
        f"linguistic_knowledge_eval_{args.language}_{args.n_replicates}x.json",
    )
    if os.path.exists(out_path):
        raise SystemExit(
            f"refusing to overwrite existing file: {out_path}\n"
            "Delete it first, or use a different --n-replicates, if you want to "
            "rebuild it."
        )

    src_path = os.path.join(
        root, "data", f"linguistic_knowledge_eval_{args.language}.json"
    )
    with open(src_path, encoding="utf-8") as f:
        items = json.load(f)

    repeated = build_repeated_dataset(items, args.n_replicates)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(repeated, f, ensure_ascii=False, indent=2)
        f.write("\n")

    out_filename = os.path.basename(out_path)
    print(f"{len(items)} items x {args.n_replicates} replicates = {len(repeated)} rows")
    print(f"Source: {src_path}")
    print(f"Wrote:  {out_path}")
    print(
        "\nNext: pass this file's NAME (not a path -- --dataset is always resolved "
        "relative to data/) as --dataset to BOTH run-bench and judge-bench (same "
        "--run-id for both), e.g.:\n"
        f"  uv run linguistic-knowledge-benchmark run-bench --language {args.language} "
        f"--run-id <run-id> --dataset {out_filename}\n"
        f"  uv run linguistic-knowledge-benchmark judge-bench --language {args.language} "
        f"--run-id <run-id> --dataset {out_filename} --num-judges 3"
    )


if __name__ == "__main__":
    main()

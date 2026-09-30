#!/usr/bin/env python3
"""
Turn a phonics run's per-model summaries into this benchmark's records files.

This is the phonics half of the leaderboard split: it writes one
results/web/leaderboard_records_<domain>.json per pair and stops there. Assembling the
five website JSON files is `fab-leaderboard publish`, which reads every benchmark's
records files at once -- the bundle spans all benchmark x language pairs, so no single
benchmark can build it alone.

Every pair in SOURCES below is published by default, since a run covering two languages
should not need two commands to reach the leaderboard. A pair whose summary CSV is not
in the run is skipped with a note; naming a pair with --pair makes its absence an error
instead, because then you asked for it specifically.

Run from the repo root; the README's "Publishing to the leaderboard website" section
carries these commands in full, ready to paste.

    # every pair the run reported (DA-en and DA-sw today)
    ... results_for_website.py --run-id run-test

    # just one
    ... results_for_website.py --run-id run-test --pair DA-sw

    # a summary CSV somewhere else; --pair says which column it fills
    ... results_for_website.py --summary-csv /path/to/summary.csv --pair DA-en

Then the bundle, with one --records-dir per benchmark:

    uv run fab-leaderboard publish --records-dir DIR --outdir results/web

INPUT: one summary CSV per pair, named in SOURCES and resolved under
outputs/<run-id>/. Each name carries its pair, which is what lets one run report
several side by side; a plain summary_by_model.csv would say nothing about which
leaderboard column it belongs in, and publishing the wrong pair's scores under DA-en is
not a mistake anything downstream could catch. Only three columns are read:

  model      the model id, which must match a model_id in models.csv or the model is
             dropped from the leaderboard with a warning
  score      the headline number, already 0-100 -- see the note on scale below
  n_items    how many items that model was scored on; varies per model, so it is read
             per row rather than assumed constant

The report also carries display_name, provider and color. Those are deliberately NOT
read: models.csv is the single source of model metadata for every benchmark on this
leaderboard, and taking one benchmark's copy would let the two disagree.

ON THE SCORE. The leaderboard shows one number per model per pair on a shared axis,
and for phonics that number is `score`, which the report computes as an adjusted
balanced accuracy -- it equals the adjusted_balanced_accuracy column exactly. It is
NOT the same measurement as the linguistic-knowledge accuracies it sits beside, which
are plain proportions correct. That is a deliberate choice about what the leaderboard
compares, worth restating wherever the figure appears.

ON THE SCALE. `score` arrives on the 0-100 scale the leaderboard needs, so nothing is
rescaled here. check_is_percentage below still guards it, because a 0-1 column would
pass the record contract's 0-100 validation untouched and put every phonics model
under 1% on the site with no error anywhere -- a silent regression this catches loudly
if the report step ever changes scale again.

NOT REPORTED: bad_format. The linguistic-knowledge benchmark derives it from answers
no judge could score; phonics has no equivalent, so the field is left unset here.
Note it does NOT stay unset downstream: the website's payload needs a number, so the
bundle publishes 0.0. This column's bad-format figure therefore reads "perfect" on the
site when it means "not measured" -- say so wherever the comparison is presented.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from fab_benchmark_publisher import BenchmarkPair, LeaderboardRecord
from fab_benchmark_publisher.leaderboard import write_records_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# One row per leaderboard column this benchmark owns.
#
# benchmark_domain must match a row in the repo-root leaderboard_pairs.json, which is
# what decides the column order on the site; display_name only has to not contradict
# it. summary_csv is relative to the run folder (outputs/<run-id>/) and is named
# explicitly rather than derived, so adding a language is one visible line here and
# there is never a question about which file feeds which column.
SOURCES: list[dict] = [
    {
        "pair": BenchmarkPair(benchmark_domain="DA-en", display_name="DA English"),
        "summary_csv": Path("report") / "da-en_summary_by_model.csv",
    },
    {
        "pair": BenchmarkPair(benchmark_domain="DA-sw", display_name="DA Kiswahili"),
        "summary_csv": Path("report") / "da-sw_summary_by_model.csv",
    },
]

MODEL_COLUMN = "model"
ACCURACY_COLUMN = "score"
N_COLUMN = "n_items"


def source_for(benchmark_domain: str) -> dict:
    """The SOURCES row for one pair, or abort listing the pairs this script owns."""
    for source in SOURCES:
        if source["pair"].benchmark_domain == benchmark_domain:
            return source
    owned = ", ".join(s["pair"].benchmark_domain for s in SOURCES)
    raise SystemExit(
        f"ERROR: '{benchmark_domain}' is not a pair this benchmark publishes ({owned})"
    )


def records_for(summary_csv: Path, benchmark_domain: str) -> list[LeaderboardRecord]:
    """One <domain>_summary_by_model.csv -> LeaderboardRecords for one pair."""
    records: list[LeaderboardRecord] = []
    with open(summary_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for column in (MODEL_COLUMN, ACCURACY_COLUMN, N_COLUMN):
            if column not in (reader.fieldnames or []):
                raise SystemExit(
                    f"ERROR: {summary_csv} has no '{column}' column.\n"
                    f"       found: {', '.join(reader.fieldnames or [])}"
                )
        for row in reader:
            model = (row.get(MODEL_COLUMN) or "").strip()
            score = row.get(ACCURACY_COLUMN)
            if not model or score in (None, ""):
                continue
            records.append(
                LeaderboardRecord(
                    benchmark_domain=benchmark_domain,
                    model=model,
                    accuracy=round(float(score), 2),
                    n=int(float(row[N_COLUMN])) if row.get(N_COLUMN) else None,
                )
            )
    check_is_percentage(records, summary_csv)
    return records


def check_is_percentage(records: list[LeaderboardRecord], summary_csv: Path) -> None:
    """Refuse a score column that still looks like it is on a 0-1 scale.

    0-1 values are inside the 0-100 range the record contract validates, so without
    this the whole benchmark would publish at under 1% and look like a real -- very
    bad -- result rather than a unit mistake.
    """
    if len(records) < 2:
        return
    highest = max(r.accuracy for r in records)
    if highest <= 1.0:
        raise SystemExit(
            f"ERROR: every {ACCURACY_COLUMN} in {summary_csv} is <= 1.0 (max "
            f"{highest:g}), so the column looks like it is on a 0-1 scale.\n"
            f"       The leaderboard needs 0-100: the report step should write "
            f"{ACCURACY_COLUMN} already scaled."
        )


def resolve_sources(args: argparse.Namespace) -> list[tuple[BenchmarkPair, Path, str]]:
    """(pair, summary CSV, source label) for every pair to publish this run.

    A pair the run did not cover is skipped with a note UNLESS --pair named it: an
    explicit request for a missing file is an error, while publishing whatever a run
    happens to contain is the normal case.
    """
    if args.summary_csv:
        pair = source_for(args.pair[0])["pair"]
        path = Path(args.summary_csv)
        if not path.is_file():
            raise SystemExit(f"ERROR: summary CSV not found: {path}")
        return [(pair, path, path.name)]

    wanted = args.pair or [s["pair"].benchmark_domain for s in SOURCES]
    run_dir = Path(args.outputs_dir) / args.run_id
    resolved: list[tuple[BenchmarkPair, Path, str]] = []
    for domain in wanted:
        source = source_for(domain)
        path = run_dir / source["summary_csv"]
        if not path.is_file():
            if args.pair:
                raise SystemExit(f"ERROR: summary CSV for {domain} not found: {path}")
            print(f"  {domain}: not in this run, skipped ({path.name})")
            continue
        resolved.append((source["pair"], path, args.run_id))
    if not resolved:
        raise SystemExit(
            f"ERROR: no summary CSV found under {run_dir}. Expected any of:\n"
            + "\n".join(f"         {s['summary_csv'].as_posix()}" for s in SOURCES)
        )
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-id", help="run to publish, i.e. outputs/<run-id>/")
    source.add_argument(
        "--summary-csv",
        help="exact summary CSV instead of a run folder; needs --pair to say which "
        "column it fills",
    )
    parser.add_argument(
        "--pair",
        action="append",
        metavar="DOMAIN",
        help="publish only this pair (repeatable). Default: every pair the run "
        f"reported ({', '.join(s['pair'].benchmark_domain for s in SOURCES)})",
    )
    parser.add_argument(
        "--outputs-dir",
        default=str(ROOT / "outputs"),
        help="where run folders live (default: this benchmark's outputs/)",
    )
    parser.add_argument(
        "--outdir",
        default=str(ROOT / "results" / "web"),
        help="where to write the records files (default: this benchmark's results/web)",
    )
    args = parser.parse_args()

    if args.summary_csv and len(args.pair or []) != 1:
        raise SystemExit(
            "ERROR: --summary-csv needs exactly one --pair, since a file path alone "
            "does not say which leaderboard column it fills"
        )

    print("Reading summaries:")
    written = []
    for pair, summary_csv, source_label in resolve_sources(args):
        records = records_for(summary_csv, pair.benchmark_domain)
        if not records:
            raise SystemExit(f"ERROR: no scored rows in {summary_csv}")
        path = write_records_file(
            records, benchmark=pair, outdir=args.outdir, source=source_label
        )
        scores = [r.accuracy for r in records]
        print(
            f"  {pair.benchmark_domain}: {summary_csv.name} -> {path.name}"
            f"  ({len(records)} models, {ACCURACY_COLUMN} "
            f"{min(scores):.2f}-{max(scores):.2f})"
        )
        written.append(path)

    print(f"\nWrote {len(written)} records file(s) to {Path(args.outdir)}")
    print("Next, from the repo root:")
    print(
        "  uv run fab-leaderboard publish "
        f"--records-dir {Path(args.outdir).as_posix()} --records-dir <other benchmarks>"
    )


if __name__ == "__main__":
    main()

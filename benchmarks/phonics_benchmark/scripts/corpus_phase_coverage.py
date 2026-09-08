"""Report how much of a reference corpus each curriculum phase can decode.

This regenerates the coverage tables quoted in ``README.md`` and
``data/README.md``. Those numbers justify the curriculum-phase buckets and the
consonant-cluster caveat, so they need to be re-derivable rather than
hand-computed — re-run this whenever the profiles or the corpus change and paste
the output back into the docs.

The corpus is a *reference* only: it is not an input to the benchmark, which asks
the model to write its own text. It exists so "how much of real Kiswahili can
this student read?" has a concrete answer.

Usage:

    uv run --package phonics-benchmark python \\
        benchmarks/phonics_benchmark/scripts/corpus_phase_coverage.py \\
        --language kiswahili
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import typer
from decodability import DECODABILITY_REGISTRY, annotate_words
from phonics_benchmark.curation import CURRICULUM_PHASES, get_curriculum_phase
from phonics_benchmark.data_io import get_data_dir, read_student_knowledge_profiles
from phonics_benchmark.labeling import (
    get_decodability_config,
    resolve_decodability_config,
)

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
DEFAULT_CORPUS_FILENAME = "decodable_texts.jsonl"

app = typer.Typer(add_completion=False)


def read_corpus_vocabulary(path: Path, language_iso_code: str) -> list[str]:
    """Return the unique lowercased words of a JSONL corpus of texts."""
    objs = DECODABILITY_REGISTRY[language_iso_code]
    words: set[str] = set()
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON — {exc}") from exc
        words.update(word.lower() for word in objs["extract_words"](record["content"]))
    if not words:
        raise ValueError(f"No words extracted from {path}")
    return sorted(words)


def decodable_share(
    student_knowledge: dict, vocabulary: list[str], language_iso_code: str
) -> float:
    """Return the percentage of ``vocabulary`` this student can decode."""
    objs = DECODABILITY_REGISTRY[language_iso_code]
    config, score_column = get_decodability_config(language_iso_code)
    scoring_methods, aggregations = resolve_decodability_config(
        config, language_iso_code
    )
    scores = annotate_words(
        vocabulary,
        scoring_methods=scoring_methods,
        student_knowledge=objs["student_knowledge_model"].model_validate(
            student_knowledge
        ),
        aggregations=aggregations,
    )
    return 100.0 * (scores[score_column] == 1.0).sum() / len(vocabulary)


@app.command()
def main(
    language: str = typer.Option(
        "kiswahili",
        "--language",
        "-l",
        help="Language folder name under the data root, for example 'kiswahili'.",
    ),
    corpus_filename: str = typer.Option(
        DEFAULT_CORPUS_FILENAME,
        "--corpus-filename",
        help="Reference corpus JSONL inside the language data folder.",
    ),
    data_root: Path = typer.Option(
        DEFAULT_DATA_ROOT, "--data-root", file_okay=False, help="Root data directory."
    ),
) -> None:
    """Print the per-phase corpus-coverage table as markdown."""
    profiles = read_student_knowledge_profiles(language, data_root)
    iso = profiles[0].language_iso_code
    corpus_path = get_data_dir(language, data_root) / corpus_filename
    vocabulary = read_corpus_vocabulary(corpus_path, iso)

    typer.echo(
        f"Reference corpus: {corpus_path.name} "
        f"({len(vocabulary)} unique words)\n"
        f"Profiles: {len(profiles)}\n"
    )

    shares = {
        profile.student_knowledge_id: decodable_share(
            profile.student_knowledge, vocabulary, iso
        )
        for profile in profiles
    }
    taught_clusters = {
        profile.student_knowledge_id: bool(
            profile.student_knowledge.get("clusters")
            or profile.student_knowledge.get("cluster_patterns")
        )
        for profile in profiles
    }

    typer.echo(
        "| Phase | Graphemes | Profiles | Taught clusters "
        "| Corpus decodable (min / median / max) |"
    )
    typer.echo("| --- | --- | --- | --- | --- |")
    lower = 0
    for phase, upper in CURRICULUM_PHASES:
        members = [
            p
            for p in profiles
            if get_curriculum_phase(len(p.student_knowledge["graphemes"])) == phase
        ]
        if not members:
            continue
        values = [shares[p.student_knowledge_id] for p in members]
        n_clusters = sum(taught_clusters[p.student_knowledge_id] for p in members)
        if upper is None:
            span = f"{lower}+"
        elif lower == 0:
            span = f"≤{upper}"  # a lower bound of 0 is noise for the first phase
        else:
            span = f"{lower}–{upper}"
        typer.echo(
            f"| `{phase}` | {span} | {len(members)} | {n_clusters} "
            f"| {min(values):.1f}% / {statistics.median(values):.1f}% "
            f"/ {max(values):.1f}% |"
        )
        lower = (upper or 0) + 1

    typer.echo(
        f"\nProfiles taught any consonant cluster: "
        f"{sum(taught_clusters.values())}/{len(profiles)}"
    )
    typer.echo("\nMost- and least-served profiles:")
    ranked = sorted(shares.items(), key=lambda kv: kv[1])
    for sid, share in ranked[:2] + ranked[-2:]:
        typer.echo(f"  {sid:24} {share:5.1f}%")


if __name__ == "__main__":
    app()

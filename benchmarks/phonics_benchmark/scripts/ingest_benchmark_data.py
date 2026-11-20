r"""Ingest a language benchmark data-entry workbook into JSONL data files.

Run from the repository root with (as an example):

    uv run --package phonics-benchmark python \\
        benchmarks/phonics_benchmark/scripts/ingest_benchmark_data.py \\
        --language kiswahili \\
        --workbook path/to/benchmark_template.xlsx

The workbook is a template filled in by language experts, with a "Student
Profiles" tab and a "Texts" tab. Machine-read columns are identified by exact
header name (case-sensitive), independent of column order, so the template
can carry additional expert-context columns without breaking this script.
Rows with a blank ID column are treated as spacing/notes and skipped.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import typer
from decodability.kiswahili.definitions import ClusterPattern
from phonics_benchmark.data_io import validate_student_knowledge, write_dataset
from phonics_benchmark.models import DecodableText, StudentKnowledge
from phonics_benchmark.utils import require_single_language_iso_code

DEFAULT_DATA_ROOT = Path(__file__).resolve().parents[1] / "data"

STUDENT_PROFILES_SHEET = "Student Profiles"
TEXTS_SHEET = "Texts"

# The profile_type cell classifies the profile, not the student's knowledge, so
# it lands on StudentKnowledge itself rather than inside student_knowledge.
PROFILE_TYPE_COLUMN = "profile_type"
CURRICULUM_ALIGNED_PROFILE = "curriculum-aligned"

NONE_SENTINELS = {"none", "n/a", "na", "-"}
WORKBOOK_CLUSTER_PATTERNS: dict[str, ClusterPattern] = {
    "m_consonant": ClusterPattern.PRENASALISED_M,
    "n_consonant": ClusterPattern.PRENASALISED_N,
    "consonant_w": ClusterPattern.LABIALISED_GLIDE,
    "consonant_y": ClusterPattern.PALATALISED_GLIDE,
    "m_consonant_w": ClusterPattern.M_LABIALISED_GLIDE,
    "n_consonant_w": ClusterPattern.N_LABIALISED_GLIDE,
    "m_consonant_y": ClusterPattern.M_PALATALISED_GLIDE,
    "m/n + c": ClusterPattern.PRENASALISED,
    "n + c": ClusterPattern.PRENASALISED,
    "c + w": ClusterPattern.LABIALISED_GLIDE,
    "c + y": ClusterPattern.PALATALISED_GLIDE,
    "m/n + c + w": ClusterPattern.PRENASALISED_LABIALISED_GLIDE,
    "m/n + c + y": ClusterPattern.PRENASALISED_PALATALISED_GLIDE,
}


def split_list(value: str) -> list[str]:
    """Split a comma-separated cell value into a stripped, non-empty list."""
    if not value or value.strip().casefold() in NONE_SENTINELS:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_cluster_patterns(value: str) -> list[str]:
    """Map workbook cluster-pattern shorthand onto ClusterPattern codes."""
    codes: list[str] = []

    for token in split_list(value):
        pattern = WORKBOOK_CLUSTER_PATTERNS.get(token.casefold())
        if pattern is None:
            raise ValueError(
                f"Unrecognised cluster pattern '{token}'. Expected one of: "
                f"{', '.join(sorted(WORKBOOK_CLUSTER_PATTERNS))}"
            )

        codes.append(pattern.value)

    return list(dict.fromkeys(codes))


# Positional markers sit at the edge of a grapheme and are written with either a
# hyphen (-ing, pre-) or an underscore (_ce, _ge); both are stripped. A medial
# underscore instead marks a split digraph (a_e, o_e) and becomes the hyphen the
# inventory uses (a-e, o-e). Order matters: strip the edges before rewriting, or
# a leading marker survives as a hyphen the inventory has no entry for.
_POSITIONAL_MARKERS = "-_"

# The inventory and ARPABET_TO_IPA in decodability.english.definitions spell the
# voiced velar stop with ASCII "g" (U+0067), but the workbook may carry the
# typographically correct IPA script g (U+0261). Fold it so parsed pairs match.
_PHONEME_TRANSLATION = str.maketrans({"ɡ": "g"})


def extract_gpc_pairs(value: str) -> list[tuple[str, str]]:
    """Parse comma-separated ``grapheme /phoneme/`` pairs from a cell value.
    Apply grapheme and phoneme translations to normalize the input.

    Example: ``"a /æ/, qu /kw/"`` -> ``[("a", "æ"), ("qu", "kw")]``.

    Exact duplicate pairs are dropped, keeping first-seen order. Note that one
    grapheme may legitimately appear more than once with *different* phonemes
    (``th`` -> ``ð`` and ``th`` -> ``θ``); only whole-pair repeats collapse.
    """
    pairs = []
    for item in split_list(value):
        grapheme, _, phoneme = item.partition("/")
        grapheme = grapheme.strip().strip(_POSITIONAL_MARKERS).replace("_", "-")
        phoneme = phoneme.strip().strip("/").translate(_PHONEME_TRANSLATION)
        pairs.append((grapheme, phoneme))
    return list(dict.fromkeys(pairs))


def extract_affixes(value: str) -> list[str]:
    """Collect marker-bearing graphemes (affixes and positional graphemes).

    Reads the same ``grapheme /phoneme/`` cell as ``extract_gpc_pairs`` but
    keeps the grapheme exactly as written, edge marker included, for items
    where the marker sits at the start or end (``-ed``, ``pre-``, ``_ce``,
    ``-le``). A medial underscore (``a_e``) marks a split digraph, not an
    affix, so it is skipped. Duplicates are dropped, keeping first-seen order.

    Example: ``"ing /ɪŋ/, -ed /d/, _ce /s/, a_e /eɪ/"`` -> ``["-ed", "_ce"]``
    for the marked items only; unmarked ``ing`` is not an affix here.
    """
    marked = []
    for item in split_list(value):
        grapheme = item.partition("/")[0].strip()
        if grapheme and (
            grapheme[0] in _POSITIONAL_MARKERS or grapheme[-1] in _POSITIONAL_MARKERS
        ):
            marked.append(grapheme)
    return list(dict.fromkeys(marked))


# Fields parsed from a workbook column other than their own name. ``affixes``
# has no column of its own: it is derived from the GPC cell, where the marker
# is written before ``extract_gpc_pairs`` strips it.
FIELD_SOURCE_COLUMNS: dict[str, str] = {"affixes": "gpcs"}


# Registry: maps language_iso_code to the fields that belong in student_knowledge
# and how to parse each one from a workbook cell string.
# Add a new entry here when adding support for a new language.
STUDENT_KNOWLEDGE_FIELDS: dict[str, dict[str, Callable[[str], Any]]] = {
    "sw": {
        "graphemes": split_list,
        "clusters": split_list,
        "cluster_patterns": parse_cluster_patterns,
        "whole_words": split_list,
    },
    "en": {
        "gpcs": extract_gpc_pairs,
        "sight_words": split_list,
        "affixes": extract_affixes,
    },
}


def get_workbook_language_iso_code(df: pd.DataFrame, description: str) -> str:
    """Read and validate the language_iso_code from a sheet DataFrame.

    Raises ValueError if the column is missing, empty, or contains more than
    one distinct value — all rows in a sheet must belong to the same language.
    """
    col = "language_iso_code"
    if col not in df.columns:
        raise ValueError(f"{description} is missing required column '{col}'")
    return require_single_language_iso_code(
        df[col].dropna().astype(str), source=description
    )


def preprocess_sheet_data(
    sheet_data: pd.DataFrame,
    required_columns: set[str],
    id_col: str,
    description: str,
) -> pd.DataFrame:
    """Validate, normalize, and filter a sheet DataFrame.

    Raises ValueError if required columns are missing, no data rows remain
    after filtering blank IDs, or any ID value is duplicated.
    """
    missing_columns = required_columns - set(sheet_data.columns)
    if missing_columns:
        raise ValueError(
            f"{description} is missing required column(s): "
            f"{', '.join(sorted(missing_columns))}"
        )

    for col in required_columns:
        sheet_data[col] = sheet_data[col].fillna("").astype(str).str.strip()

    filtered_data = sheet_data[sheet_data[id_col] != ""]

    if filtered_data.empty:
        raise ValueError(f"No records extracted from {description}")

    dupes = filtered_data[filtered_data[id_col].duplicated(keep=False)][id_col].unique()
    if len(dupes):
        raise ValueError(
            f"Duplicate {id_col} in {description}: {', '.join(sorted(dupes))}"
        )

    return filtered_data


def is_jagged_profile(value: str) -> bool:
    """Classify a workbook 'Profile type' cell as a jagged knowledge profile.

    Curriculum-aligned profiles follow the taught sequence, so everything else
    ("jagged", "heart-word jagged", "severely behind", "non-sequence", ...) is
    treated as jagged. Written as an exclusion rather than a list of jagged
    types so that a new profile-type label does not silently read as aligned.
    """
    return CURRICULUM_ALIGNED_PROFILE not in value.casefold()


def extract_student_profiles(
    df: pd.DataFrame,
    fields: dict[str, Callable[[str], Any]],
) -> list[StudentKnowledge]:
    return [
        StudentKnowledge(
            student_knowledge_id=str(row.student_knowledge_id),
            language_iso_code=str(row.language_iso_code),
            jagged=is_jagged_profile(str(getattr(row, PROFILE_TYPE_COLUMN))),
            student_knowledge={
                field: parser(str(getattr(row, FIELD_SOURCE_COLUMNS.get(field, field))))
                for field, parser in fields.items()
            },
        )
        for row in df.itertuples(index=False)
    ]


def extract_texts(df: pd.DataFrame) -> list[DecodableText]:
    return [
        DecodableText(
            text_id=str(row.text_id),
            language_iso_code=str(row.language_iso_code),
            content=str(row.content),
        )
        for row in df.itertuples(index=False)
    ]


def ingest_benchmark_data(
    language: str = typer.Option(
        ...,
        "--language",
        "-l",
        help="Language folder name under the data root, for example 'kiswahili'.",
    ),
    workbook: Path = typer.Option(
        ...,
        "--workbook",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to the filled-in .xlsx data-entry template.",
    ),
    data_root: Path = typer.Option(
        DEFAULT_DATA_ROOT,
        "--data-root",
        file_okay=False,
        help="Root data directory. Defaults to the package's data/ folder.",
    ),
) -> None:
    """Parse a data-entry workbook into decodable texts and student profiles."""
    student_df = pd.read_excel(workbook, sheet_name=STUDENT_PROFILES_SHEET)
    texts_df = pd.read_excel(workbook, sheet_name=TEXTS_SHEET)

    student_iso = get_workbook_language_iso_code(
        student_df, f"'{STUDENT_PROFILES_SHEET}' sheet"
    )

    student_fields = STUDENT_KNOWLEDGE_FIELDS.get(student_iso)
    if student_fields is None:
        raise ValueError(
            f"No student_knowledge field registry for language '{student_iso}'. "
            f"Add an entry to STUDENT_KNOWLEDGE_FIELDS."
        )

    student_df = preprocess_sheet_data(
        student_df,
        {"student_knowledge_id", "language_iso_code", PROFILE_TYPE_COLUMN}
        | {FIELD_SOURCE_COLUMNS.get(field, field) for field in student_fields},
        "student_knowledge_id",
        f"'{STUDENT_PROFILES_SHEET}' sheet",
    )
    texts_df = preprocess_sheet_data(
        texts_df,
        {"text_id", "language_iso_code", "content"},
        "text_id",
        f"'{TEXTS_SHEET}' sheet",
    )

    profiles = extract_student_profiles(student_df, student_fields)
    validate_student_knowledge(profiles, strict=True)
    texts = extract_texts(texts_df)

    data_dir = write_dataset(language, data_root, texts=texts, profiles=profiles)

    typer.echo(
        f"Wrote {len(profiles)} student knowledge profiles and "
        f"{len(texts)} decodable texts to {data_dir}"
    )
    typer.echo(
        "Next: build_benchmark_item_candidates.py to pair them and add analysis "
        "columns."
    )


def main() -> None:
    """Entry point."""
    typer.run(ingest_benchmark_data)


if __name__ == "__main__":
    main()

r"""Shared helpers for ingesting benchmark data-entry workbooks.

The workbook is a template filled in by language experts. Machine-read columns
are identified by exact header name (case-sensitive), independent of column
order, so the template can carry additional expert-context columns without
breaking ingestion. Rows with a blank ID column are treated as spacing/notes
and skipped.

The cell parsers and sheet-prep helpers here are benchmark-agnostic so other
benchmarks (e.g. decodable_text_generation) can reuse them instead of copying.
"""

from collections.abc import Callable
from typing import Any

import pandas as pd
from decodability.english.definitions import normalize_gpc_grapheme
from decodability.kiswahili.definitions import ClusterPattern

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


# The inventory and ARPABET_TO_IPA in decodability.english.definitions spell the
# voiced velar stop with ASCII "g" (U+0067), but the workbook may carry the
# typographically correct IPA script g (U+0261). Fold it so parsed pairs match.
_PHONEME_TRANSLATION = str.maketrans({"ɡ": "g"})


def normalize_gpc_pair(grapheme: str, phoneme: str) -> tuple[str, str]:
    """Normalize one ``(grapheme, phoneme)`` pair to inventory notation.

    Strips positional edge markers and rewrites medial split-digraph
    underscores via decodability's ``normalize_gpc_grapheme``, and folds the
    IPA script ɡ to ASCII g.
    """
    return (
        normalize_gpc_grapheme(grapheme.strip()),
        phoneme.strip().strip("/").translate(_PHONEME_TRANSLATION),
    )


def extract_gpc_pairs(value: str) -> list[tuple[str, str]]:
    """Parse comma-separated ``grapheme /phoneme/`` pairs from a cell value.
    Apply grapheme and phoneme translations to normalize the input.

    Example: ``"a /æ/, qu /kw/"`` -> ``[("a", "æ"), ("qu", "kw")]``.

    Exact duplicate pairs are dropped, keeping first-seen order. Note that one
    grapheme may legitimately appear more than once with *different* phonemes
    (``th`` -> ``ð`` and ``th`` -> ``θ``); only whole-pair repeats collapse.
    """
    pairs = [
        normalize_gpc_pair(*item.partition("/")[::2]) for item in split_list(value)
    ]
    return list(dict.fromkeys(pairs))


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
    },
}


def get_language_iso_code(df: pd.DataFrame, description: str) -> str:
    """Read and validate the language_iso_code from a sheet DataFrame.

    Raises ValueError if the column is missing, empty, or contains more than
    one distinct value — all rows in a sheet must belong to the same language.
    """
    col = "language_iso_code"
    if col not in df.columns:
        raise ValueError(f"{description} is missing required column '{col}'")
    values = df[col].dropna().astype(str).str.strip()
    values = values[values != ""]
    if values.empty:
        raise ValueError(f"No '{col}' values found in {description}")
    unique = values.unique()
    if len(unique) > 1:
        raise ValueError(
            f"Multiple language_iso_code values in {description}: "
            f"{', '.join(sorted(unique))}"
        )
    return unique[0]


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

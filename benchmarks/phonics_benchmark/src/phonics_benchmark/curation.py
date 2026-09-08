"""Build the generation benchmark's item set from student profiles + scenarios."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from phonics_benchmark.models import DifficultyBucket

# Difficulty buckets by number of graphemes the student has been taught, each
# with its inclusive upper bound. The last bucket is open-ended (None): a
# profile that knows the whole inventory still belongs in "late".
DIFFICULTY_BUCKETS: tuple[tuple[str, int | None], ...] = (
    ("very-early", 10),
    ("early", 17),
    ("mid", 24),
    ("late", None),
)

# How many profiles of each bucket the dev set samples. Weighted towards the
# larger buckets while keeping the sparse "very-early" bucket represented,
# because that is where the task is hardest.
DEV_SET_BUCKET_SIZES: Mapping[str, int] = {
    "very-early": 6,
    "early": 8,
    "mid": 8,
    "late": 8,
}

DEV_SET_RANDOM_STATE = 42

# Marker recorded on every item built with placeholder cluster knowledge.
PLACEHOLDER_CLUSTER_SOURCE = "placeholder_all_patterns"
ORIGINAL_CLUSTER_SOURCE = "dataset"


def get_difficulty_bucket(n_graphemes: int) -> DifficultyBucket:
    """Return the difficulty bucket for a profile that knows n_graphemes.

    The final bucket is the catch-all, so every grapheme count maps somewhere.
    """
    *bounded, (open_ended, _) = DIFFICULTY_BUCKETS
    for name, upper_bound in bounded:
        assert upper_bound is not None
        if n_graphemes <= upper_bound:
            return name  # type: ignore[return-value]
    return open_ended  # type: ignore[return-value]


def apply_placeholder_cluster_knowledge(
    student_knowledge: Mapping[str, Any], language: str
) -> tuple[dict[str, Any], str]:
    """Fill in placeholder consonant-cluster knowledge where the profile has none.

    TODO(data-team): replace with real per-stage cluster knowledge. Tracked as a
    follow-up; the profiles currently carry no cluster information at all.

    Every profile in the Kiswahili dataset omits ``clusters`` and
    ``cluster_patterns``, which under the Kenya-Tusome aggregator makes *any*
    word containing a consonant cluster non-decodable for *every* student. For
    the most advanced profile only 4 word types in the corpus fail on graphemes
    while 299 fail purely on clusters, putting everyday words like "mwalimu",
    "ng'ombe" and "ndege" out of reach for a student who knows the entire
    grapheme inventory. That is a data gap, not a curriculum rule, and it would
    dominate the benchmark.

    The placeholder marks every native cluster *pattern* as known and leaves
    ``clusters`` empty, so loan clusters ("bl", "st", "br") stay undecodable.
    Beginners are not inflated by this: the grapheme scorer still gates them, so
    a four-grapheme profile moves from 1.6% to 1.8% of the corpus while a
    full-inventory profile moves from 74.1% to 96.8%.

    Returns the (possibly unchanged) knowledge dict and a provenance string.
    A profile that already carries cluster knowledge is returned untouched.
    """
    knowledge = dict(student_knowledge)
    if knowledge.get("clusters") or knowledge.get("cluster_patterns"):
        return knowledge, ORIGINAL_CLUSTER_SOURCE

    if language != "sw":
        # Only Kiswahili models cluster knowledge; nothing to fill in elsewhere.
        return knowledge, ORIGINAL_CLUSTER_SOURCE

    from decodability.kiswahili.definitions import ClusterPattern

    knowledge["cluster_patterns"] = [pattern.value for pattern in ClusterPattern]
    return knowledge, PLACEHOLDER_CLUSTER_SOURCE


def extract_unique_profiles(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Collapse a record stream down to its distinct student knowledge profiles.

    Profiles are deduplicated on the *content* of ``student_knowledge``, not on
    ``student_knowledge_id``: the dataset reuses one knowledge blob across
    several ids (15 of the grade-2 ids share a single profile), and running the
    same prompt many times under different labels would buy no information while
    skewing the difficulty mix.

    The first id seen for each distinct profile is kept as its representative.
    """
    profiles: dict[str, dict[str, Any]] = {}
    for record in records:
        knowledge = record["student_knowledge"]
        key = json.dumps(knowledge, sort_keys=True)
        if key not in profiles:
            profiles[key] = {
                "student_knowledge_id": record["student_knowledge_id"],
                "language_iso_code": record["language_iso_code"],
                "student_knowledge": knowledge,
            }

    frame = pd.DataFrame(list(profiles.values()))
    if frame.empty:
        raise ValueError("No student knowledge profiles found in the input records")

    frame["n_graphemes_known"] = frame["student_knowledge"].apply(
        lambda knowledge: len(knowledge.get("graphemes", []))
    )
    frame["difficulty_bucket"] = frame["n_graphemes_known"].apply(get_difficulty_bucket)
    return frame.sort_values("student_knowledge_id").reset_index(drop=True)


def build_generation_items(
    profiles: pd.DataFrame,
    scenarios: Mapping[str, str],
    language: str,
) -> pd.DataFrame:
    """Cross-join profiles x scenarios into the benchmark's item set.

    ``benchmark_item_id`` is ``<scenario_id>_<student_knowledge_id>``.
    """
    if profiles.empty:
        raise ValueError("Cannot build items: no student knowledge profiles")
    if not scenarios:
        raise ValueError("Cannot build items: no scenarios")

    scenarios_df = pd.DataFrame(
        [
            {"scenario_id": scenario_id, "scenario": scenario}
            for scenario_id, scenario in scenarios.items()
        ]
    )
    items = profiles.join(scenarios_df, how="cross")

    cluster_knowledge = items["student_knowledge"].apply(
        lambda knowledge: apply_placeholder_cluster_knowledge(knowledge, language)
    )
    items["student_knowledge"] = [knowledge for knowledge, _ in cluster_knowledge]
    items["cluster_knowledge_source"] = [source for _, source in cluster_knowledge]

    items["benchmark_item_id"] = (
        items["scenario_id"].astype(str)
        + "_"
        + items["student_knowledge_id"].astype(str)
    )
    return items.sort_values("benchmark_item_id").reset_index(drop=True)


def sample_dev_profiles(
    profiles: pd.DataFrame,
    *,
    bucket_sizes: Mapping[str, int] = DEV_SET_BUCKET_SIZES,
    random_state: int = DEV_SET_RANDOM_STATE,
) -> pd.DataFrame:
    """Take a stratified sample of profiles for the fast dev item set.

    Sampling per difficulty bucket keeps the full difficulty range represented
    rather than letting the larger, easier buckets dominate. A bucket with fewer
    profiles than requested contributes all of them. Seeded for reproducibility,
    so the dev set is a fixed set of items rather than a fresh draw per run.
    """
    sampled = []
    for bucket, size in bucket_sizes.items():
        available = profiles[profiles["difficulty_bucket"] == bucket]
        if available.empty:
            continue
        n = min(size, len(available))
        sampled.append(available.sample(n, random_state=random_state))

    if not sampled:
        raise ValueError("Dev sample is empty: no profiles matched any bucket")

    return pd.concat(sampled).sort_values("student_knowledge_id").reset_index(drop=True)


def summarize_items(items: pd.DataFrame) -> dict[str, Any]:
    """Return a provenance/inspection summary of a built item set."""
    return {
        "Total items": len(items),
        "Unique profiles": int(items["student_knowledge_id"].nunique()),
        "Unique scenarios": int(items["scenario_id"].nunique()),
        "Items per difficulty bucket": items["difficulty_bucket"]
        .value_counts()
        .sort_index()
        .to_dict(),
        "Profiles per difficulty bucket": items.drop_duplicates("student_knowledge_id")[
            "difficulty_bucket"
        ]
        .value_counts()
        .sort_index()
        .to_dict(),
        "Cluster knowledge source": items["cluster_knowledge_source"]
        .value_counts()
        .to_dict(),
    }

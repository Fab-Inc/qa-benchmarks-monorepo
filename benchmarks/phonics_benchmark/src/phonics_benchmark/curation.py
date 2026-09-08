"""Build the generation benchmark's item set from student profiles + scenarios."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import pandas as pd

from phonics_benchmark.models import CurriculumPhase, StudentKnowledge

# Curriculum phases by number of graphemes the student has been taught, each
# with its inclusive upper bound. The last phase is open-ended (None): a
# profile that knows the whole inventory still belongs in "late".
CURRICULUM_PHASES: tuple[tuple[str, int | None], ...] = (
    ("very-early", 10),
    ("early", 17),
    ("mid", 24),
    ("late", None),
)

# How many profiles of each phase the dev set samples. Weighted towards the
# larger phases while keeping the sparse "very-early" phase represented,
# because that is where the task is hardest.
DEV_SET_PHASE_SIZES: Mapping[str, int] = {
    "very-early": 6,
    "early": 8,
    "mid": 8,
    "late": 8,
}

DEV_SET_RANDOM_STATE = 42


def get_curriculum_phase(n_graphemes: int) -> CurriculumPhase:
    """Return the curriculum phase for a profile that knows n_graphemes.

    The final phase is the catch-all, so every grapheme count maps somewhere.
    """
    *bounded, (open_ended, _) = CURRICULUM_PHASES
    for name, upper_bound in bounded:
        assert upper_bound is not None
        if n_graphemes <= upper_bound:
            return name  # type: ignore[return-value]
    return open_ended  # type: ignore[return-value]


def profiles_to_dataframe(profiles: Iterable[StudentKnowledge]) -> pd.DataFrame:
    """Turn validated student profiles into the frame the item builder consumes.

    Adds the derived columns the benchmark groups and reports by:
    ``n_graphemes_known``, ``n_clusters_known``, ``n_cluster_patterns_known``
    and ``curriculum_phase``.

    Cluster counts are recorded because they carry real signal that the grapheme
    count alone hides: only the grade-2 profiles have been taught consonant
    clusters, so two profiles that both know all 31 graphemes can differ by more
    than 20 percentage points of readable vocabulary depending on whether
    clusters are among them.
    """
    frame = pd.DataFrame(
        [
            {
                "student_knowledge_id": profile.student_knowledge_id,
                "language_iso_code": profile.language_iso_code,
                "student_knowledge": profile.student_knowledge,
            }
            for profile in profiles
        ]
    )
    if frame.empty:
        raise ValueError("No student knowledge profiles found in the input records")

    duplicated = frame["student_knowledge_id"].duplicated()
    if duplicated.any():
        raise ValueError(
            "Duplicate student_knowledge_id in profiles: "
            f"{sorted(frame.loc[duplicated, 'student_knowledge_id'])}"
        )

    def count(field: str) -> pd.Series:
        return frame["student_knowledge"].apply(
            lambda knowledge: len(knowledge.get(field) or [])
        )

    frame["n_graphemes_known"] = count("graphemes")
    frame["n_clusters_known"] = count("clusters")
    frame["n_cluster_patterns_known"] = count("cluster_patterns")
    frame["curriculum_phase"] = frame["n_graphemes_known"].apply(get_curriculum_phase)
    return frame.sort_values("student_knowledge_id").reset_index(drop=True)


def build_generation_items(
    profiles: pd.DataFrame,
    scenarios: Mapping[str, str],
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

    items["benchmark_item_id"] = (
        items["scenario_id"].astype(str)
        + "_"
        + items["student_knowledge_id"].astype(str)
    )
    return items.sort_values("benchmark_item_id").reset_index(drop=True)


def sample_dev_profiles(
    profiles: pd.DataFrame,
    *,
    phase_sizes: Mapping[str, int] = DEV_SET_PHASE_SIZES,
    random_state: int = DEV_SET_RANDOM_STATE,
) -> pd.DataFrame:
    """Take a stratified sample of profiles for the fast dev item set.

    Sampling per curriculum phase keeps the full range represented rather than
    letting the larger, later phases dominate. A phase with fewer
    profiles than requested contributes all of them. Seeded for reproducibility,
    so the dev set is a fixed set of items rather than a fresh draw per run.
    """
    sampled = []
    for phase, size in phase_sizes.items():
        available = profiles[profiles["curriculum_phase"] == phase]
        if available.empty:
            continue
        n = min(size, len(available))
        sampled.append(available.sample(n, random_state=random_state))

    if not sampled:
        raise ValueError("Dev sample is empty: no profiles matched any phase")

    return pd.concat(sampled).sort_values("student_knowledge_id").reset_index(drop=True)


def summarize_items(items: pd.DataFrame) -> dict[str, Any]:
    """Return a provenance/inspection summary of a built item set."""
    profiles = items.drop_duplicates("student_knowledge_id")
    taught_clusters = (
        profiles["n_clusters_known"] + profiles["n_cluster_patterns_known"]
    )
    return {
        "Total items": len(items),
        "Unique profiles": int(items["student_knowledge_id"].nunique()),
        "Unique scenarios": int(items["scenario_id"].nunique()),
        "Items per curriculum phase": items["curriculum_phase"]
        .value_counts()
        .sort_index()
        .to_dict(),
        "Profiles per curriculum phase": profiles["curriculum_phase"]
        .value_counts()
        .sort_index()
        .to_dict(),
        "Profiles taught any consonant cluster": int((taught_clusters > 0).sum()),
    }

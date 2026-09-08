"""Tests for building the generation item set from profiles and scenarios."""

import pandas as pd
import pytest
from phonics_benchmark.curation import (
    build_generation_items,
    get_curriculum_phase,
    profiles_to_dataframe,
    sample_dev_profiles,
    summarize_items,
)
from phonics_benchmark.models import StudentKnowledge

SCENARIOS = {"story": "Write a story.", "dialogue": "Write a dialogue."}


def make_profile(
    sk_id: str,
    graphemes: list[str],
    *,
    clusters: list[str] | None = None,
    cluster_patterns: list[str] | None = None,
) -> StudentKnowledge:
    return StudentKnowledge(
        student_knowledge_id=sk_id,
        language_iso_code="sw",
        student_knowledge={
            "graphemes": graphemes,
            "clusters": clusters or [],
            "cluster_patterns": cluster_patterns or [],
            "whole_words": ["mama"],
        },
    )


class TestGetCurriculumPhase:
    """Mapping grapheme counts onto reporting phases."""

    @pytest.mark.parametrize(
        ("n_graphemes", "expected"),
        [
            (0, "very-early"),
            (2, "very-early"),
            (10, "very-early"),
            (11, "early"),
            (17, "early"),
            (18, "mid"),
            (24, "mid"),
            (25, "late"),
            (31, "late"),
            (99, "late"),
        ],
    )
    def test_phase_boundaries(self, n_graphemes: int, expected: str) -> None:
        """Boundaries are inclusive of their upper bound, and 'late' is open."""
        assert get_curriculum_phase(n_graphemes) == expected


class TestProfilesToDataFrame:
    """Turning canonical profiles into the item builder's input frame."""

    def test_keeps_every_profile(self) -> None:
        """The canonical file is the source of truth: no profile is dropped.

        Profiles are no longer deduplicated on knowledge content — the data
        team's file has 156 distinct ids and 156 distinct knowledge blobs, and
        two stages that happen to teach the same units are still two stages.
        """
        profiles = [make_profile(f"id-{i}", ["a", "m"]) for i in range(5)]
        frame = profiles_to_dataframe(profiles)
        assert len(frame) == 5
        assert set(frame["student_knowledge_id"]) == {f"id-{i}" for i in range(5)}

    def test_derives_counts_and_phase(self) -> None:
        """Grouping columns are derived once, here, from the knowledge itself."""
        frame = profiles_to_dataframe(
            [
                make_profile(
                    "id-a",
                    ["a", "m", "k"],
                    clusters=["mk"],
                    cluster_patterns=["prenasalised"],
                )
            ]
        )
        row = frame.iloc[0]
        assert row["n_graphemes_known"] == 3
        assert row["n_clusters_known"] == 1
        assert row["n_cluster_patterns_known"] == 1
        assert row["curriculum_phase"] == "very-early"

    def test_counts_absent_cluster_fields_as_zero(self) -> None:
        """A profile taught no clusters yet records zero, not a missing value."""
        frame = profiles_to_dataframe([make_profile("id-a", ["a", "m"])])
        assert frame.loc[0, "n_clusters_known"] == 0
        assert frame.loc[0, "n_cluster_patterns_known"] == 0

    def test_is_sorted_by_id(self) -> None:
        """Deterministic ordering keeps the built items_sha256 stable."""
        frame = profiles_to_dataframe(
            [make_profile("id-c", ["a"]), make_profile("id-a", ["a"])]
        )
        assert list(frame["student_knowledge_id"]) == ["id-a", "id-c"]

    def test_duplicate_ids_raise(self) -> None:
        """Two profiles sharing an id would collide into one benchmark item."""
        profiles = [make_profile("id-a", ["a"]), make_profile("id-a", ["a", "m"])]
        with pytest.raises(ValueError, match="Duplicate student_knowledge_id"):
            profiles_to_dataframe(profiles)

    def test_empty_input_raises(self) -> None:
        """No profiles means no benchmark; fail rather than write an empty set."""
        with pytest.raises(ValueError, match="No student knowledge profiles"):
            profiles_to_dataframe([])


class TestBuildGenerationItems:
    """The scenario x profile cross product."""

    @pytest.fixture
    def profiles(self) -> pd.DataFrame:
        return profiles_to_dataframe(
            [make_profile("id-a", ["a", "m"]), make_profile("id-b", ["a", "m", "k"])]
        )

    def test_produces_the_full_cross_product(self, profiles: pd.DataFrame) -> None:
        """Every profile is paired with every scenario, exactly once."""
        items = build_generation_items(profiles, SCENARIOS)
        assert len(items) == len(profiles) * len(SCENARIOS)
        assert set(items["scenario_id"]) == set(SCENARIOS)

    def test_item_ids_are_unique_and_readable(self, profiles: pd.DataFrame) -> None:
        """The id is <scenario_id>_<student_knowledge_id>."""
        items = build_generation_items(profiles, SCENARIOS)
        assert items["benchmark_item_id"].is_unique
        assert "story_id-a" in set(items["benchmark_item_id"])

    def test_passes_student_knowledge_through_unchanged(self) -> None:
        """The knowledge reaching the prompt is exactly the canonical knowledge.

        Nothing is injected or filled in at build time any more.
        """
        profile = make_profile("id-a", ["a", "m", "k"], clusters=["mk"])
        items = build_generation_items(profiles_to_dataframe([profile]), SCENARIOS)
        assert items.loc[0, "student_knowledge"] == profile.student_knowledge

    def test_carries_the_grouping_columns_onto_every_item(
        self, profiles: pd.DataFrame
    ) -> None:
        """Each item knows its own phase and cluster counts for the report."""
        items = build_generation_items(profiles, SCENARIOS)
        for column in (
            "curriculum_phase",
            "n_graphemes_known",
            "n_clusters_known",
            "n_cluster_patterns_known",
        ):
            assert items[column].notna().all()

    def test_is_deterministic(self, profiles: pd.DataFrame) -> None:
        """Rebuilding gives identical ordering, so items_sha256 is stable."""
        first = build_generation_items(profiles, SCENARIOS)
        second = build_generation_items(profiles, SCENARIOS)
        assert list(first["benchmark_item_id"]) == list(second["benchmark_item_id"])

    def test_no_profiles_raises(self) -> None:
        """An empty profile frame is an error, not an empty item set."""
        with pytest.raises(ValueError, match="no student knowledge profiles"):
            build_generation_items(pd.DataFrame(), SCENARIOS)

    def test_no_scenarios_raises(self, profiles: pd.DataFrame) -> None:
        """An empty scenario map is an error too."""
        with pytest.raises(ValueError, match="no scenarios"):
            build_generation_items(profiles, {})


class TestSampleDevProfiles:
    """The stratified dev-set sample."""

    @pytest.fixture
    def profiles(self) -> pd.DataFrame:
        records = [make_profile(f"early-{i}", ["a"] * 4) for i in range(10)]
        records += [make_profile(f"late-{i}", ["a"] * 30) for i in range(10)]
        return profiles_to_dataframe(records)

    def test_samples_the_requested_number_per_phase(
        self, profiles: pd.DataFrame
    ) -> None:
        """Each phase contributes its configured share, keeping the range."""
        sampled = sample_dev_profiles(
            profiles, phase_sizes={"very-early": 3, "late": 5}
        )
        assert sampled["curriculum_phase"].value_counts().to_dict() == {
            "very-early": 3,
            "late": 5,
        }

    def test_a_small_phase_contributes_all_it_has(self, profiles: pd.DataFrame) -> None:
        """Asking for more than exist takes everything rather than failing."""
        sampled = sample_dev_profiles(profiles, phase_sizes={"very-early": 99})
        assert len(sampled) == 10

    def test_absent_phases_are_skipped(self, profiles: pd.DataFrame) -> None:
        """A phase with no profiles simply contributes nothing."""
        sampled = sample_dev_profiles(profiles, phase_sizes={"mid": 5, "late": 2})
        assert set(sampled["curriculum_phase"]) == {"late"}

    def test_is_reproducible(self, profiles: pd.DataFrame) -> None:
        """A fixed seed makes the dev set a stable set of items across runs."""
        first = sample_dev_profiles(profiles)
        second = sample_dev_profiles(profiles)
        assert list(first["student_knowledge_id"]) == list(
            second["student_knowledge_id"]
        )

    def test_a_different_seed_gives_a_different_sample(
        self, profiles: pd.DataFrame
    ) -> None:
        """The seed is doing real work, not silently ignored."""
        first = sample_dev_profiles(profiles, phase_sizes={"late": 3})
        other = sample_dev_profiles(profiles, phase_sizes={"late": 3}, random_state=7)
        assert list(first["student_knowledge_id"]) != list(
            other["student_knowledge_id"]
        )

    def test_no_matching_phase_raises(self, profiles: pd.DataFrame) -> None:
        """A sample that would be empty is an error."""
        with pytest.raises(ValueError, match="Dev sample is empty"):
            sample_dev_profiles(profiles, phase_sizes={"mid": 5})


class TestSummarizeItems:
    """The provenance summary written alongside the item files."""

    def test_reports_counts_and_composition(self) -> None:
        """The summary records what was built, for the run's provenance trail."""
        profiles = profiles_to_dataframe(
            [
                make_profile("id-a", ["a", "m"]),
                make_profile("id-b", ["a"] * 30, clusters=["mb"]),
            ]
        )
        summary = summarize_items(build_generation_items(profiles, SCENARIOS))
        assert summary["Total items"] == 4
        assert summary["Unique profiles"] == 2
        assert summary["Unique scenarios"] == 2
        assert summary["Profiles per curriculum phase"] == {
            "late": 1,
            "very-early": 1,
        }
        assert summary["Profiles taught any consonant cluster"] == 1

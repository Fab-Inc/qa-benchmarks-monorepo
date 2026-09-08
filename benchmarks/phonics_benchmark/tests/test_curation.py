"""Tests for building the generation item set from profiles and scenarios."""

import pandas as pd
import pytest
from decodability.kiswahili.definitions import ClusterPattern
from decodability.kiswahili.models import KiswahiliStudentKnowledge
from phonics_benchmark.curation import (
    ORIGINAL_CLUSTER_SOURCE,
    PLACEHOLDER_CLUSTER_SOURCE,
    apply_placeholder_cluster_knowledge,
    build_generation_items,
    extract_unique_profiles,
    get_difficulty_bucket,
    sample_dev_profiles,
    summarize_items,
)

SCENARIOS = {"story": "Write a story.", "dialogue": "Write a dialogue."}


def make_record(sk_id: str, graphemes: list[str], **extra: object) -> dict:
    return {
        "student_knowledge_id": sk_id,
        "language_iso_code": "sw",
        "student_knowledge": {"graphemes": graphemes, "whole_words": ["mama"]},
        **extra,
    }


class TestGetDifficultyBucket:
    """Mapping grapheme counts onto reporting buckets."""

    @pytest.mark.parametrize(
        ("n_graphemes", "expected"),
        [
            (0, "very-early"),
            (4, "very-early"),
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
    def test_bucket_boundaries(self, n_graphemes: int, expected: str) -> None:
        """Boundaries are inclusive of their upper bound, and 'late' is open."""
        assert get_difficulty_bucket(n_graphemes) == expected


class TestApplyPlaceholderClusterKnowledge:
    """The documented stand-in for missing cluster knowledge."""

    def test_fills_in_every_native_cluster_pattern(self) -> None:
        """A profile with no cluster data gets all native patterns marked known."""
        knowledge, source = apply_placeholder_cluster_knowledge(
            {"graphemes": ["a", "m"]}, "sw"
        )
        assert source == PLACEHOLDER_CLUSTER_SOURCE
        assert set(knowledge["cluster_patterns"]) == {
            pattern.value for pattern in ClusterPattern
        }

    def test_leaves_explicit_clusters_empty(self) -> None:
        """Loan clusters stay untaught, so words like "stesheni" remain blocked."""
        knowledge, _ = apply_placeholder_cluster_knowledge(
            {"graphemes": ["a", "m"]}, "sw"
        )
        assert not knowledge.get("clusters")

    def test_result_validates_against_the_language_model(self) -> None:
        """The placeholder must survive KiswahiliStudentKnowledge's validators."""
        knowledge, _ = apply_placeholder_cluster_knowledge(
            {"graphemes": ["a", "m", "u", "k"], "whole_words": ["mama"]}, "sw"
        )
        assert KiswahiliStudentKnowledge.model_validate(knowledge)

    def test_existing_cluster_knowledge_is_left_alone(self) -> None:
        """Real cluster data wins; the placeholder never overwrites it."""
        original = {"graphemes": ["a", "m"], "cluster_patterns": ["prenasalised"]}
        knowledge, source = apply_placeholder_cluster_knowledge(original, "sw")
        assert knowledge["cluster_patterns"] == ["prenasalised"]
        assert source == ORIGINAL_CLUSTER_SOURCE

    def test_explicit_clusters_also_count_as_existing_knowledge(self) -> None:
        """A profile listing clusters directly is already specified."""
        _, source = apply_placeholder_cluster_knowledge(
            {"graphemes": ["s", "t"], "clusters": ["st"]}, "sw"
        )
        assert source == ORIGINAL_CLUSTER_SOURCE

    def test_is_idempotent(self) -> None:
        """Applying twice changes nothing after the first pass."""
        once, _ = apply_placeholder_cluster_knowledge({"graphemes": ["a"]}, "sw")
        twice, source = apply_placeholder_cluster_knowledge(once, "sw")
        assert twice == once
        assert source == ORIGINAL_CLUSTER_SOURCE

    def test_does_not_mutate_its_input(self) -> None:
        """The caller's dict is left untouched."""
        original = {"graphemes": ["a", "m"]}
        apply_placeholder_cluster_knowledge(original, "sw")
        assert original == {"graphemes": ["a", "m"]}

    def test_other_languages_are_untouched(self) -> None:
        """Only Kiswahili models cluster knowledge, so nothing is invented."""
        knowledge, source = apply_placeholder_cluster_knowledge({"gpcs": []}, "en")
        assert knowledge == {"gpcs": []}
        assert source == ORIGINAL_CLUSTER_SOURCE


class TestExtractUniqueProfiles:
    """Collapsing a record stream to its distinct profiles."""

    def test_deduplicates_on_knowledge_content_not_id(self) -> None:
        """Two ids sharing one knowledge blob are one profile, not two.

        The dataset reuses a single grade-2 profile across 15 ids; running the
        same prompt under each label would buy no information.
        """
        records = [
            make_record("id-a", ["a", "m"]),
            make_record("id-b", ["a", "m"]),
            make_record("id-c", ["a", "m", "k"]),
        ]
        profiles = extract_unique_profiles(records)
        assert len(profiles) == 2
        assert set(profiles["student_knowledge_id"]) == {"id-a", "id-c"}

    def test_repeated_pairings_of_one_profile_collapse(self) -> None:
        """A labelled-items file repeats each profile once per text."""
        records = [make_record("id-a", ["a", "m"]) for _ in range(20)]
        assert len(extract_unique_profiles(records)) == 1

    def test_adds_grapheme_count_and_bucket(self) -> None:
        """Difficulty columns are derived once, here, for later grouping."""
        profiles = extract_unique_profiles([make_record("id-a", ["a", "m", "k"])])
        assert profiles.loc[0, "n_graphemes_known"] == 3
        assert profiles.loc[0, "difficulty_bucket"] == "very-early"

    def test_empty_input_raises(self) -> None:
        """No profiles means no benchmark; fail rather than write an empty set."""
        with pytest.raises(ValueError, match="No student knowledge profiles"):
            extract_unique_profiles([])


class TestBuildGenerationItems:
    """The scenario x profile cross product."""

    @pytest.fixture
    def profiles(self) -> pd.DataFrame:
        return extract_unique_profiles(
            [make_record("id-a", ["a", "m"]), make_record("id-b", ["a", "m", "k"])]
        )

    def test_produces_the_full_cross_product(self, profiles: pd.DataFrame) -> None:
        """Every profile is paired with every scenario, exactly once."""
        items = build_generation_items(profiles, SCENARIOS, "sw")
        assert len(items) == len(profiles) * len(SCENARIOS)
        assert set(items["scenario_id"]) == set(SCENARIOS)

    def test_item_ids_are_unique_and_readable(self, profiles: pd.DataFrame) -> None:
        """The id is <scenario_id>_<student_knowledge_id>."""
        items = build_generation_items(profiles, SCENARIOS, "sw")
        assert items["benchmark_item_id"].is_unique
        assert "story_id-a" in set(items["benchmark_item_id"])

    def test_records_cluster_knowledge_provenance(self, profiles: pd.DataFrame) -> None:
        """Every item says where its cluster knowledge came from."""
        items = build_generation_items(profiles, SCENARIOS, "sw")
        assert set(items["cluster_knowledge_source"]) == {PLACEHOLDER_CLUSTER_SOURCE}

    def test_applies_the_cluster_placeholder_to_knowledge(
        self, profiles: pd.DataFrame
    ) -> None:
        """The knowledge written into items is the patched knowledge."""
        items = build_generation_items(profiles, SCENARIOS, "sw")
        assert items.loc[0, "student_knowledge"]["cluster_patterns"]

    def test_is_deterministic(self, profiles: pd.DataFrame) -> None:
        """Rebuilding gives byte-identical ordering, so items_sha256 is stable."""
        first = build_generation_items(profiles, SCENARIOS, "sw")
        second = build_generation_items(profiles, SCENARIOS, "sw")
        assert list(first["benchmark_item_id"]) == list(second["benchmark_item_id"])

    def test_no_profiles_raises(self) -> None:
        """An empty profile frame is an error, not an empty item set."""
        with pytest.raises(ValueError, match="no student knowledge profiles"):
            build_generation_items(pd.DataFrame(), SCENARIOS, "sw")

    def test_no_scenarios_raises(self, profiles: pd.DataFrame) -> None:
        """An empty scenario map is an error too."""
        with pytest.raises(ValueError, match="no scenarios"):
            build_generation_items(profiles, {}, "sw")


class TestSampleDevProfiles:
    """The stratified dev-set sample."""

    @pytest.fixture
    def profiles(self) -> pd.DataFrame:
        records = [make_record(f"early-{i}", ["a"] * 4) for i in range(10)]
        records += [make_record(f"late-{i}", ["a"] * 30) for i in range(10)]
        # Distinct knowledge per id so dedup keeps them all.
        for index, record in enumerate(records):
            record["student_knowledge"]["whole_words"] = [f"w{index}"]
        return extract_unique_profiles(records)

    def test_samples_the_requested_number_per_bucket(
        self, profiles: pd.DataFrame
    ) -> None:
        """Each bucket contributes its configured share, keeping the range."""
        sampled = sample_dev_profiles(
            profiles, bucket_sizes={"very-early": 3, "late": 5}
        )
        counts = sampled["difficulty_bucket"].value_counts().to_dict()
        assert counts == {"very-early": 3, "late": 5}

    def test_a_small_bucket_contributes_all_it_has(
        self, profiles: pd.DataFrame
    ) -> None:
        """Asking for more than exist takes everything rather than failing."""
        sampled = sample_dev_profiles(profiles, bucket_sizes={"very-early": 99})
        assert len(sampled) == 10

    def test_absent_buckets_are_skipped(self, profiles: pd.DataFrame) -> None:
        """A bucket with no profiles simply contributes nothing."""
        sampled = sample_dev_profiles(profiles, bucket_sizes={"mid": 5, "late": 2})
        assert set(sampled["difficulty_bucket"]) == {"late"}

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
        first = sample_dev_profiles(profiles, bucket_sizes={"late": 3})
        other = sample_dev_profiles(profiles, bucket_sizes={"late": 3}, random_state=7)
        assert list(first["student_knowledge_id"]) != list(
            other["student_knowledge_id"]
        )

    def test_no_matching_bucket_raises(self, profiles: pd.DataFrame) -> None:
        """A sample that would be empty is an error."""
        with pytest.raises(ValueError, match="Dev sample is empty"):
            sample_dev_profiles(profiles, bucket_sizes={"mid": 5})


class TestSummarizeItems:
    """The provenance summary written alongside the item files."""

    def test_reports_counts_and_composition(self) -> None:
        """The summary records what was built, for the run's provenance trail."""
        profiles = extract_unique_profiles(
            [make_record("id-a", ["a", "m"]), make_record("id-b", ["a"] * 30)]
        )
        summary = summarize_items(build_generation_items(profiles, SCENARIOS, "sw"))
        assert summary["Total items"] == 4
        assert summary["Unique profiles"] == 2
        assert summary["Unique scenarios"] == 2
        assert summary["Profiles per difficulty bucket"] == {
            "late": 1,
            "very-early": 1,
        }
        assert summary["Cluster knowledge source"] == {PLACEHOLDER_CLUSTER_SOURCE: 4}

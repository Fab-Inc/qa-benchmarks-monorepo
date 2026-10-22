"""Tests for deterministic story metrics (judge/metrics.py)."""

from __future__ import annotations

from decodable_text_generation.judge.metrics import (
    detect_frame,
    max_frame_repeats,
    story_metrics,
)


class TestDetectFrame:
    def test_same_length_near_identical_sentences(self):
        text = "The cat sat. The dog sat. The pig sat. Then it ran away fast."
        n, template = detect_frame(text)
        assert n == 3
        assert template == "the … sat"

    def test_shared_three_word_prefix(self):
        text = "I saw a cat run. I saw a dog jump very high. I saw a bird fly away."
        n, template = detect_frame(text)
        assert n == 3
        assert template is not None

    def test_no_frame(self):
        # clusters of 1 or 2 are below the repeats-frame-3 gate
        assert detect_frame("The cat sat. It was happy. Birds flew past.")[0] < 3
        assert max_frame_repeats("One sentence only.") == 1

    def test_short_sentences_ignored(self):
        # sentences under 3 words can't form a frame
        assert detect_frame("No. Go. Stop. Sit. Run.")[0] == 0

    def test_template_keeps_variable_length_gap(self):
        text = "He ran to the red barn. He ran to the big old barn. He ran to it."
        _, template = detect_frame(text)
        assert template == "he ran to …"


def _knowledge(**kw):
    base = {"gpcs": [], "sight_words": [], "affixes": []}
    base.update(kw)
    return base


class TestStoryMetrics:
    def test_non_english_returns_empty(self):
        assert (
            story_metrics(
                "the cat",
                student_knowledge_id="tusome-ke-l001",
                student_knowledge=_knowledge(),
                language="sw",
            )
            == {}
        )

    def test_empty_knowledge_returns_empty(self):
        assert (
            story_metrics(
                "the cat", student_knowledge_id="ufli-en-l005", student_knowledge={}
            )
            == {}
        )

    def test_counts_decodable_share(self):
        sk = _knowledge(
            gpcs=[
                ["c", "k"],
                ["a", "æ"],
                ["t", "t"],
                ["s", "s"],
                ["d", "d"],
                ["o", "ɑ"],
                ["g", "g"],
                ["m", "m"],
                ["n", "n"],
                ["p", "p"],
                ["r", "r"],
                ["e", "ɛ"],
                ["h", "h"],
                ["i", "ɪ"],
                ["u", "ʌ"],
            ],
            sight_words=["the"],
        )
        m = story_metrics(
            "The cat sat. The dog sat.",
            student_knowledge_id="ufli-en-l005",
            student_knowledge=sk,
        )
        assert m["n_words"] == 6
        assert m["pct_decodable_words"] == 100.0
        assert m["max_frame_repeats"] == 2

    def test_length_bounds_and_score(self):
        # lesson 5 is S1: reference range [20, 80] words
        sk = _knowledge(sight_words=["the", "a"])
        m = story_metrics(
            "the " * 40,
            student_knowledge_id="ufli-en-l005",
            student_knowledge=sk,
        )
        assert m["reference_words_min"] == 20
        assert m["reference_words_max"] == 80
        assert m["length_in_range"] is True
        assert m["length_score"] == 1.0

    def test_ooov_scope_profile_skips_level_metrics(self):
        # a non-aligned id gets only the decodability core, no band metrics
        m = story_metrics(
            "the cat",
            student_knowledge_id="ufli-en-j05",
            student_knowledge=_knowledge(),
        )
        assert "reference_words_min" not in m
        assert m["n_words"] == 2

"""Deterministic story metrics attached to judge judgments.

English-only, curriculum-aligned scope — same as the rubric judge. Most
data lives in the packaged ``english_reference.json`` written by
``make_profiles/build_judge_reference_data.py``:

- ``recent_gpcs`` / ``recent_sight_words``: units introduced in the most
  recent ``recent_window`` lessons of each level — the numerators for
  ``recent_pattern_share`` and ``pattern_spread``.

Length bounds come from the manually-set ``rubrics/scoring.yaml`` (band
``length`` range), not the reference data.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from typing import Any

from decodability import DECODABILITY_REGISTRY, annotate_words
from decodability.english.definitions import normalize_gpc_grapheme
from phonics_benchmark.labeling import resolve_decodability_config

from decodable_text_generation.judge.knowledge_units import (
    PROFILE_BUILDERS,
    WORD_UNITS,
)
from decodable_text_generation.judge.rubric import (
    lesson_for_profile,
    level_for_lesson,
    load_scoring_config,
)
from decodable_text_generation.rules import rule_exercised

DATA_PACKAGE = "decodable_text_generation.judge.data"
REFERENCE_FILE = "english_reference.json"

# Default decodability configs per language. ``score_column`` picks which
# aggregation output decides decodability, and so differs by language.
DECODABILITY_CONFIGS: dict[str, dict] = {
    "sw": {
        "scoring_methods": [
            "score_known_graphemes_kiswahili",
            "score_known_clusters_and_patterns_kiswahili",
            "score_whole_words_kiswahili",
        ],
        "aggregations": {
            "kenya_tusome_score": {"name": "kiswahili_kenya_tusome_aggregator"},
        },
        "score_column": "kenya_tusome_score",
    },
    "en": {
        "scoring_methods": [
            "score_known_gpcs_english",
            "score_sight_words_english",
        ],
        "aggregations": {
            "ufli_score": {"name": "ufli_curriculum_aggregator"},
        },
        "score_column": "ufli_score",
    },
}


def get_decodability_config(language: str) -> tuple[dict, str]:
    """Default (config, score_column) for a language's decodability scoring."""
    config = DECODABILITY_CONFIGS[language]
    return config, config["score_column"]


@dataclass(frozen=True)
class ScoredText:
    """Decodability scores for the words of a text."""

    words: tuple[str, ...]
    unique_words: tuple[str, ...]
    n_words: int
    n_unique_words: int
    n_nondecodable_words: int
    n_unique_nondecodable_words: int
    unique_nondecodable_words: tuple[str, ...]


def score_text_decodability(
    text: str,
    student_knowledge: dict,
    config: dict,
    language: str,
    *,
    score_column: str,
) -> ScoredText:
    """Score every word of ``text`` and return token/unique decodability stats.

    ``score_column`` names the aggregation output that decides decodability;
    words scoring 0.0 there count as non-decodable.
    """
    objs = DECODABILITY_REGISTRY[language]
    words = [w.lower() for w in objs["extract_words"](text)]
    if not words:
        return ScoredText((), (), 0, 0, 0, 0, ())

    scoring_methods, aggregations = resolve_decodability_config(config, language)
    if score_column not in aggregations:
        raise ValueError(
            f"score_column '{score_column}' is not one of the config's aggregation "
            f"outputs: {sorted(aggregations)}"
        )
    knowledge = objs["student_knowledge_model"].model_validate(student_knowledge)
    label_df = annotate_words(
        sorted(set(words)),
        scoring_methods=scoring_methods,
        student_knowledge=knowledge,
        aggregations=aggregations,
    )

    nondecodable = set(label_df["word"][label_df[score_column] == 0.0])
    return ScoredText(
        words=tuple(words),
        unique_words=tuple(sorted(set(words))),
        n_words=len(words),
        n_unique_words=len(set(words)),
        n_nondecodable_words=sum(1 for w in words if w in nondecodable),
        n_unique_nondecodable_words=len(nondecodable),
        unique_nondecodable_words=tuple(sorted(nondecodable)),
    )


@lru_cache(maxsize=1)
def load_reference_data() -> dict[str, Any]:
    """Packaged per-level UFLI reference data for judge metrics."""
    resource = files(DATA_PACKAGE).joinpath(REFERENCE_FILE)
    return json.loads(resource.read_text(encoding="utf-8"))


# --- repeated sentence frames -------------------------------------------------
#
# A "frame" is a sentence shape reused across sentences: either sentences of
# equal length differing in at most two words, or sentences starting with the
# same three words regardless of length. ``max_frame_repeats`` is the size of
# the largest such group; ``repeats-frame-3`` gating fires at 3+.

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|[\r\n]+")
_WORDS = re.compile(r"[a-zA-Z']+")


def _sentence_words(sentence: str) -> list[str]:
    return _WORDS.findall(sentence.lower())


def _frame_template(cluster: list[list[str]]) -> str:
    """The shared shape of a frame group: common words kept, differing
    slots shown as an ellipsis (e.g. ``the rat sat in the ...``)."""
    members = cluster
    min_len = min(len(w) for w in members)
    template: list[str] = []
    gap = False
    for i in range(min_len):
        col = {w[i] for w in members}
        if len(col) == 1:
            template.append(col.pop())
            gap = False
        elif not gap:
            template.append("…")
            gap = True
    if any(len(w) > min_len for w in members) and not gap:
        template.append("…")
    return " ".join(template)


def detect_frame(text: str) -> tuple[int, str | None]:
    """(size, template) of the largest repeated sentence frame in ``text``.

    A frame is either 3+ sentences of equal length differing in at most two
    words, or 3+ sentences starting with the same three words (any length).
    """
    sentences = [
        words
        for s in _SENTENCE_SPLIT.split(text)
        if len(words := _sentence_words(s)) >= 3
    ]
    best: list[list[str]] = []
    by_prefix: dict[tuple[str, ...], list[list[str]]] = {}
    for words in sentences:
        by_prefix.setdefault(tuple(words[:3]), []).append(words)
    by_len: dict[int, list[list[str]]] = {}
    for words in sentences:
        by_len.setdefault(len(words), []).append(words)
    for group in by_len.values():
        clusters: list[list[list[str]]] = []
        for words in group:
            for cluster in clusters:
                if sum(a != b for a, b in zip(words, cluster[0])) <= 2:
                    cluster.append(words)
                    break
            else:
                clusters.append([words])
        for cluster in clusters:
            if len(cluster) > len(best):
                best = cluster
    for group in by_prefix.values():
        if len(group) > len(best):
            best = group
    return len(best), _frame_template(best) if best else None


def max_frame_repeats(text: str) -> int:
    return detect_frame(text)[0]


def story_metrics(
    text: str,
    *,
    student_knowledge_id: str,
    student_knowledge: Mapping[str, Any],
    target_knowledge: Mapping[str, Any] | None = None,
    language: str = "en",
) -> dict[str, Any]:
    """Deterministic metrics for one generated story.

    Returns ``{}`` outside the supported scope (non-English or missing
    student knowledge). For an aligned English profile the result is:

    - ``n_words``, ``n_unique_words``, ``pct_decodable_words``
    - ``reference_words_min``/``_max`` and ``length_in_range`` against the
      band's human reference stories (null when the band has none, e.g. S4)
    - ``recent_pattern_share``: share of word *tokens* containing a GPC from
      the most recent ``recent_window`` lessons, or that are a heartword
      introduced in that window
    - ``pattern_spread``: number of *distinct* words containing a recent GPC
    - ``includes_target``: share of the profile's ``target_knowledge`` units
      (GPCs + heartwords taught at that lesson) appearing in the story;
      ``target_knowledge`` comes from the item record when present, else the
      baked per-lesson target sets
    - ``recent_gpc_words`` / ``recent_heartwords_used`` /
      ``target_units_missing``: the words/units behind those counts, for
      auditability
    """
    if language != "en" or not student_knowledge:
        return {}

    config, score_column = get_decodability_config(language)
    scored = score_text_decodability(
        text, dict(student_knowledge), config, language, score_column=score_column
    )
    pct_decodable = (
        100.0 * (scored.n_words - scored.n_nondecodable_words) / scored.n_words
        if scored.n_words
        else None
    )

    metrics: dict[str, Any] = {
        "n_words": scored.n_words,
        "n_unique_words": scored.n_unique_words,
        "n_nondecodable_words": scored.n_nondecodable_words,
        "n_unique_nondecodable_words": scored.n_unique_nondecodable_words,
        "unique_nondecodable_words": scored.unique_nondecodable_words,
        "pct_decodable_words": pct_decodable,
    }
    frame_n, frame_example = detect_frame(text)
    metrics["max_frame_repeats"] = frame_n
    metrics["frame_example"] = frame_example

    lesson = lesson_for_profile(student_knowledge_id)
    if lesson is None:
        return metrics

    data = load_reference_data()
    band = level_for_lesson(lesson)
    length_range = ((load_scoring_config().get("stages") or {}).get(band) or {}).get(
        "length"
    )
    bounds = tuple(length_range) if length_range else None
    metrics["reference_words_min"] = bounds[0] if bounds else None
    metrics["reference_words_max"] = bounds[1] if bounds else None
    metrics["length_in_range"] = (
        bounds[0] <= scored.n_words <= bounds[1] if bounds else None
    )
    if bounds:
        lo, hi = bounds
        n = scored.n_words
        metrics["length_score"] = (
            max(0.0, (n - lo / 2) / (lo / 2))
            if n < lo
            else 1.0
            if n <= hi
            else max(0.0, 1.0 - (n - hi) / hi)
        )
    else:
        metrics["length_score"] = None

    # Reference GPCs carry workbook edge markers ("-er", "pre-"); word units
    # use the parser's stripped symbols, so normalize before comparing.
    recent_gpcs = {
        (normalize_gpc_grapheme(g), p)
        for g, p in data["recent_gpcs"].get(str(lesson), [])
    }
    recent_sight = set(data["recent_sight_words"].get(str(lesson), []))
    metrics["recent_window"] = data["recent_window"]

    if not scored.n_words:
        metrics.update(
            recent_pattern_share=None,
            pattern_spread=0,
            n_recent_heartword_types=0,
            recent_gpc_words=[],
            recent_heartwords_used=[],
        )
        return metrics

    sk_model = DECODABILITY_REGISTRY["en"]["student_knowledge_model"].model_validate(
        dict(student_knowledge)
    )
    profile = PROFILE_BUILDERS["en"](sk_model)
    word_units = WORD_UNITS["en"]

    recent_gpc_words: set[str] = set()
    recent_heartwords: set[str] = set()
    for word in scored.unique_words:
        units = word_units(word, profile)
        if any(
            unit[0] == "gpc" and (unit[1], unit[2]) in recent_gpcs for unit in units
        ):
            recent_gpc_words.add(word)
        if any(unit[0] == "sight" and unit[1] in recent_sight for unit in units):
            recent_heartwords.add(word)

    words = DECODABILITY_REGISTRY["en"]["extract_words"](text)
    token_counts = Counter(w.lower() for w in words)
    recent_tokens = sum(
        token_counts[word] for word in recent_gpc_words | recent_heartwords
    )

    metrics.update(
        recent_pattern_share=recent_tokens / scored.n_words,
        pattern_spread=len(recent_gpc_words),
        n_recent_heartword_types=len(recent_heartwords),
        recent_gpc_words=sorted(recent_gpc_words),
        recent_heartwords_used=sorted(recent_heartwords),
    )

    # includes_target: share of this lesson's target units present in the story.
    # Per-item target_knowledge wins; otherwise the baked per-lesson sets
    # (same TSV source) are used. A target GPC counts when some word's
    # fully-taught parse credits it; a target heartword counts when the word
    # (or an inflected form of it) is in the story. ``description`` targets
    # (rule lessons) are not measurable — includes_target stays None.
    if target_knowledge is not None:
        metrics["target_description"] = target_knowledge.get("description")
        target_gpcs = {
            (normalize_gpc_grapheme(g), p)
            for g, p in target_knowledge.get("gpcs") or []
        }
        target_sight = {
            str(w).lower() for w in target_knowledge.get("sight_words") or []
        }
        target_rules = [str(r) for r in target_knowledge.get("rules") or []]
    else:
        target_gpcs = {
            (normalize_gpc_grapheme(g), p)
            for g, p in data["target_gpcs"].get(str(lesson), [])
        }
        target_sight = set(data["target_sight_words"].get(str(lesson), []))
        target_rules: list[str] = []

    gpcs_in_story: set[tuple] = set()
    sight_in_story: set[str] = set()
    # Which words exercised each target unit — so the judge can assess
    # whether the target patterns are woven in naturally. Words known as
    # sight words don't contribute GPC evidence: "does"/"goes" are known
    # whole, not as examples of the -es plural GPC.
    target_gpc_words: dict[str, list[str]] = {}
    for word in scored.unique_words:
        units = word_units(word, profile)
        is_sight = any(unit[0] == "sight" for unit in units)
        for unit in units:
            if unit[0] == "gpc" and not is_sight:
                unit_key = (unit[1], unit[2])
                gpcs_in_story.add(unit_key)
                if unit_key in target_gpcs:
                    target_gpc_words.setdefault(
                        f"{unit_key[0]}→{unit_key[1]}", []
                    ).append(word)
            elif unit[0] == "sight":
                sight_in_story.add(unit[1])
    sight_in_story |= set(scored.unique_words) & target_sight

    # Rule targets: a word exercises the rule when its spelling shows the
    # pattern (stem+doubled consonant+suffix, base-e dropped, VC/CVC shape).
    # Checked over all story words, not just fully-decodable ones.
    rule_words: dict[str, list[str]] = {}
    for rule in target_rules:
        for word in scored.unique_words:
            if rule_exercised(rule, word):
                rule_words.setdefault(rule, []).append(word)

    hit_gpcs = target_gpcs & gpcs_in_story
    hit_sight = target_sight & sight_in_story
    hit_rules = set(rule_words)
    n_hit = len(hit_gpcs) + len(hit_sight) + len(hit_rules)
    n_target = len(target_gpcs) + len(target_sight) + len(target_rules)
    # Unique words carrying any target unit — 0→0.0, 1→0.5, 2+→1.0.
    target_word_count = (
        len(set().union(*target_gpc_words.values(), *rule_words.values()) | hit_sight)
        if n_target
        else None
    )
    metrics.update(
        n_target_units=n_target,
        n_target_units_in_story=n_hit,
        includes_target=(n_hit / n_target if n_target else None),
        n_target_words=target_word_count,
        target_practice=(
            min(1.0, target_word_count / 2) if target_word_count is not None else None
        ),
        target_gpc_words={
            unit: sorted(ws) for unit, ws in sorted(target_gpc_words.items())
        },
        target_sight_words_used=sorted(hit_sight),
        target_rule_words={rule: sorted(ws) for rule, ws in sorted(rule_words.items())},
        target_units_missing=sorted(f"gpc:{g}-{p}" for g, p in target_gpcs - hit_gpcs)
        + [f"sight:{w}" for w in sorted(target_sight - hit_sight)]
        + [f"rule:{r}" for r in sorted(set(target_rules) - hit_rules)],
    )
    return metrics

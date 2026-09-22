import csv
import json
import re
from functools import cache
from pathlib import Path

from decodability.english.models import EnglishStudentKnowledge
from decodability.english.scorers import (
    score_known_gpcs_english,
    score_sight_words_english,
)

FIXTURES = Path(__file__).parent / "fixtures"
ID_PATTERN = re.compile(r"ufli-en-l(\d{3})$")
PROFILE_LEVELS = (
    14,
    21,
    31,
    34,
    43,
    46,
    50,
    53,
    58,
    62,
    63,
    68,
    72,
    78,
    83,
    85,
    89,
    94,
    97,
    110,
    112,
    116,
    128,
)


def level(item_id: str) -> int:
    match = ID_PATTERN.fullmatch(item_id)
    assert match, f"invalid curriculum id: {item_id}"
    return int(match.group(1))


def load_stories() -> dict[str, tuple[str, ...]]:
    with (FIXTURES / "english_stories.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    stories = {}
    for row in rows:
        words = tuple(row["words"].split("|"))
        assert words and all(word == word.lower() for word in words)
        assert len(words) == len(set(words))
        assert row["story_id"] not in stories
        stories[row["story_id"]] = words
    assert len(stories) == 120
    assert sum(map(len, stories.values())) == 7_536
    assert len({word for words in stories.values() for word in words}) == 2_238
    assert tuple(map(level, stories)) == tuple(i for i in range(8, 129) if i != 29)
    return stories


def load_profiles() -> dict[str, EnglishStudentKnowledge]:
    profiles = {}
    knowledge = {"gpcs": [], "sight_words": [], "affixes": []}
    with (FIXTURES / "english_profiles.jsonl").open() as handle:
        rows = [json.loads(line) for line in handle]
    for row in rows:
        profile_id = row.pop("profile_id")
        assert set(row) <= {"add_gpcs", "add_sight_words", "add_affixes"}
        for fixture_key, model_key in (
            ("add_gpcs", "gpcs"),
            ("add_sight_words", "sight_words"),
            ("add_affixes", "affixes"),
        ):
            additions = row.get(fixture_key, [])
            existing = set(map(tuple, knowledge[model_key]))
            assert not set(map(tuple, additions)) & existing
            knowledge[model_key].extend(additions)
        profiles[profile_id] = EnglishStudentKnowledge(**knowledge)
    assert tuple(map(level, profiles)) == PROFILE_LEVELS
    final_profile = profiles["ufli-en-l128"]
    assert (
        len(final_profile.gpcs),
        len(final_profile.sight_words),
        len(final_profile.affixes),
    ) == (165, 121, 31)
    return profiles


def load_expectations(
    stories: dict[str, tuple[str, ...]],
    profiles: dict[str, EnglishStudentKnowledge],
) -> dict[tuple[str, str], frozenset[str]]:
    with (FIXTURES / "english_nondecodable.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    expectations = {}
    for row in rows:
        key = (row["profile_id"], row["story_id"])
        words = frozenset(row["nondecodable_words"].split("|"))
        assert key not in expectations
        assert key[0] in profiles and key[1] in stories
        assert level(key[1]) <= level(key[0])
        assert words and words <= set(stories[key[1]])
        expectations[key] = words
    assert len(expectations) == 189
    assert sum(map(len, expectations.values())) == 257
    assert all(
        "also" not in words or level(profile_id) < 68
        for (profile_id, _), words in expectations.items()
    )
    assert all(
        not ({"cassie", "cassie's"} & words) or level(profile_id) < 110
        for (profile_id, _), words in expectations.items()
    )
    return expectations


STORIES = load_stories()
PROFILES = load_profiles()
EXPECTATIONS = load_expectations(STORIES, PROFILES)


@cache
def is_decodable(profile_id: str, word: str) -> bool:
    knowledge = PROFILES[profile_id]
    return bool(
        score_sight_words_english(word, knowledge)
        or score_known_gpcs_english(word, knowledge)
    )


def test_english_curriculum_decodability_regression() -> None:
    comparisons = 0
    mismatches = []
    for profile_id in PROFILES:
        for story_id, words in STORIES.items():
            if level(story_id) > level(profile_id):
                continue
            comparisons += 1
            actual = frozenset(
                word for word in words if not is_decodable(profile_id, word)
            )
            expected = EXPECTATIONS.get((profile_id, story_id), frozenset())
            if actual != expected:
                unexpected = sorted(actual - expected)
                missing = sorted(expected - actual)
                mismatches.append(
                    f"{profile_id} / {story_id}: "
                    f"unexpected={unexpected}, missing={missing}"
                )
    assert comparisons == 1_425
    assert not mismatches, (
        f"{len(mismatches)} of {comparisons} curriculum comparisons mismatched:\n"
        + "\n".join(mismatches)
    )

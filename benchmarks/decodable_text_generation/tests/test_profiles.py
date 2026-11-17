"""Tests for generation profile reading and target validation."""

import json
from pathlib import Path

import pytest
from decodable_text_generation.profiles import (
    profiles_path,
    read_generation_profiles,
    validate_target,
)


def _write_profiles(path: Path, records: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


def test_profiles_path_layout(tmp_path: Path) -> None:
    assert profiles_path("child", "english", tmp_path) == (
        tmp_path / "child_adapted" / "english" / "student_knowledge_profiles.jsonl"
    )
    assert profiles_path("curriculum", "swahili", tmp_path) == (
        tmp_path / "curriculum_aligned" / "swahili" / "student_knowledge_profiles.jsonl"
    )


def test_read_profiles_child_shape(tmp_path: Path) -> None:
    path = _write_profiles(
        tmp_path / "child_adapted" / "english" / "student_knowledge_profiles.jsonl",
        [
            {
                "student_knowledge_id": "p1",
                "language_iso_code": "en",
                "is_jagged": True,
                "student_knowledge": {"gpcs": [["a", "æ"]], "sight_words": []},
            }
        ],
    )
    (profile,) = read_generation_profiles(path)
    assert profile.student_knowledge_id == "p1"
    assert profile.is_jagged is True
    assert profile.target_knowledge is None


def test_read_profiles_curriculum_shape(tmp_path: Path) -> None:
    # Curriculum profiles have no is_jagged field; they default to aligned.
    path = _write_profiles(
        tmp_path / "p.jsonl",
        [
            {
                "student_knowledge_id": "p2",
                "language_iso_code": "en",
                "student_knowledge": {"gpcs": [["a", "æ"]], "sight_words": ["the"]},
                "target_knowledge": {"gpcs": [["i", "ɪ"]], "sight_words": []},
            }
        ],
    )
    (profile,) = read_generation_profiles(path)
    assert profile.is_jagged is False
    # Normalized GPC pairs come back as tuples (the inventory's canonical form).
    assert profile.target_knowledge == {"gpcs": [("i", "ɪ")], "sight_words": []}


def test_read_profiles_rejects_bad_records(tmp_path: Path) -> None:
    path = _write_profiles(tmp_path / "p.jsonl", [{"nope": True}])
    with pytest.raises(ValueError, match="invalid profile"):
        read_generation_profiles(path)


def test_marked_graphemes_split_into_gpcs_and_affixes(tmp_path: Path) -> None:
    """Edge markers become affixes; the stripped pair stays in gpcs."""
    path = _write_profiles(
        tmp_path / "p.jsonl",
        [
            {
                "student_knowledge_id": "p1",
                "language_iso_code": "en",
                "is_jagged": True,
                "student_knowledge": {
                    "gpcs": [
                        ["a", "æ"],
                        ["-s", "s"],
                        ["a_e", "eɪ"],  # medial underscore: digraph, not affix
                        ["un-", "ʌn"],
                        ["_ce", "s"],
                    ],
                    "sight_words": [],
                },
            }
        ],
    )
    (profile,) = read_generation_profiles(path)
    assert profile.student_knowledge["gpcs"] == [
        ("a", "æ"),
        ("s", "s"),
        ("a-e", "eɪ"),
        ("un", "ʌn"),
        ("ce", "s"),
    ]
    assert profile.student_knowledge["affixes"] == ["-ce", "-s", "un-"]


def test_validate_target_uses_language_model() -> None:
    # n /n/ is a real English GPC; zz /zz/ is not.
    validate_target({"gpcs": [["n", "n"]]}, "en", profile_id="p1")
    with pytest.raises(ValueError, match="Invalid target_knowledge"):
        validate_target({"gpcs": [["zz", "zz"]]}, "en", profile_id="p1")


def test_validate_target_unknown_language() -> None:
    with pytest.raises(ValueError, match="No student_knowledge validator"):
        validate_target({"gpcs": []}, "xx", profile_id="p1")


def test_read_profiles_skips_blank_lines(tmp_path: Path) -> None:
    path = _write_profiles(
        tmp_path / "p.jsonl",
        [
            {
                "student_knowledge_id": "p1",
                "language_iso_code": "en",
                "is_jagged": True,
                "student_knowledge": {"gpcs": [["a", "æ"]], "sight_words": []},
            }
        ],
    )
    path.write_text("\n" + path.read_text() + "\n\n", encoding="utf-8")
    (profile,) = read_generation_profiles(path)
    assert profile.student_knowledge_id == "p1"


def test_read_profiles_empty_file_raises(tmp_path: Path) -> None:
    path = _write_profiles(tmp_path / "p.jsonl", [])
    with pytest.raises(ValueError, match="No student knowledge profiles found"):
        read_generation_profiles(path)


def test_validate_target_en_field_errors() -> None:
    with pytest.raises(ValueError, match="no populated fields"):
        validate_target({"gpcs": []}, "en", profile_id="p1")
    with pytest.raises(ValueError, match="'description' must be a non-empty"):
        validate_target(
            {"gpcs": [["n", "n"]], "description": "   "}, "en", profile_id="p1"
        )
    with pytest.raises(ValueError, match="'description' must be a non-empty"):
        validate_target({"gpcs": [["n", "n"]], "description": 5}, "en", profile_id="p1")
    with pytest.raises(ValueError, match="'rules' must be a list of strings"):
        validate_target(
            {"gpcs": [["n", "n"]], "rules": "vc_cvc"}, "en", profile_id="p1"
        )
    with pytest.raises(ValueError, match="'rules' must be a list of strings"):
        validate_target({"gpcs": [["n", "n"]], "rules": [1]}, "en", profile_id="p1")


def test_validate_target_sw_field_errors() -> None:
    with pytest.raises(ValueError, match="no populated fields"):
        validate_target({"graphemes": []}, "sw", profile_id="p1")
    with pytest.raises(ValueError, match="not in VALID_GRAPHEMES"):
        validate_target({"graphemes": ["bogus"]}, "sw", profile_id="p1")
    with pytest.raises(ValueError, match="non-lowercase whole_words"):
        validate_target({"whole_words": ["Ana"]}, "sw", profile_id="p1")
    with pytest.raises(ValueError, match="unknown cluster pattern"):
        validate_target({"cluster_patterns": ["bogus"]}, "sw", profile_id="p1")
    # A registered cluster pattern passes field validation
    validate_target({"cluster_patterns": ["prenasalised"]}, "sw", profile_id="p1")
    # Kiswahili has no registered rules: any description fails validation
    with pytest.raises(ValueError, match="unknown rules"):
        validate_target(
            {"graphemes": ["a"], "description": "note", "rules": ["vc_cvc"]},
            "sw",
            profile_id="p1",
        )
    with pytest.raises(ValueError, match="'rules' must be a list of strings"):
        validate_target(
            {"graphemes": ["a"], "description": "note", "rules": "x"},
            "sw",
            profile_id="p1",
        )
    with pytest.raises(ValueError, match="'description' must be a string"):
        validate_target({"graphemes": ["a"], "description": 5}, "sw", profile_id="p1")


def test_validate_target_description_requires_valid_rule() -> None:
    """A description alone is unmeasurable — it must name a checkable rule."""
    with pytest.raises(ValueError, match="'description' requires"):
        validate_target({"description": "Doubling rule"}, "en", profile_id="p1")
    with pytest.raises(ValueError, match="'description' requires"):
        validate_target(
            {"gpcs": [["n", "n"]], "description": "a note"}, "en", profile_id="p1"
        )
    with pytest.raises(ValueError, match="unknown rules"):
        validate_target(
            {"description": "Doubling rule", "rules": ["not_a_rule"]},
            "en",
            profile_id="p1",
        )
    # description + a registered rule passes
    validate_target(
        {"description": "VC & CVC practice", "rules": ["vc_cvc"]},
        "en",
        profile_id="p1",
    )


def test_validate_target_sw_description_requires_valid_rule() -> None:
    """Kiswahili has no registered rules, so a description always fails."""
    with pytest.raises(ValueError, match="'description' requires"):
        validate_target(
            {"graphemes": ["a"], "description": "Nasal cluster practice"},
            "sw",
            profile_id="p1",
        )
    # sw targets without a description are unaffected
    validate_target({"graphemes": ["a"], "clusters": []}, "sw", profile_id="p1")

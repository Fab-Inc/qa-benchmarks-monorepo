"""Check that kiswahili_word_analysis labels syllabic nasal words correctly."""

# kiswahili_word_analysis resolves via the scripts/ path that conftest.py adds.
import kiswahili_word_analysis as analysis
import pandas as pd
import pytest
from decodability.kiswahili import KiswahiliStudentKnowledge
from decodability.kiswahili.definitions import ClusterPattern

FalseAlarm = analysis.FalseAlarmFailureMode
Miss = analysis.MissesFailureMode
TAUGHT = FalseAlarm.SYLLABIC_NASAL_TAUGHT
UNTAUGHT = FalseAlarm.SYLLABIC_NASAL_UNTAUGHT


def profile(graphemes, clusters=(), patterns=()):
    return KiswahiliStudentKnowledge(
        graphemes=set(graphemes),
        clusters=set(clusters),
        cluster_patterns=set(patterns),
    )


# (item id, word, student profile, decodable, expected failure mode)
CASES = [
    # Syllabic nasal whose whole run is taught: misreading the nasal as fused
    # still finds the word decodable, so the misreading is harmless.
    (
        "mjomba_m_pattern",
        "mjomba",
        profile("mbjoa", patterns=[ClusterPattern.PRENASALISED_M]),
        True,
        TAUGHT,  # the "m" + consonant pattern covers mj (and the fused mb)
    ),
    # Another syllabic nasal with the whole run taught.
    (
        "nje_nj_taught",
        "nje",
        profile("nje", clusters=["nj"]),
        True,
        TAUGHT,
    ),
    # Syllabic nasals whose whole run is not taught (potential trap for models)
    (
        "mjomba_mb",
        "mjomba",
        profile("mbjoa", clusters=["mb"]),
        True,
        UNTAUGHT,  # the fused mb is taught, the syllabic mj is not
    ),
    ("nje", "nje", profile("nje"), True, UNTAUGHT),
    (
        "mchwa_glide",
        "mchwa",
        profile(["m", "ch", "w", "a"], patterns=[ClusterPattern.LABIALISED_GLIDE]),
        True,
        UNTAUGHT,  # the glide pattern covers chw, but not mchw
    ),
    (
        "alimpenda",
        "alimpenda",
        profile("alimpend", clusters=["nd"]),
        True,
        UNTAUGHT,  # medial syllabic m: mp is not taught
    ),
    (
        "mbwa_mb_bw",  # added: not in syllabic_ungating_test_data
        "mbwa",
        profile("mbwa", clusters=["mb", "bw"]),
        True,
        UNTAUGHT,  # decodable via bw; mb is taught but the whole run mbw is not
    ),
    # Controls that must not land in either syllabic mode.
    (
        "mjomba_bare",
        "mjomba",
        profile("mbjoa"),
        False,
        Miss.UNTAUGHT_NASAL_CLUSTER,  # nondecodable: the fused mb is untaught
    ),
    (
        "mbwa_bare",
        "mbwa",
        profile("mbwa"),
        False,
        Miss.UNTAUGHT_NON_NASAL_CLUSTER,  # nondecodable: bw is untaught
    ),
    (
        "mbwea_mbw",
        "mbwea",
        profile("mbwea", clusters=["mbw"], patterns=[ClusterPattern.PRENASALISED_M]),
        True,
        FalseAlarm.TAUGHT_CLUSTER,  # not on the syllabic list: mbw is fused
    ),
]

ALL = {item for item, *_ in CASES}
NONDECODABLE = {item for item, _, _, decodable, _ in CASES if not decodable}
TRAPS = {item for item, *_, mode in CASES if mode == UNTAUGHT}

# Made-up models: tier, the items they flag as nondecodable, and the expected
# false alarm rates on the (taught, untaught) syllabic modes, which hold 2 and
# 5 words.
MODELS = {
    "oracle": ("Cloud", NONDECODABLE, (0, 0)),
    "fused_reader": ("Cloud", NONDECODABLE | TRAPS, (0, 1)),
    "flag_all": ("Cloud", ALL, (1, 1)),
    "flag_none": ("Cloud", set(), (0, 0)),
    "two_traps": ("Mobile + Tablet", NONDECODABLE | {"mjomba_mb", "nje"}, (0, 0.4)),
    "flags_one_taught": ("Mobile + Tablet", {"mjomba_m_pattern"}, (0.5, 0)),
    "flags_taught_only": (
        "Mobile + Tablet",
        {"mjomba_m_pattern", "nje_nj_taught"},
        (1, 0),
    ),
    "flags_one_of_each": (
        "Mobile + Tablet",
        {"nje_nj_taught", "alimpenda"},
        (0.5, 0.2),
    ),
}


@pytest.fixture(scope="module")
def words():
    """build() output for every (model, item), shaped like load_results output."""
    results = pd.DataFrame(
        [
            {
                "benchmark_item_id": item,
                "model": model,
                "hardware_tier": tier,
                "vocab_set": {word},
                # The results CSV stores the profile as a dict literal.
                "student_knowledge": str(knowledge.model_dump(mode="json")),
                "response_parsed": {word} if item in flagged else set(),
                "nondecodable_words": set() if decodable else {word},
            }
            for model, (tier, flagged, _) in MODELS.items()
            for item, word, knowledge, decodable, _ in CASES
        ]
    )
    return analysis.build(results)


@pytest.mark.parametrize("item, expected", [(item, mode) for item, *_, mode in CASES])
def test_failure_mode(words, item, expected):
    # Every model sees the same item, so every row carries the same mode.
    rows = words[words.benchmark_item_id == item]
    assert len(rows) == len(MODELS)
    assert set(rows.failure_mode) == {expected}


@pytest.mark.parametrize("model", MODELS)
def test_syllabic_false_alarm_rates(words, model):
    table = analysis.failure_mode_table(words).set_index(["model", "failure_mode"])
    taught_rate, untaught_rate = MODELS[model][2]
    assert table.n[(model, TAUGHT)] == 2
    assert table.n[(model, UNTAUGHT)] == 5
    assert table.error_rate[(model, TAUGHT)] == pytest.approx(taught_rate)
    assert table.error_rate[(model, UNTAUGHT)] == pytest.approx(untaught_rate)

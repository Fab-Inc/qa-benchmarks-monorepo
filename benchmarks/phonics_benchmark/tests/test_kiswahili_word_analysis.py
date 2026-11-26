"""Check kiswahili_word_analysis's failure modes (syllabic nasals, compound
clusters) and the chart and Markdown table built from them."""

# kiswahili_word_analysis resolves via the scripts/ path that conftest.py adds.
import kiswahili_word_analysis as analysis
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from decodability.kiswahili import KiswahiliStudentKnowledge
from decodability.kiswahili.definitions import ClusterPattern

FalseAlarm = analysis.FalseAlarmFailureMode
Miss = analysis.MissesFailureMode
# A syllabic nasal whose fused reading is taught is an ordinary cluster word.
CLUSTER = FalseAlarm.TAUGHT_CLUSTER
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
    # still finds the word decodable, so the misreading is harmless and the
    # word is an ordinary taught-cluster word.
    (
        "mjomba_m_pattern",
        "mjomba",
        profile("mbjoa", patterns=[ClusterPattern.PRENASALISED_M]),
        True,
        CLUSTER,  # the "m" + consonant pattern covers mj (and the fused mb)
    ),
    # Another syllabic nasal with the whole run taught.
    (
        "nje_nj_taught",
        "nje",
        profile("nje", clusters=["nj"]),
        True,
        CLUSTER,
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
        Miss.UNTAUGHT_CLUSTER,  # nondecodable: the fused mb is untaught
    ),
    (
        "mbwa_bare",
        "mbwa",
        profile("mbwa"),
        False,
        Miss.UNTAUGHT_CLUSTER,  # nondecodable: bw is untaught
    ),
    (
        "mbwea_mbw",
        "mbwea",
        profile("mbwea", clusters=["mbw"], patterns=[ClusterPattern.PRENASALISED_M]),
        True,
        CLUSTER,  # not on the syllabic list: mbw is fused
    ),
    # Compound clusters: a fused run of 3+ graphemes not taught as a unit,
    # with a smaller run inside it taught. Each is paired with its mirror,
    # where nothing inside the run is taught.
    (
        "mbwea_mb",
        "mbwea",
        profile("mbwea", clusters=["mb"]),
        False,
        Miss.UNTAUGHT_COMPOUND_CLUSTER,  # mb taught, mbw not
    ),
    (
        "mbwea_bare",
        "mbwea",
        profile("mbwea"),
        False,
        Miss.UNTAUGHT_CLUSTER,  # mbw untaught, and no part of it taught
    ),
    (
        "magonjwa_nj_jw",
        "magonjwa",
        profile("magonjw", clusters=["nj", "jw"]),
        False,
        Miss.UNTAUGHT_COMPOUND_CLUSTER,  # nj and jw taught, njw not
    ),
    (
        "kwenda_bare",
        "kwenda",
        profile("kwenad"),
        False,
        Miss.UNTAUGHT_CLUSTER,  # two plain untaught clusters is one reason
    ),
]

ALL = {item for item, *_ in CASES}
NONDECODABLE = {item for item, _, _, decodable, _ in CASES if not decodable}
TRAPS = {item for item, *_, mode in CASES if mode == UNTAUGHT}

# Made-up models: tier, the items they flag as nondecodable, and the expected
# false alarm rates on the (taught cluster, syllabic nasal) modes, which hold
# 3 and 5 words.
MODELS = {
    "oracle": ("Cloud", NONDECODABLE, (0, 0)),
    "fused_reader": ("Cloud", NONDECODABLE | TRAPS, (0, 1)),
    "flag_all": ("Cloud", ALL, (1, 1)),
    "flag_none": ("Cloud", set(), (0, 0)),
    "two_traps": ("Mobile + Tablet", NONDECODABLE | {"mjomba_mb", "nje"}, (0, 0.4)),
    "flags_one_taught": ("Mobile + Tablet", {"mjomba_m_pattern"}, (1 / 3, 0)),
    "flags_taught_only": (
        "Mobile + Tablet",
        {"mjomba_m_pattern", "nje_nj_taught"},
        (2 / 3, 0),
    ),
    "flags_one_of_each": (
        "Mobile + Tablet",
        {"nje_nj_taught", "alimpenda"},
        (1 / 3, 0.2),
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
def test_false_alarm_rates(words, model):
    table = analysis.failure_mode_table(words).set_index(["model", "failure_mode"])
    cluster_rate, untaught_rate = MODELS[model][2]
    assert table.n[(model, CLUSTER)] == 3
    assert table.n[(model, UNTAUGHT)] == 5
    assert table.error_rate[(model, CLUSTER)] == pytest.approx(cluster_rate)
    assert table.error_rate[(model, UNTAUGHT)] == pytest.approx(untaught_rate)


# --------------------------------------------------------------------------
# Tier table as Markdown
# --------------------------------------------------------------------------
CHART_MODES = [mode for rows in analysis.FAILURE_ROWS.values() for mode, _ in rows]


def test_every_chart_row_has_an_example():
    assert set(CHART_MODES) <= set(analysis.EXAMPLE_WORDS)


@pytest.mark.parametrize("mode", CHART_MODES)
def test_example_lands_in_its_mode(mode):
    # Build the example's cut-down profile and run it through build(), as one
    # single-word item, to check the example illustrates the row it sits in.
    word, taught, *_ = analysis.EXAMPLE_WORDS[mode]
    knowledge = KiswahiliStudentKnowledge(**{k: set(v) for k, v in taught.items()})
    nondecodable = isinstance(mode, analysis.MissesFailureMode)
    results = pd.DataFrame(
        [
            {
                "benchmark_item_id": "example",
                "model": "any",
                "hardware_tier": "Cloud",
                "vocab_set": {word},
                "student_knowledge": str(knowledge.model_dump(mode="json")),
                "response_parsed": set(),
                "nondecodable_words": {word} if nondecodable else set(),
            }
        ]
    )
    assert analysis.build(results).failure_mode.item() == mode


# A tier table with one mode per panel above MIN_WORDS, one below it, and the
# rest absent.
TIERS = ["Cloud", "Mobile + Tablet"]
TIER_FAILURES = pd.DataFrame(
    {
        "hardware_tier": TIERS * 3,
        "failure_mode": [UNTAUGHT] * 2
        + [Miss.UNTAUGHT_CLUSTER] * 2
        + [Miss.INVALID_GRAPHEME] * 2,
        "error_rate": [0.864, 0.586, 0.02, 0.45, 0.0, 0.37],
        "n": [777] * 2 + [1000] * 2 + [15] * 2,
    }
)
SELECTED = pd.DataFrame({"model": ["a", "b"], "hardware_tier": TIERS})


def test_tier_failure_markdown():
    text = analysis.tier_failure_markdown(TIER_FAILURES, SELECTED)
    lines = text.splitlines()
    # One table per chart panel, each worded for its error type.
    assert any("Mean false alarm rate: Cloud" in line for line in lines)
    assert any("Why the word is not decodable" in line for line in lines)
    row = next(line for line in lines if line.startswith("| Word with a syllabic"))
    assert row.endswith("| 86% | 59% |")
    assert "| mtoto | graphemes: m, t, o | clusters: mt |" in row
    # Too few item-words: left out of the table, but named in a note.
    assert not any(line.startswith("| Non-Kiswahili grapheme") for line in lines)
    assert "fewer than 20 item-words: Non-Kiswahili grapheme" in text
    # No item-words at all: left out, and not named.
    assert "Known whole word" not in text


def test_example_chart_line_names_its_word():
    # The chart line starts with the example word (bold parts aside).
    for word, *_, chart_line in analysis.EXAMPLE_WORDS.values():
        plain = chart_line.replace(r"$\mathbf{", "").replace("}$", "")
        assert plain.startswith(f"{word}:")


def test_plot_leaves_out_small_modes():
    fig = analysis.plot_failure_modes(TIER_FAILURES, SELECTED)
    # Row names and example lines are annotations; the tick labels are empty.
    labels = [t.get_text() for ax in fig.axes for t in ax.texts]
    assert any(label.startswith("Word with a syllabic nasal") for label in labels)
    assert analysis.EXAMPLE_WORDS[UNTAUGHT][4] in labels
    assert "Untaught cluster" in labels
    assert analysis.EXAMPLE_WORDS[Miss.UNTAUGHT_CLUSTER][4] in labels
    assert "Non-Kiswahili grapheme" not in labels
    assert analysis.EXAMPLE_WORDS[Miss.INVALID_GRAPHEME][4] not in labels
    assert any(
        "fewer than 20 item-words: Non-Kiswahili grapheme" in t.get_text()
        for t in fig.texts
    )
    plt.close(fig)

"""Tests for the GPC (grapheme-phoneme correspondence) layer."""

from primerpro import (
    Consonant,
    GPCMapping,
    Settings,
    Vowel,
    Word,
    get_phoneme_annotation,
    get_phonemic_form,
)

from .fixtures import create_kiswahili_gpc_settings
import pytest


@pytest.mark.parametrize(
    "word_text, expected_annotation",
    [
        ("ngoma", [("ng", "ŋg"), ("o", "ɔ"), ("m", "m"), ("a", "a")]),
        ("shika", [("sh", "ʃ"), ("i", "i"), ("k", "k"), ("a", "a")]),
        ("nyumba", [("ny", "ɲ"), ("u", "u"), ("m", "m"), ("b", "b"), ("a", "a")]),
        ("ng'oa", [("ng'", "ŋ"), ("o", "ɔ"), ("a", "a")]),
        ("ng'ang'ania", [("ng'", "ŋ"), ("a", "a"), ("ng'", "ŋ"), ("a", "a"), ("n", "n"), ("i", "i"), ("a", "a")]),
        ("ungana", [("u", "u"), ("ng", "ŋg"), ("a", "a"), ("n", "n"), ("a", "a")]),
    ]
)
def test_phoneme_annotation_maps_each_grapheme(word_text, expected_annotation):
    """Each segmented grapheme is paired with its phoneme."""

    settings, gpc_mapping = create_kiswahili_gpc_settings()
    word = Word(word_text, settings)

    assert get_phoneme_annotation(word, gpc_mapping) == expected_annotation


def test_multigraph_is_segmented_and_annotated():
    """The multigraph 'sh' is one grapheme mapped to 'ʃ', not 's' + 'h'."""

    settings, gpc_mapping = create_kiswahili_gpc_settings()
    word = Word("ameshiba", settings)

    assert get_phoneme_annotation(word, gpc_mapping) == [
        ("a", "a"),
        ("m", "m"),
        ("e", "ɛ"),
        ("sh", "ʃ"),
        ("i", "i"),
        ("b", "b"),
        ("a", "a"),
    ]


def test_uppercase_initial_maps_to_phoneme():
    """An uppercase first letter resolves via the grapheme's uppercase form."""

    settings, gpc_mapping = create_kiswahili_gpc_settings()
    word = Word("Mtoto", settings)

    assert get_phoneme_annotation(word, gpc_mapping)[0] == ("m", "m")


@pytest.mark.parametrize(
    "word_text, expected_phonemic_form",
    [
        ("ngoma", "ŋgɔma"),
        ("shika", "ʃika"),
        ("nyumba", "ɲumba"),
        ("ng'oa", "ŋɔa"),
        ("ng'ang'ania", "ŋaŋania"),
        ("ungana", "uŋgana"),
    ]
)
def test_phonemic_form_concatenates_phonemes(word_text, expected_phonemic_form):
    """The phonemic form concatenates the grapheme phonemes."""

    settings, gpc_mapping = create_kiswahili_gpc_settings()
    word = Word(word_text, settings)

    assert get_phonemic_form(word, gpc_mapping) == expected_phonemic_form


def test_unknown_character_annotates_with_no_phoneme():
    """A character absent from the inventory is annotated with phoneme None."""

    settings, gpc_mapping = create_kiswahili_gpc_settings()
    word = Word("aq", settings)  # 'q' is not in the Kiswahili inventory

    annotation = get_phoneme_annotation(word, gpc_mapping)
    assert annotation == [("a", "a"), ("q", None)]
    
    
def test_unknown_character_phonemic_form_raises_error():
    """Attempting to get a phonemic form with an unknown grapheme raises an error."""

    settings, gpc_mapping = create_kiswahili_gpc_settings()
    word = Word("aq", settings)  # 'q' is not in the Kiswahili inventory

    with pytest.raises(ValueError, match="contains graphemes with missing phonemes"):
        get_phonemic_form(word, gpc_mapping)


def test_get_phoneme_returns_none_for_unknown_grapheme():
    """Looking up a grapheme that was never registered returns None."""

    gpc_mapping = GPCMapping()
    existing_consonant = Consonant("b")
    gpc_mapping.add(existing_consonant, "b")

    assert gpc_mapping.get_phoneme(existing_consonant) == "b"
    assert gpc_mapping.get_phoneme(Consonant("z")) is None

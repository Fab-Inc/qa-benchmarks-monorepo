"""Tests for the GPC (grapheme-phoneme correspondence) layer."""

from primerpro import (
    Consonant,
    GPCInventory,
    Settings,
    Vowel,
    Word,
    get_phoneme_annotation,
    get_phonemic_form,
)

from .fixtures import create_kiswahili_gpc_settings, KISWAHILI_CONSONANT_GPCS, KISWAHILI_VOWEL_GPCS
import pytest

def test_every_registered_grapheme_has_a_phoneme():
    """Registering via (grapheme, phoneme) pairs guarantees completeness."""
    settings = Settings()
    gpc_inventory = GPCInventory(settings)

    for grapheme, phoneme in KISWAHILI_CONSONANT_GPCS:
        c = Consonant(grapheme)
        c.uppercase = grapheme.upper()
        gpc_inventory.add_consonant(c, phoneme)
    
    for grapheme, phoneme in KISWAHILI_VOWEL_GPCS:
        v = Vowel(grapheme)
        v.uppercase = grapheme.upper()
        gpc_inventory.add_vowel(v, phoneme)

    grapheme_inventory = gpc_inventory.grapheme_inventory

    for grapheme in (*grapheme_inventory.consonants, *grapheme_inventory.vowels):
        assert gpc_inventory.get_phoneme(grapheme) is not None


def test_gpc_inventory_drives_segmentation():
    """The GPC's wrapped grapheme inventory segments words."""

    settings, gpc = create_kiswahili_gpc_settings()
    assert settings.grapheme_inventory is gpc.grapheme_inventory

    word = Word("sh", settings)
    assert word.grapheme_count() == 1  # 'sh' is one grapheme, not 's' + 'h'


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

    settings, gpc = create_kiswahili_gpc_settings()
    word = Word(word_text, settings)

    assert get_phoneme_annotation(word, gpc) == expected_annotation


def test_multigraph_is_segmented_and_annotated():
    """The multigraph 'sh' is one grapheme mapped to 'ʃ', not 's' + 'h'."""

    settings, gpc = create_kiswahili_gpc_settings()
    word = Word("ameshiba", settings)

    assert get_phoneme_annotation(word, gpc) == [
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

    settings, gpc = create_kiswahili_gpc_settings()
    word = Word("Mtoto", settings)

    assert get_phoneme_annotation(word, gpc)[0] == ("m", "m")


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

    settings, gpc = create_kiswahili_gpc_settings()
    word = Word(word_text, settings)

    assert get_phonemic_form(word, gpc) == expected_phonemic_form


def test_unknown_character_annotates_with_no_phoneme():
    """A character absent from the inventory is annotated with phoneme None."""

    settings, gpc = create_kiswahili_gpc_settings()
    word = Word("aq", settings)  # 'q' is not in the Kiswahili inventory

    annotation = get_phoneme_annotation(word, gpc)
    assert annotation == [("a", "a"), ("q", None)]
    
def test_unknown_character_phonemic_form_raises_error():
    """Attempting to get a phonemic form with an unknown grapheme raises an error."""

    settings, gpc = create_kiswahili_gpc_settings()
    word = Word("aq", settings)  # 'q' is not in the Kiswahili inventory

    with pytest.raises(ValueError, match="contains graphemes with missing phonemes"):
        get_phonemic_form(word, gpc)


def test_get_phoneme_returns_none_for_unknown_grapheme():
    """Looking up a grapheme that was never registered returns None."""

    settings = Settings()
    gpc = GPCInventory(settings)
    c = Consonant("b")
    gpc.add_consonant(c, "b")

    assert gpc.get_phoneme(c) == "b"
    assert gpc.get_phoneme(Consonant("z")) is None

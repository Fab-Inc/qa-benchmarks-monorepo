"""

"""
from primerpro.model import Consonant, GraphemeInventory, Settings, Syllograph, SyllographGPC, TextData, Tone, ToneGPC, Vowel, VowelGPC, Word, ConsonantGPC
import pytest
from .fixtures import create_english_settings, create_kiswahili_gpc_settings


@pytest.mark.parametrize(
    "grapheme, marker", 
    [
        (Consonant("b"), "consonant"), 
        (Vowel("a"), "vowel"), 
        (Syllograph("ka"), "syllograph"), 
        (Tone("á"), "tone")
   ]
)
def test_inventory_add_raises_error_for_non_gpc_graphemes_for_gpc_based_settings(grapheme, marker):
    """
    GIVEN GPC-based settings
    WHEN adding a non-GPC grapheme to the inventory
    THEN expect a TypeError
    """
    settings = create_kiswahili_gpc_settings()
    
    with pytest.raises(TypeError, match=f"GPC-enabled settings require {marker}s to be {marker.capitalize()}GPC"):
        getattr(settings.grapheme_inventory, f"add_{marker}")(grapheme)


@pytest.mark.parametrize(
    "gpc_based_grapheme, marker", 
    [
        (ConsonantGPC("b", "b"), "consonant"), 
        (VowelGPC("a", "a"), "vowel"), 
        (SyllographGPC("ka", "ka"), "syllograph"), 
        (ToneGPC("á", "á"), "tone")
    ]
)
def test_inventory_add_passes_for_gpcs_for_gpc_based_settings(gpc_based_grapheme, marker):
    """
    GIVEN GPC-based settings
    WHEN adding a GPC to the inventory
    THEN expect no error
    """
    settings = create_kiswahili_gpc_settings()
    getattr(settings.grapheme_inventory, f"add_{marker}")(gpc_based_grapheme)


@pytest.mark.parametrize(
    "grapheme",
    [Consonant("b"), ConsonantGPC("b", "b")],
)
def test_non_gpc_settings_accept_any_consonant(grapheme):
    """
    GIVEN non-GPC settings
    WHEN adding either a plain or GPC grapheme
    THEN no error is raised (backward compatibility)
    """
    settings = create_english_settings()
    settings.grapheme_inventory.add_consonant(grapheme)


def test_inventory_without_settings_does_not_crash():
    """
    # TODO: Find out when settings should/can be None in practice

    GIVEN an inventory constructed without settings
    WHEN adding a grapheme
    THEN the guard is skipped instead of raising AttributeError
    """
    inventory = GraphemeInventory(settings=None)
    inventory.add_consonant(Consonant("b"))

    assert inventory.consonant_count() == 1


# --- Phoneme lookup ----------------------------------------------------------


def test_phoneme_annotation_maps_each_grapheme():
    """
    GIVEN GPC-based settings
    WHEN a word is annotated
    THEN each grapheme maps to its corresponding phoneme
    """
    settings = create_kiswahili_gpc_settings()
    word = Word("mtoto", settings)

    assert word.get_phoneme_annotation() == [
        ("m", "m"),
        ("t", "t"),
        ("o", "ɔ"),
        ("t", "t"),
        ("o", "ɔ"),
    ]



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
def test_multigraph_is_segmented_and_annotated(word_text, expected_annotation):
    """
    GIVEN GPC-based Kiswahili settings
    WHEN a word contains the multigraph 'sh'
    THEN it is one grapheme mapped to 'ʃ', not 's' + 'h'
    """
    settings = create_kiswahili_gpc_settings()
    word = Word(word_text, settings)

    assert word.get_phoneme_annotation() == expected_annotation


def test_phonemic_form_concatenates_phonemes():
    """
    GIVEN GPC-based Kiswahili settings
    WHEN the phonemic form is requested
    THEN it concatenates the grapheme phonemes into a transcription
    """
    settings = create_kiswahili_gpc_settings()
    word = Word("ameshiba", settings)

    assert word.get_phonemic_form() == "amɛʃiba"


def test_unknown_character_annotates_with_no_phoneme():
    """
    GIVEN GPC-based Kiswahili settings
    WHEN a word contains a character absent from the inventory
    THEN it is annotated as a bare grapheme with phoneme None
    """
    settings = create_kiswahili_gpc_settings()
    word = Word("aq", settings)  # 'q' is not in the Kiswahili inventory

    annotation = word.get_phoneme_annotation()
    assert annotation == [("a", "a"), ("q", None)]
    # get_phonemic_form skips the None phoneme rather than failing.
    assert word.get_phonemic_form() == "a"


@pytest.mark.parametrize("method", ["get_phoneme_annotation", "get_phonemic_form"])
def test_phoneme_methods_require_gpc_settings(method):
    """
    GIVEN non-GPC settings
    WHEN phoneme annotation or form is requested
    THEN a ValueError is raised
    """
    settings = create_english_settings()
    word = Word("sat", settings)

    with pytest.raises(ValueError, match="only available for GPC-based settings"):
        getattr(word, method)()


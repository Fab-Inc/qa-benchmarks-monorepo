from decodability.english.definitions import ARPABET_TO_IPA, VOWEL_INITIALS
from decodability.english.load_g2p_alignments import alignments


def phoneme_arpabet_lookup(phoneme: str) -> str:
    """Lookup an ARPABET phoneme's IPA representation."""
    if phoneme in ARPABET_TO_IPA:
        return ARPABET_TO_IPA[phoneme]
    base = phoneme.rstrip("012")
    if base in ARPABET_TO_IPA:
        return ARPABET_TO_IPA[base]
    raise ValueError(f"Unknown ARPABET phoneme: {phoneme}")


def arpabet_to_ipa(phoneme: str) -> str:
    """Convert an ARPABET phoneme (e.g. 'AH0', 'K|S') to IPA."""
    return "".join(phoneme_arpabet_lookup(p) for p in phoneme.split("|"))


def _normalize_phonemes(phonemes: tuple[str, ...]) -> tuple[str, ...]:
    """Apply dialect normalizations expected by the UPL inventory.

    - Collapses /hw/ → /w/ (the aspirated wh-cluster, merged with /w/ in
      most General American speech) so words like ``whip`` — encoded as
      /hwɪp/ — match ``wh → w`` in the inventory.
    - Undoes CMU's pre-r laxing of long-e: /ɪ/ + /r/ → /i/ + /r/ when the
      /r/ is in coda position (word-final or before a consonant). This
      lets ``beer, ear, beard, clear`` match the ``ee → i``/``ea → i``
      GPCs the phonics program teaches. /ɪ/+/r/ before a vowel (``mirror,
      spirit``) is preserved.
    """
    result: tuple[str, ...] = ()
    i = 0
    while i < len(phonemes):
        if phonemes[i] == "h" and i + 1 < len(phonemes) and phonemes[i + 1] == "w":
            i += 1
            continue
        if (
            phonemes[i] == "ɪ"
            and i + 1 < len(phonemes)
            and phonemes[i + 1] == "r"
            and (i + 2 == len(phonemes) or phonemes[i + 2][0] not in VOWEL_INITIALS)
        ):
            result += ("i",)
            i += 1
            continue
        result += (phonemes[i],)
        i += 1
    return result


def get_word_phonemes(word: str) -> tuple[str, ...]:
    """Get the IPA phonemes for a word, one phoneme per list item.

    Multi-phoneme graphemes in the alignments (encoded as ``'X|Y'`` — e.g.
    ``x -> K|S``) are flattened into individual entries so the output can
    be passed straight to ``parse_gpcs``. Silent ``_`` markers are dropped
    and dialect normalizations (e.g. /hw/ → /w/) are applied.
    """
    if word not in alignments:
        raise ValueError(f"Word not found in alignments: '{word}'")
    phonemes: tuple[str, ...] = ()
    for arpa in alignments[word]["phonemes"]:
        for part in arpa.split("|"):
            if part == "_":
                continue
            phonemes += (phoneme_arpabet_lookup(part),)
    return _normalize_phonemes(phonemes)

from decodability.english.definitions import (
    UFLI_GPC_INVENTORY,
    normalize_gpc_grapheme,
)
from decodability.english.extract_phonemes import get_word_phonemes
from decodability.english.gpc_parser import GPCMatch, parse_gpcs
from decodability.english.models import EnglishStudentKnowledge

GPC_INVENTORY = UFLI_GPC_INVENTORY

# Teaching any of these doubled consonants (lesson 67) introduces the
# general doubling convention, unlocking DERIVED_DOUBLE_/
# DERIVED_REPEAT_FINAL_ matches for all consonants.
_DOUBLING_CONSONANTS = frozenset({"bb", "cc", "dd", "gg", "mm", "nn", "pp", "rr", "tt"})

# The silent-e family: split digraphs, soft c/g before silent e, -ve, and
# short-u o-e. The last of these is taught at lesson 62, when the
# silent-final-e convention counts as established — before then, a student
# cannot be assumed to read a trailing e as silent.
_SILENT_E_FAMILY = frozenset(
    {
        ("a-e", "eɪ"),
        ("i-e", "aɪ"),
        ("o-e", "oʊ"),
        ("e-e", "i"),
        ("u-e", "u"),
        ("u-e", "ju"),
        ("ce", "s"),
        ("ge", "dʒ"),
        ("ve", "v"),
        ("o-e", "ʌ"),
    }
)


def normalize_taught_gpcs(
    gpcs: list[tuple[str, str]],
) -> tuple[set[tuple[str, str]], set[str]]:
    """Canonical form of a student's taught GPC list.

    Returns ``(taught_pairs, taught_graphemes)`` with graphemes normalized so
    a profile entry like ``("-ed", "t")`` and the parser's ``-ed`` match
    (whose ``symbol`` is ``ed``) compare equal.
    """
    pairs = {(normalize_gpc_grapheme(g), p) for g, p in gpcs}
    return pairs, {g for g, _ in pairs}


def _is_taught_match(match: GPCMatch, taught_set: set, taught_graphemes: set) -> bool:
    if match.gpc.id == "SILENT_FINAL_E":
        return _SILENT_E_FAMILY <= taught_set
    if match.gpc.is_silent:
        return True
    if match.gpc.id == "REDUCED_YOD_U":
        # Stricter than the plain reduced-vowel rule: the glide is not free.
        # Only a student taught the unreduced pair u → /ju/ gets u → /jə/.
        return ("u", "ju") in taught_set
    if match.gpc.id == "DERIVED_INTERJECTION":
        return match.gpc.grapheme[0] in taught_graphemes
    if match.gpc.id.startswith("LETTER_NAME_"):
        return match.gpc.grapheme in taught_graphemes
    if match.gpc.id in {"DERIVED_X_VOICED", "DERIVED_XC", "DERIVED_XH"}:
        return ("x", "ks") in taught_set
    if match.gpc.id == "DERIVED_DROPPED_E_CE":
        return ("ce", "s") in taught_set
    if match.gpc.id == "DERIVED_DROPPED_E_GE":
        return ("ge", "dʒ") in taught_set
    if match.gpc.id == "DERIVED_VELAR_N":
        return ("n", "n") in taught_set
    if match.gpc.id == "DERIVED_GU":
        return {"g", "u"} <= taught_graphemes and ("g", "g") in taught_set
    if match.gpc.id.startswith("DERIVED_DOUBLE_"):
        singleton = (match.gpc.symbol[0], "".join(match.gpc.phonemes))
        return bool(taught_graphemes & _DOUBLING_CONSONANTS) and singleton in taught_set
    if match.gpc.id.startswith("DERIVED_REPEAT_FINAL_"):
        base = (match.gpc.symbol[:-1], "".join(match.gpc.phonemes))
        return bool(taught_graphemes & _DOUBLING_CONSONANTS) and base in taught_set
    if match.gpc.id == "DERIVED_AL":
        return ("all", "ɔl") in taught_set
    if match.gpc.id == "DERIVED_IE_FINAL":
        return ("y", "i") in taught_set
    if match.gpc.id == "DERIVED_IEW":
        return ("ew", "ju") in taught_set
    if match.gpc.id == "DERIVED_LK":
        return ("k", "k") in taught_set
    if match.gpc.id == "DERIVED_AWE":
        return ("aw", "ɔ") in taught_set
    if match.gpc.id == "DERIVED_OE_U":
        return ("oo", "u") in taught_set
    if match.gpc.id == "DERIVED_WH_H":
        return ("wh", "w") in taught_set
    if match.gpc.id == "DERIVED_AR_AIR":
        return bool({("air", "ɛr"), ("are", "ɛr"), ("ear", "ɛr")} & taught_set)
    if match.gpc.id == "DERIVED_IRE":
        return ("i-e", "aɪ") in taught_set and ("er", "ər") in taught_set
    if match.gpc.id == "DERIVED_SSION":
        return ("sion", "ʃən") in taught_set
    if match.gpc.id == "DERIVED_TION_CH":
        return ("tion", "ʃən") in taught_set
    if match.gpc.id == "DERIVED_GE_ZH":
        return ("g", "dʒ") in taught_set
    if match.gpc.id == "DERIVED_O_SHORT_U":
        return ("o-e", "ʌ") in taught_set
    if match.gpc.is_reduced_vowel:
        return match.gpc.grapheme in taught_graphemes
    return (match.gpc.symbol, "".join(match.gpc.phonemes)) in taught_set


def score_known_gpcs_english(
    word: str, student_knowledge: EnglishStudentKnowledge
) -> float:
    """Score a word based on whether the student can decode it.

    The score is 1.0 if at least one valid parse of the word uses only GPCs the
    student has been taught; 0.0 otherwise. Multiple parses can arise when prefix
    or suffix GPCs overlap with letter-by-letter GPCs.

    Reduced-vowel matches (parser-synthesized ``REDUCED_*`` GPCs from unstressed
    schwa/lax positions) count as taught whenever the student knows *any* GPC
    for the same grapheme letter — vowel reduction is a universal spoken-
    English process, not a separate correspondence the student needs taught.
    The exception is ``REDUCED_YOD_U`` (``u`` covering /j/ + a reduced vowel):
    its glide comes from the taught correspondence ``u → ju``, so it counts
    only when that exact pair is in the student's set.
    """
    taught_gpcs = student_knowledge.gpcs

    if not taught_gpcs:
        return 0.0

    try:
        variants = get_word_phonemes(word)
    except KeyError:
        return 0.0

    taught_set, taught_graphemes = normalize_taught_gpcs(taught_gpcs)

    for variant in variants:
        for parse in parse_gpcs(word, variant, GPC_INVENTORY):
            if all(
                _is_taught_match(match, taught_set, taught_graphemes) for match in parse
            ):
                return 1.0

    return 0.0


def score_sight_words_english(
    word: str, student_knowledge: EnglishStudentKnowledge
) -> float:
    """Score a word based on whether the student knows it as a sight word.

    The score is binary: if the student knows the word as a sight word, the score is 1.
    Otherwise, the score is 0.

    Affixed forms of known sight words also score 1.0 when the student has
    been taught the affix as a unit: ``friends`` passes for a student who
    knows ``friend`` and has ``-s`` in their affixes, ``walking`` for
    ``walk`` plus ``-ing``, ``preview`` for ``view`` plus ``pre-``. The
    affix unit is what was taught, so it covers the ending's allomorphs
    without a phoneme check on the affix. One affix at each edge can stack:
    ``unfriendly`` needs ``un-`` and ``-ly`` around ``friend``.

    Compounds score 1.0 too: a word that segments into chunks each of which
    is a known sight word or independently GPC-decodable — ``noone`` =
    ``no`` + ``one``, ``someone`` = ``some`` (decodable via s, o-e, m) +
    ``one``. At least one chunk must be a sight word; a compound of purely
    decodable non-sight chunks is left to the GPC scorer.
    """
    taught_sight_words = student_knowledge.sight_words

    if not taught_sight_words:
        return 0.0

    lowered = word.lower()
    if lowered in taught_sight_words:
        return 1.0

    base, apostrophe, clitic = lowered.partition("'")
    if (
        apostrophe
        and base in taught_sight_words
        and clitic
        in {
            "d",
            "ll",
            "m",
            "re",
            "s",
            "ve",
        }
    ):
        return 1.0

    taught_pairs, _ = normalize_taught_gpcs(student_knowledge.gpcs)
    if (
        _sight_word_stem(
            word.lower(), taught_sight_words, student_knowledge.affixes, taught_pairs
        )
        is not None
    ):
        return 1.0

    if _compound_segments(word.lower(), student_knowledge) is not None:
        return 1.0

    return 0.0


# Word endings a taught affix unit unlocks on a sight-word stem, and the
# affix entries (as marked in the profile data) that license each ending.
# Longer endings come first only for readability — every rule is tried.
_SUFFIX_MORPHEMES: tuple[tuple[str, frozenset[str]], ...] = (
    ("ies", frozenset({"-ies"})),
    ("ied", frozenset({"-ied"})),
    ("'s", frozenset({"-s"})),
    ("es", frozenset({"-es"})),
    ("ed", frozenset({"-ed"})),
    ("ing", frozenset({"-ing"})),
    ("est", frozenset({"-est"})),
    ("er", frozenset({"-er"})),
    ("ly", frozenset({"-ly"})),
    ("less", frozenset({"-less"})),
    ("ful", frozenset({"-ful"})),
    ("ness", frozenset({"-ness"})),
    ("ment", frozenset({"-ment"})),
    ("ish", frozenset({"-ish"})),
    ("able", frozenset({"-able"})),
    ("ible", frozenset({"-ible"})),
    ("s", frozenset({"-s"})),
)


# The ``-ed`` affix unit (taught at lesson 18) covers only the /əd/
# allomorph — the e is pronounced after t/d stems (``needed``). After other
# stems the e is silent (``walked`` /t/, ``saved`` /d/), which relies on the
# silent-e convention — those endings need the taught -ed→d / -ed→t pair
# (lesson 64), not the affix alone.
_ED_SCHWA_STEM_FINALS = frozenset("td")
_ED_SILENT_E_PAIRS = frozenset({("ed", "d"), ("ed", "t")})
_ED_SCHWA_PAIR = ("ed", "əd")

# Every affix string → the normalized (symbol, phonemes) pairs that license
# it — including unmarked siblings, since the marker doesn't change the
# correspondence (``s → z`` reads the ``-s`` ending in ``friends``). A
# correspondence decodes before the affix unit is officially taught, so a
# taught pair unlocks the same ending on a sight-word stem:
# ``("ing", "ɪŋ")`` at lesson 51 reads ``walk`` + ``ing`` well before the
# ``-ing`` affix is introduced at 65.
_AFFIX_PAIRS: dict[str, frozenset[tuple[str, str]]] = {
    grapheme: frozenset(
        (gpc.symbol, "".join(gpc.phonemes))
        for gpc in GPC_INVENTORY
        if gpc.symbol == symbol
    )
    for grapheme, symbol in {
        (gpc.grapheme, gpc.symbol)
        for gpc in GPC_INVENTORY
        if gpc.is_suffix or gpc.is_prefix
    }
}
_ALL_AFFIX_PAIRS = frozenset().union(*_AFFIX_PAIRS.values())


# Word openings a taught prefix unit unlocks on a sight-word stem. Prefixes
# don't trigger spelling adjustments at the boundary in English, so the stem
# is just the remainder.
_PREFIX_MORPHEMES: tuple[tuple[str, frozenset[str]], ...] = (
    ("dis", frozenset({"dis-"})),
    ("pre", frozenset({"pre-"})),
    ("tri", frozenset({"tri-"})),
    ("uni", frozenset({"uni-"})),
    ("un", frozenset({"un-"})),
    ("re", frozenset({"re-"})),
    ("bi", frozenset({"bi-"})),
)


def _suffix_stems(word: str) -> tuple[tuple[str, frozenset[str]], ...]:
    """Candidate ``(stem, licensing affixes)`` splits, with spelling adjustments."""
    candidates = []
    for ending, morphemes in _SUFFIX_MORPHEMES:
        if not word.endswith(ending) or len(word) <= len(ending):
            continue
        base = word[: -len(ending)]
        stems = [base, base + "e"]  # silent-e restoration: makes→make
        if len(base) >= 2 and base[-1] == base[-2]:
            stems.append(base[:-1])  # consonant doubling: running→run
        if ending in {"ies", "ied"}:
            stems.append(base + "y")  # cries→cry, tried→try
        candidates.extend((stem, morphemes) for stem in stems)
    return tuple(candidates)


def _prefix_stems(word: str) -> tuple[tuple[str, frozenset[str]], ...]:
    """Candidate ``(stem, licensing prefixes)`` splits at the word start."""
    return tuple(
        (word[len(prefix) :], morphemes)
        for prefix, morphemes in _PREFIX_MORPHEMES
        if word.startswith(prefix) and len(word) > len(prefix)
    )


def _sight_word_stem(
    word: str,
    sight_words: set[str],
    affixes: set[str],
    taught_pairs: set[tuple[str, str]],
    *,
    _depth: int = 2,
) -> str | None:
    """Resolve ``word`` to a known sight word by stripping taught affixes.

    Returns the sight-word stem when ``word`` is that stem wearing affix
    units the student was taught — ``walking`` → ``walk`` under ``-ing``,
    ``preview`` → ``view`` under ``pre-``. Returns ``None`` when no licensed
    split reaches a sight word. ``_depth`` bounds the recursion so a stem
    can carry one affix at each edge (``unfriendly`` → ``un-`` + ``friend``
    + ``-ly``) but no more.
    """
    if _depth <= 0 or not (affixes or taught_pairs & _ALL_AFFIX_PAIRS):
        return None
    for stem, morphemes in (*_suffix_stems(word), *_prefix_stems(word)):
        if "-ed" in morphemes:
            # The /əd/ reading (e pronounced, t/d stems) is licensed by the
            # -ed affix or the -ed→əd pair; the silent-e /d/,/t/ readings
            # need the -ed affix or an -ed→d/-ed→t pair.
            ed_pairs = (
                _ED_SILENT_E_PAIRS | {_ED_SCHWA_PAIR}
                if stem[-1:] in _ED_SCHWA_STEM_FINALS
                else _ED_SILENT_E_PAIRS
            )
            licensed = "-ed" in affixes or bool(taught_pairs & ed_pairs)
        else:
            licensed = bool(morphemes & affixes) or any(
                taught_pairs & _AFFIX_PAIRS.get(morpheme, frozenset())
                for morpheme in morphemes
            )
        if not licensed:
            continue
        if stem in sight_words:
            return stem
        inner = _sight_word_stem(
            stem, sight_words, affixes, taught_pairs, _depth=_depth - 1
        )
        if inner is not None:
            return inner
    return None


def _compound_segments(
    word: str, student_knowledge: EnglishStudentKnowledge
) -> list[str] | None:
    """Segment ``word`` into known sight words and decodable chunks.

    Returns the segmentation — a list of two or more contiguous chunks where
    every chunk is either a taught sight word or a string
    ``score_known_gpcs_english`` decodes on its own — when at least one chunk
    is a sight word; ``None`` otherwise. ``noone`` → ``["no", "one"]``,
    ``someone`` → ``["some", "one"]`` (``some`` decodes via s, o-e, m).

    The split is orthographic, not phonetic: a sight-word chunk is credited
    as a whole memorized unit regardless of how its letters would sound out
    inside the compound.
    """
    sight_words = student_knowledge.sight_words
    if not any(sight_word in word for sight_word in sight_words):
        return None

    def _is_sight_form(chunk: str) -> bool:
        if chunk in sight_words:
            return True
        stem = _sight_word_stem(
            chunk,
            sight_words,
            student_knowledge.affixes,
            normalize_taught_gpcs(student_knowledge.gpcs)[0],
        )
        return stem is not None and (len(stem) >= 3 or "'" in chunk)

    for split in range(2, len(word) - 1):
        chunks = [word[:split], word[split:]]
        sight_forms = [_is_sight_form(chunk) for chunk in chunks]
        if not any(sight_forms):
            continue
        if not all(
            is_sight or score_known_gpcs_english(chunk, student_knowledge) == 1.0
            for chunk, is_sight in zip(chunks, sight_forms, strict=True)
        ):
            continue
        short_chunks = {chunk for chunk in chunks if len(chunk) < 3}
        prefix_chunks = {prefix for prefix, _ in _PREFIX_MORPHEMES}
        if short_chunks and not all(sight_forms) and not short_chunks <= prefix_chunks:
            continue
        return chunks
    return None

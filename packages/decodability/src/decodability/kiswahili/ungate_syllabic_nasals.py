"""Cluster analysis for ungating Kiswahili syllabic nasals.

A stop-gap: the tool treats a word-initial syllabic nasal (the ``m-``
of ``mti``) as part of a consonant cluster, so words that a student can actually
read get labelled nondecodable. The helpers here decide, per word and per student
profile, whether that gate should be lifted.
"""

from decodability.kiswahili import KiswahiliStudentKnowledge
from decodability.kiswahili.segment import (
    ClusterSpan,
    GraphemeSpan,
    WordDecomposition,
)

# words identified as having syllabic nasals
TO_UNGATE = [
    "akamsalimia",
    "akamtuma",
    "alimnunulia",
    "alimpeleka",
    "alimpenda",
    "hakumpenda",
    "humpeleka",
    "inamlinda",
    "kumchapa",
    "kumfukuza",
    "kumkaribisha",
    "mbunge",
    "mbwa",
    "mchana",
    "mchanganyiko",
    "mchele",
    "mchezo",
    "mchi",
    "mchwa",
    "mdogo",
    "mdomo",
    "mdudu",
    "mfalme",
    "mfano",
    "mfupi",
    "mgeni",
    "mgonjwa",
    "mheshimiwa",
    "mjini",
    "mjomba",
    "mkali",
    "mkate",
    "mkono",
    "mkononi",
    "mkubwa",
    "mkuu",
    "mlango",
    "mlingoti",
    "mmoja",
    "mpeketoni",
    "mpunga",
    "mraba",
    "mrefu",
    "msafiri",
    "msalani",
    "msichana",
    "msingi",
    "msituni",
    "mstatili",
    "mti",
    "mtihani",
    "mtini",
    "mtoto",
    "mtu",
    "mvua",
    "mzazi",
    "mzee",
    "nchi",
    "nne",
    "wamsalimie",
    "wanamjua",
    "zilizompatia",
    "nzi",
    "kuamka",
    "kiamshakinywa",
    "kuzungumza",
    "namna",
    "umri",
    "huteremsha",
    "nje",
]

# n fuses in front of these anywhere in a word, unless the word starts with the
# cluster and only one syllable follows (nzi, nje)
N_FUSED_AFTER = frozenset({"d", "j", "z", "g"})

# m/n + w or y is a glide cluster, so gated
GLIDES = frozenset({"w", "y"})


def _following_grapheme(
    word_decomposition: WordDecomposition, cluster: ClusterSpan
) -> GraphemeSpan:
    """The grapheme straight after the cluster's nasal, e.g. "ch" in "mchwa".

    The nasal is always the single letter m or n, so the next grapheme starts one
    character after the cluster does.
    """
    return next(
        grapheme
        for grapheme in word_decomposition.graphemes
        if grapheme.start == cluster.start + 1
    )


def _syllables_after(
    word_decomposition: WordDecomposition, cluster: ClusterSpan
) -> int:
    """Syllables (vowel nuclei) after the cluster's nasal, e.g. 1 for "nje"."""
    return sum(
        1
        for grapheme in word_decomposition.graphemes
        if not grapheme.is_consonant and grapheme.start > cluster.start
    )


def is_syllabic_nasal(
    word_decomposition: WordDecomposition, cluster: ClusterSpan
) -> bool:
    """Determine if the nasal that starts the given cluster is syllabic.

    Applies the linguists' rule tables in order, first match wins. Rules that need
    a lexicon or a verb parser aren't supported yet, as noted inline.

    ASSUMPTION: "monosyllabic root" is approximated as exactly one syllable after
    the nasal. Suffixed forms (e.g. locative -ni in "nzini") aren't stripped yet.
    """
    if not cluster.nasal:
        return False
    nasal = cluster.normalised[0]
    following = _following_grapheme(word_decomposition, cluster).symbol
    syllable_count = _syllables_after(word_decomposition, cluster)
    at_start = cluster.start == 0

    # Fused with glides
    if following in GLIDES:
        return False

    # Word-initial n with one syllable after it is syllabic (nzi, nje)
    if at_start and syllable_count == 1:
        return True

    if nasal == "m":
        if following in ("b", "v"):
            # Medial mb/mv is fused (nyumba).
            # An object marker before b/v (akambeba) needs a verb parser
            # Word-initial mb/mv stays fused until we have support to
            # distinguish syllabic cases
            return False
        # m + any other consonant is syllabic
        return True
    elif nasal == "n":
        # N is fused before d, j, z, g; syllabic before anything else
        return following not in N_FUSED_AFTER

    return False


def ok_to_ungate(
    word_decomposition: WordDecomposition, profile: KiswahiliStudentKnowledge
) -> bool:
    """Returns True if the word can be ungated based on the student's knowledge
    profile.
    """
    ungate_word = False
    word_graphemes = {g.symbol for g in word_decomposition.graphemes}
    if not word_graphemes.issubset(profile.graphemes):
        # if the word contains graphemes that the student doesn't know, it can't be
        # ungated
        return False

    for cluster_span in word_decomposition.clusters:
        # iterate through all consonant clusters in the potential word to ungate
        if is_syllabic_nasal(word_decomposition, cluster_span):
            ungate_word = True
        elif cluster_span.normalised not in profile.all_known_clusters:
            # student doesn't know this cluster, so we can't ungate the word
            # this is a sufficient condition keep the word nondecodable
            return False
    return ungate_word


__all__ = [
    "GLIDES",
    "N_FUSED_AFTER",
    "TO_UNGATE",
    "is_syllabic_nasal",
    "ok_to_ungate",
]

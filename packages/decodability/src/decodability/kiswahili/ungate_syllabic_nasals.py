"""Cluster analysis for ungating Kiswahili syllabic nasals.

A stop-gap: the tool treats a word-initial syllabic nasal (the ``m-``
of ``mti``) as part of a consonant cluster, so words that a student can actually
read get labelled nondecodable. The helpers here decide, per word and per student
profile, whether that gate should be lifted.
"""

from decodability.kiswahili import KiswahiliStudentKnowledge
from decodability.kiswahili.segment import ClusterSpan, WordDecomposition

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

# word-initial n fuses in front of these, always
N_FUSED_AFTER = frozenset({"d", "j", "z", "g"})

# m/n + w or y is a glide cluster, so gated
GLIDES = frozenset({"w", "y"})


def is_syllabic_nasal(cluster: ClusterSpan) -> bool:
    """Determine if the given cluster is a syllabic nasal.
    Not yet a generalised check."""
    if not cluster.nasal:
        return False
    if cluster.text[0] == "m":
        if cluster.text[1] not in GLIDES:
            return True
    elif cluster.text[0] == "n":
        if cluster.text[1] not in N_FUSED_AFTER.union(GLIDES):
            return True
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
        if is_syllabic_nasal(cluster_span):
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

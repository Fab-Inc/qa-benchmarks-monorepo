"""
A stop-gap solution for ungating syllabic nasals in benchmark items.

Given a list of words with syllabic nasals in our dataset, this script
will compare them against the student knowledge to determine which
words should be _decodable_ (i.e. dropped from nondecodable words list.)

Usage:

    uv run --frozen --package phonics_benchmark \
        benchmarks/phonics_benchmark/scripts/drop_syllabic_nasals.py \
        <path_to_load_items> \
        <path_to_save_items>


"""

import json
import logging
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import typer
from decodability.kiswahili import KiswahiliStudentKnowledge, extract_words_kiswahili
from decodability.kiswahili.definitions import get_clusters, get_graphemes
from decodability.kiswahili.segment import GraphemeSpan, decompose_word
from phonics_benchmark.labeling import get_vocabulary_set

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


@dataclass
class ConsonantCluster:
    """A consonant cluster within a word with analysis"""

    cluster: str
    graphemes: set[str]
    syllabic_nasal: bool


@dataclass
class WordClusterAnalysis:
    """Cluster analysis of a word - all clusters and their analyses"""

    word: str
    clusters: list[ConsonantCluster]


def count_syllables(decomposition: list[GraphemeSpan]) -> int:
    """Syllable count by vowel nuclei."""
    return sum(1 for grapheme in decomposition if not grapheme.is_consonant)


def has_monosyllabic_root(decomposition: list[GraphemeSpan]) -> bool:
    """True when stripping the leading consonant leaves a one-syllable root."""
    assert decomposition[0].is_consonant, "The first grapheme must be a consonant"
    return count_syllables(decomposition[1:]) == 1


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_as_jsonl(items: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item))
            f.write("\n")


def get_words_to_ungate(words: list[str]) -> list[WordClusterAnalysis]:
    """Returns a list of WordClusterAnalysis for words that have syllabic nasals."""
    words_to_ungate = []
    for word in words:
        cluster_strings = get_clusters(word)
        clusters = []

        for cluster in cluster_strings:
            decomposition = decompose_word(cluster)
            syllabic_nasal = False
            if decomposition[0].symbol == "m":
                # manually identified all m- clusters that should be ungated already.
                # There shouldn't be any glides but we check here anyway
                if decomposition[1].symbol not in GLIDES:
                    syllabic_nasal = True
            elif decomposition[0].symbol == "n":
                if decomposition[1].symbol not in N_FUSED_AFTER.union(GLIDES):
                    syllabic_nasal = True
                elif has_monosyllabic_root(decomposition):
                    syllabic_nasal = True

            clusters.append(
                ConsonantCluster(cluster, set(get_graphemes(cluster)), syllabic_nasal)
            )

        words_to_ungate.append(WordClusterAnalysis(word=word, clusters=clusters))

    return words_to_ungate


def ok_to_ungate(word_cluster_analysis: WordClusterAnalysis, profile) -> bool:
    """Returns True if the word can be ungated based on the student's knowledge
    profile.
    """
    ungate_word = False
    for cluster_analysis in word_cluster_analysis.clusters:
        # iterate through all consonant clusters in the potential word to ungate
        if cluster_analysis.syllabic_nasal:
            if cluster_analysis.graphemes.issubset(profile.graphemes):
                # student knows all graphemes in this cluster, then ungate the word
                ungate_word = True
        elif cluster_analysis.cluster not in profile.all_known_clusters:
            # student doesn't know this cluster, so we can't ungate the word
            # this is a sufficient condition keep the word nondecodable
            return False
    return ungate_word


@typer.run
def main(load_items_path: str, save_items_path: str, log_level: str = "INFO") -> None:
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO))

    words_to_ungate = get_words_to_ungate(TO_UNGATE)
    logging.info(
        "Words to ungate: \n" + "\n".join(str(word) for word in words_to_ungate)
    )

    items = load_jsonl(Path(load_items_path))
    logging.info(f"Loaded {len(items)} items from {load_items_path}")

    n_items_with_removal = 0
    n_removed_words = 0

    for item in items:
        label = item["nondecodable_words"].copy()
        removed = []
        profile = KiswahiliStudentKnowledge.model_validate(item["student_knowledge"])

        for labelled_word, word_to_ungate in product(label, words_to_ungate):
            if labelled_word == word_to_ungate.word:
                # these are the words we _might_ need to drop from nondecodable list
                # this is only valid if the student already knows all graphemes and
                # any other gated clusters.
                if ok_to_ungate(word_to_ungate, profile):
                    label.remove(labelled_word)
                    removed.append(labelled_word)

        if removed:
            n_items_with_removal += 1
            n_removed_words += len(removed)
            logging.info(
                f"Removed from {item['benchmark_item_id']}'s nondecodable words: "
                f"{removed}"
            )

            item["nondecodable_words"] = label
            item["num_nondecodable_words"] = len(label)
            unique_words = {w.lower() for w in extract_words_kiswahili(item["content"])}
            item["nondecodable_ratio"] = len(label) / len(unique_words)

    save_as_jsonl(items, Path(save_items_path))
    logging.info(f"Saved updated items to {save_items_path}")
    logging.info(f"Number of items with removal: {n_items_with_removal}")
    logging.info(f"Number of removed words: {n_removed_words}")

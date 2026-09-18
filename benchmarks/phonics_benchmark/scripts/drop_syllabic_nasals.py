"""
A stop-gap solution for ungating syllabic nasals in benchmark items.

Given a list of words with syllabic nasals in our dataset, this script
will compare them against the student knowledge to determine which
words should be _decodable_ (i.e. dropped from nondecodable words list.)

It also double-checks every word to include any accidentally dropped
nondecodable words.

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

import pandas as pd
import typer
from decodability.kiswahili import KiswahiliStudentKnowledge, extract_words_kiswahili
from decodability.kiswahili.analyse import is_decodable_kenya_tusome
from decodability.kiswahili.definitions import get_clusters, get_graphemes
from decodability.kiswahili.segment import GraphemeSpan, decompose_word
from phonics_benchmark.data_io import convert_records_to_dataframe
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
    graphemes: set[str]
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
        graphemes = set(get_graphemes(word))
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

        words_to_ungate.append(
            WordClusterAnalysis(word=word, graphemes=graphemes, clusters=clusters)
        )

    return words_to_ungate


def ok_to_ungate(
    word_cluster_analysis: WordClusterAnalysis, profile: KiswahiliStudentKnowledge
) -> bool:
    """Returns True if the word can be ungated based on the student's knowledge
    profile.
    """
    ungate_word = False
    if not word_cluster_analysis.graphemes.issubset(profile.graphemes):
        # if the word contains graphemes that the student doesn't know, it can't be
        # ungated
        return False

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

    items_df = pd.read_json(load_items_path, lines=True)
    items_df["vocab_set"] = get_vocabulary_set(items_df, "sw")
    logging.info(f"Loaded {len(items_df)} items from {load_items_path}")

    n_items_with_removal = 0
    n_items_with_addition = 0
    n_removed_words = 0
    n_added_words = 0

    for _, item in items_df.iterrows():
        label = item["nondecodable_words"].copy()
        removed = []
        added = []
        profile = KiswahiliStudentKnowledge.model_validate(item["student_knowledge"])

        for word_in_text, word_to_ungate in product(item["vocab_set"], words_to_ungate):
            if word_in_text == word_to_ungate.word:
                # these are the words we _might_ need to drop from nondecodable list
                # this is only valid if the student already knows all graphemes and
                # any other gated clusters.
                if (
                    not is_decodable_kenya_tusome(word_in_text, profile)
                    and word_in_text not in label
                ):
                    label.append(word_in_text)
                    added.append(word_in_text)
                elif ok_to_ungate(word_to_ungate, profile) and word_in_text in label:
                    label.remove(word_in_text)
                    removed.append(word_in_text)

        if removed or added:
            if removed:
                n_items_with_removal += 1
                n_removed_words += len(removed)
                logging.info(
                    f"Removed from {item['benchmark_item_id']}'s nondecodable words: "
                    f"{removed}"
                )

            if added:
                n_items_with_addition += 1
                n_added_words += len(added)
                logging.info(
                    f"Added to {item['benchmark_item_id']}'s nondecodable words: "
                    f"{added}"
                )

            item["nondecodable_words"] = label
            item["num_nondecodable_words"] = len(label)
            item["nondecodable_ratio"] = len(label) / len(item["vocab_set"])

    items_df.drop(["vocab_set"], axis=1, inplace=True)
    save_as_jsonl(items_df.to_dict(orient="records"), Path(save_items_path))
    logging.info(f"Saved updated items to {save_items_path}")
    logging.info(f"Number of items with removal: {n_items_with_removal}")
    logging.info(f"Number of removed words: {n_removed_words}")
    logging.info(f"Number of items with addition: {n_items_with_addition}")
    logging.info(f"Number of added words: {n_added_words}")

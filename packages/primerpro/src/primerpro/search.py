"""Search helpers ported from PrimerPro's word-list and text-data searches."""

from __future__ import annotations

from .model import (
    HIGHLIGHT_OFF,
    HIGHLIGHT_ON,
    GraphemeInventory,
    Settings,
    TextData,
    WordList,
)


class Search:
    """Base result container shared by PrimerPro search objects."""

    def __init__(self, search_id: int, settings: Settings) -> None:
        self.search_id = search_id
        self.settings = settings
        self.search_results = ""
        self.search_count = 0


class BuildableWordSearchWL(Search):
    """Find word-list entries buildable from a taught grapheme set.

    ``graphemes`` is the taught list. ``highlights`` optionally wraps matching
    result words in PrimerPro highlight tags.
    """

    def __init__(self, search_id: int, settings: Settings) -> None:
        super().__init__(search_id, settings)
        self.graphemes: list[str] = []
        self.highlights: list[str] = []

    def execute_buildable_search(self, word_list: WordList) -> None:
        rows = []
        count = 0
        for index, word in enumerate(word_list.iter_words()):
            if word.is_buildable_word(self.graphemes):
                count += 1
                rows.append(word_list.get_display_line_for_word(index, self.highlights))
        self.search_count = count
        self.search_results = "\n".join(rows) if rows else "***No Results***"


class BuildableWordSearchTD(Search):
    """Find buildable words in parsed text data.

    ``no_duplicates`` reports only the first occurrence of each display word.
    ``para_format`` preserves paragraph order and highlights buildable words in
    context instead of returning a plain buildable-word list.
    """

    def __init__(self, search_id: int, settings: Settings) -> None:
        super().__init__(search_id, settings)
        self.graphemes: list[str] = []
        self.highlights: list[str] = []
        self.no_duplicates = False
        self.para_format = False

    def execute_buildable_word_search(self, text_data: TextData) -> None:
        if self.para_format:
            self._execute_paragraph_format(text_data)
            return

        rows = []
        seen = set()
        count = 0
        for word in text_data.iter_words():
            if self.no_duplicates and word.display_word in seen:
                continue
            if word.is_buildable_word(self.graphemes):
                count += 1
                seen.add(word.display_word)
                rows.append(word.display_word)
        self.search_count = count
        self.search_results = "\n".join(rows) if rows else "***No Results***"

    def _execute_paragraph_format(self, text_data: TextData) -> None:
        rows = []
        count = 0
        for paragraph in text_data.paragraphs or []:
            words = []
            for sentence in paragraph.sentences:
                for word in sentence.words:
                    if word.is_buildable_word(self.graphemes):
                        count += 1
                        words.append(
                            f"{HIGHLIGHT_ON}{word.display_word}{HIGHLIGHT_OFF}"
                        )
                    else:
                        words.append(word.display_word)
            rows.append(" ".join(words))
        self.search_count = count
        self.search_results = "\n".join(rows)


class ResidueSearch(Search):
    """Report words in text data that are not buildable from taught graphemes.

    Residue output highlights exact untaught graphemes. ``ignore_sight_words``
    applies PrimerPro's sight-word exception before residue is counted.
    """

    def __init__(self, search_id: int, settings: Settings) -> None:
        super().__init__(search_id, settings)
        self.graphemes: list[str] = []
        self.ignore_sight_words = False
        self.reading_level_info = False

    def execute_residue_search(self, text_data: TextData) -> None:
        rows = []
        count = 0
        for word in text_data.iter_words():
            if self.ignore_sight_words and word.is_sight_word():
                continue
            if not word.is_buildable_word(self.graphemes):
                count += 1
                rows.append(word.highlight_missing_graphemes(self.graphemes))
        if self.reading_level_info:
            rows.extend(
                [
                    f"Number of words in story: {text_data.word_count()}",
                    f"Number of sentences in story: {text_data.sentence_count()}",
                ]
            )
        self.search_count = count
        self.search_results = "\n".join(rows) if rows else "***No Results***"


class MinPairsSearch(Search):
    """Find word-list pairs differing only by ``grapheme1`` to ``grapheme2``."""

    def __init__(self, search_id: int, settings: Settings) -> None:
        super().__init__(search_id, settings)
        self.grapheme1 = ""
        self.grapheme2 = ""

    def execute_min_pairs_search(self, word_list: WordList) -> None:
        inventory = self.settings.grapheme_inventory
        if inventory is None:
            return
        grapheme1 = inventory.get_grapheme(self.grapheme1)
        grapheme2 = inventory.get_grapheme(self.grapheme2)
        if grapheme1 is None or grapheme2 is None:
            return
        rows = []
        words = word_list.iter_words()
        for left_index, left in enumerate(words):
            for right in words:
                if left is right:
                    continue
                if left.is_minimal_pair_for_graphemes(
                    right,
                    False,
                    grapheme1,
                    grapheme2,
                ):
                    rows.append(f"{left.display_word}\t{right.display_word}")
        self.search_count = len(rows)
        self.search_results = "\n".join(rows) if rows else "***No Results***"


class ToneChartSearch(Search):
    """List configured tone graphemes with tone-bearing units and levels."""

    def execute_tone_chart(self, inventory: GraphemeInventory) -> None:
        rows = ["Tone\tTBU\tLevel"]
        for tone in inventory.tones:
            tbu = tone.tone_bearing_unit.symbol if tone.tone_bearing_unit else ""
            rows.append(f"{tone.symbol}\t{tbu}\t{tone.level}")
        self.search_count = len(inventory.tones)
        self.search_results = "\n".join(rows)


class TonePairsSearch(Search):
    """Find tone-marked words that have a tone-stripped counterpart."""

    def __init__(self, search_id: int, settings: Settings) -> None:
        super().__init__(search_id, settings)
        self.grapheme = ""

    def execute_tone_pairs_search(self, word_list: WordList) -> None:
        inventory = self.settings.grapheme_inventory
        if inventory is None:
            return
        tone_index = inventory.find_tone_index(self.grapheme)
        if tone_index < 0:
            return
        tone = inventory.get_tone(tone_index)
        rows = []
        words = word_list.iter_words()
        for word in words:
            if not word.is_tone_contain_in_word(tone.symbol):
                continue
            plain = word.get_word_without_tone()
            for candidate in words:
                if candidate.display_word == plain:
                    rows.append(f"{candidate.display_word}\t{word.display_word}")
                    break
        self.search_count = len(rows)
        self.search_results = "\n".join(rows) if rows else "***No Results***"

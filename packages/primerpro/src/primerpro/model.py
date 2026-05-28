"""Object model for the Python port of PrimerPro decodability behavior.

PrimerPro models text for primer development by segmenting words into a
project-defined grapheme inventory, then checking those graphemes against a
taught list. The classes here mirror the original C# objects closely enough to
preserve its search, residue, frequency, tone, and minimal-pair behavior.
"""

from __future__ import annotations

from dataclasses import dataclass

HIGHLIGHT_ON = "<HL>"
HIGHLIGHT_OFF = "</HL>"


@dataclass
class OptionSettings:
    """Project parsing and display options used by PrimerPro objects."""

    max_size_grapheme: int = 4
    general_punct: str = " ,:;-\"'"
    ending_punct: str = ".?!"
    import_ignore_chars: str = ""
    cv_cns: str = "C"
    cv_vwl: str = "V"
    cv_tone: str = "T"
    cv_syllograph: str = "S"


class LocalizationTable:
    """Placeholder localization table matching the original API shape."""

    def get_message(self, _key: str) -> str:
        return ""


class Settings:
    """Container for the project objects needed by PrimerPro operations."""

    def __init__(self) -> None:
        self.option_settings = OptionSettings()
        self.localization_table = LocalizationTable()
        self.grapheme_inventory: GraphemeInventory | None = None
        self.graphemes_taught: GraphemeTaughtOrder | None = None
        self.sight_words: SightWords | None = None
        self.text_data: TextData | None = None
        self.word_list: WordList | None = None


class Grapheme:
    """A project-defined written unit used as PrimerPro's atomic token.

    A grapheme may be a single character, a multigraph such as ``sh`` or
    ``igh``, a tone-bearing written form, or a syllograph. All decodability and
    search behavior follows the configured inventory rather than raw Unicode
    characters.
    """

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self.uppercase = ""
        self.is_consonant = False
        self.is_vowel = False
        self.is_tone = False
        self.is_syllograph = False
        self.is_syllabic_consonant = False
        self.is_complex = False
        self.count_in_word_list = 0
        self.count_in_text_data = 0

    def is_same(self, other: Grapheme | None) -> bool:
        """Return whether another grapheme has the same non-empty symbol."""

        return other is not None and self.symbol != "" and self.symbol == other.symbol

    def get_marker(self) -> str:
        """Return the CV-pattern marker for this grapheme type."""

        if self.is_consonant:
            return "C"
        if self.is_vowel:
            return "V"
        if self.is_syllograph:
            return "S"
        if self.is_tone:
            return "T"
        return "?"

    def init_count_in_text_data(self) -> None:
        self.count_in_text_data = 0

    def incr_count_in_text_data(self) -> None:
        self.count_in_text_data += 1


class Consonant(Grapheme):
    def __init__(self, symbol: str) -> None:
        super().__init__(symbol)
        self.is_consonant = True
        self.is_syllabic = False
        self.is_aspirated = False
        self.is_velarized = False
        self.is_palatalized = False
        self.is_labialized = False
        self.is_prenasalized = False


class Vowel(Grapheme):
    def __init__(self, symbol: str) -> None:
        super().__init__(symbol)
        self.is_vowel = True
        self.is_long = False
        self.is_nasal = False


class Tone(Grapheme):
    """Tone grapheme with a level and optional tone-bearing unit."""

    def __init__(self, symbol: str) -> None:
        super().__init__(symbol)
        self.is_tone = True
        self.level = ""
        self.tone_bearing_unit: Grapheme | None = Grapheme("")
        self.teaching_order = 0

    def matches_level(self, level: str) -> bool:
        return self.level == level

    def matches_tone_bearing_unit(self, grapheme: Grapheme) -> bool:
        """Match PrimerPro's original tone-bearing-unit predicate.

        The C# method compares the supplied grapheme to the tone grapheme itself,
        not to ``tone_bearing_unit``. The port preserves that behavior.
        """

        return self.is_same(grapheme)

    def get_marker(self) -> str:
        """Return the CV marker of this tone's tone-bearing unit."""

        if self.tone_bearing_unit is None:
            return ""
        if self.tone_bearing_unit.is_consonant:
            return "C"
        if self.tone_bearing_unit.is_vowel:
            return "V"
        if self.tone_bearing_unit.is_syllograph:
            return "S"
        return ""


class Syllograph(Grapheme):
    def __init__(self, symbol: str) -> None:
        super().__init__(symbol)
        self.is_syllograph = True


class GraphemeInventory:
    """Configured grapheme inventory that drives PrimerPro tokenization.

    Words and syllables are segmented by longest inventory match up to
    ``OptionSettings.max_size_grapheme``. Inventory membership therefore
    determines whether strings such as ``ng`` or ``sh`` are one grapheme or two.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings
        self.consonants: list[Consonant] = []
        self.vowels: list[Vowel] = []
        self.tones: list[Tone] = []
        self.syllographs: list[Syllograph] = []

    def add_consonant(self, consonant: Consonant) -> None:
        self.consonants.append(consonant)

    def add_vowel(self, vowel: Vowel) -> None:
        self.vowels.append(vowel)

    def add_tone(self, tone: Tone) -> None:
        self.tones.append(tone)

    def add_syllograph(self, syllograph: Syllograph) -> None:
        self.syllographs.append(syllograph)

    def consonant_count(self) -> int:
        return len(self.consonants)

    def vowel_count(self) -> int:
        return len(self.vowels)

    def tone_count(self) -> int:
        return len(self.tones)

    def syllograph_count(self) -> int:
        return len(self.syllographs)

    def get_consonant(self, index: int) -> Consonant:
        return self.consonants[index]

    def get_vowel(self, index: int) -> Vowel:
        return self.vowels[index]

    def get_tone(self, index: int) -> Tone:
        return self.tones[index]

    def get_syllograph(self, index: int) -> Syllograph:
        return self.syllographs[index]

    def find_consonant_index(self, symbol: str) -> int:
        return _find_index(self.consonants, symbol)

    def find_vowel_index(self, symbol: str) -> int:
        return _find_index(self.vowels, symbol)

    def find_tone_index(self, symbol: str) -> int:
        return _find_index(self.tones, symbol)

    def find_syllograph_index(self, symbol: str) -> int:
        return _find_index(self.syllographs, symbol)

    def is_in_inventory(self, symbol: str) -> bool:
        return self.get_grapheme(symbol) is not None

    def get_grapheme(self, symbol: str) -> Grapheme | None:
        """Return any inventory grapheme matching a symbol or uppercase form."""

        for grapheme in self._all_graphemes():
            if symbol == grapheme.symbol or symbol == grapheme.uppercase:
                return grapheme
        return None

    def reset_text_counts(self) -> None:
        for grapheme in self._all_graphemes():
            grapheme.init_count_in_text_data()

    def _all_graphemes(self) -> list[Grapheme]:
        return [*self.consonants, *self.vowels, *self.tones, *self.syllographs]


def _find_index(graphemes: list[Grapheme], symbol: str) -> int:
    for index, grapheme in enumerate(graphemes):
        if symbol == grapheme.symbol or symbol == grapheme.uppercase:
            return index
    return -1


class Syllable:
    """A syllable segmented into PrimerPro graphemes and a CV pattern."""

    def __init__(self, text: str, settings: Settings) -> None:
        self.settings = settings
        self.display_syllable = text
        self.graphemes = _decompose(text, settings)
        self.cv_pattern = self._build_cv_pattern()

    def grapheme_count(self) -> int:
        return len(self.graphemes)

    def get_grapheme(self, index: int) -> Grapheme | None:
        if 0 <= index < len(self.graphemes):
            return self.graphemes[index]
        return None

    def get_grapheme_without_tone(self, index: int) -> Grapheme | None:
        """Return a grapheme with tone substituted by its tone-bearing unit."""

        grapheme = self.get_grapheme(index)
        return _without_tone(grapheme, self.settings)

    def get_syllable_in_lower_case(self) -> str:
        return "".join(grapheme.symbol for grapheme in self.graphemes)

    def is_open_syllable(self) -> bool:
        """Return whether the syllable ends in a vowel after tone stripping."""

        grapheme = self.get_grapheme_without_tone(self.grapheme_count() - 1)
        return bool(grapheme and grapheme.is_vowel)

    def is_closed_syllable(self) -> bool:
        """Return whether the syllable ends in a consonant after tone stripping."""

        grapheme = self.get_grapheme_without_tone(self.grapheme_count() - 1)
        return bool(grapheme and grapheme.is_consonant)

    def is_onset(self, grapheme: str) -> bool:
        first = self.get_grapheme(0)
        return bool(first and first.symbol == grapheme)

    def is_coda(self, grapheme: str) -> bool:
        """Return whether ``grapheme`` is the final grapheme of the syllable.

        Linguistic codas can be clusters. PrimerPro's original method uses the
        simpler final-grapheme test, and the port preserves that behavior.
        """

        last = self.get_grapheme(self.grapheme_count() - 1)
        return bool(last and last.symbol == grapheme)

    def is_syllable_initial(self, grapheme: str) -> bool:
        return self.is_onset(grapheme)

    def is_syllable_medial(self, grapheme: str) -> bool:
        """Return whether text appears between first and final graphemes.

        This mirrors PrimerPro's original substring check over the medial
        grapheme symbols, so a query can match inside a multigraph.
        """

        medial = "".join(grf.symbol for grf in self.graphemes[1:-1])
        return grapheme in medial

    def is_syllable_final(self, grapheme: str) -> bool:
        return self.is_coda(grapheme)

    def is_in_syllable(self, grapheme: str) -> bool:
        """Return whether ``grapheme`` occurs in the syllable after tone stripping."""

        return any(
            grf is not None and grf.symbol == grapheme
            for grf in (
                self.get_grapheme_without_tone(i) for i in range(len(self.graphemes))
            )
        )

    def is_buildable(self, graphemes_taught: list[str]) -> bool:
        """Return whether every grapheme is allowed by the taught list.

        PrimerPro supports positional taught forms: ``t_`` can satisfy syllable
        initial ``t``, and ``_t`` can satisfy syllable-final ``t``.
        """

        for index, grapheme in enumerate(self.graphemes):
            if not _matches_taught_position(
                grapheme.symbol,
                graphemes_taught,
                index,
                len(self.graphemes),
            ):
                return False
        return True

    def _build_cv_pattern(self) -> str:
        return "".join(
            _cv_marker(grapheme, self.settings) for grapheme in self.graphemes
        )


class Word:
    """A cleaned, segmented PrimerPro word.

    The word stores its original text, display text, grapheme sequence, inferred
    or explicit syllables, CV pattern, and sort key. Its behavior follows the
    original PrimerPro word object, including punctuation cleanup and longest
    inventory-match segmentation.
    """

    syllable_break_character = "."

    def __init__(self, text: str, settings: Settings) -> None:
        self.settings = settings
        self.orig_word = text
        self.graphemes: list[Grapheme] = []
        self.syllables: list[Syllable] = []
        self.display_word = ""
        self.cv_pattern = ""
        self.available = True
        self.key = ""
        self._build_word()
        self.key = self.get_key()

    def grapheme_count(self) -> int:
        return len(self.graphemes)

    def syllable_count(self) -> int:
        return len(self.syllables)

    def get_grapheme(self, index: int) -> Grapheme | None:
        if 0 <= index < len(self.graphemes):
            return self.graphemes[index]
        return None

    def get_syllable(self, index: int) -> Syllable | None:
        if 0 <= index < len(self.syllables):
            return self.syllables[index]
        return None

    def get_grapheme_without_tone(self, index: int) -> Grapheme | None:
        """Return a grapheme with tone substituted by its tone-bearing unit."""

        grapheme = self.get_grapheme(index)
        return _without_tone(grapheme, self.settings)

    def get_word_without_tone(self) -> str:
        """Return display text with tone graphemes replaced by their TBUs."""

        symbols = []
        for index in range(self.grapheme_count()):
            grapheme = self.get_grapheme(index)
            without_tone = self.get_grapheme_without_tone(index)
            symbols.append((without_tone or grapheme).symbol)
        return "".join(symbols)

    def get_word_in_lower_case(self) -> str:
        return "".join(grapheme.symbol for grapheme in self.graphemes)

    def get_cv_shape_of_word(self) -> str:
        """Return the consonant/vowel shape of the word's grapheme sequence."""

        return "".join(
            _cv_marker(self.get_grapheme_without_tone(index), self.settings)
            for index in range(self.grapheme_count())
        )

    def get_word_with_highlight_grapheme(self, grapheme: str | list[str]) -> str:
        """Return the display word wrapped in highlight tags when matched."""

        if isinstance(grapheme, list):
            found = self.contain_in_word_list(grapheme)
        else:
            found = self.contain_in_word(grapheme)
        if found:
            return f"{HIGHLIGHT_ON}{self.display_word}{HIGHLIGHT_OFF}"
        return self.display_word

    def get_word_with_syll_breaks(self) -> str:
        return ".".join(
            syllable.get_syllable_in_lower_case() for syllable in self.syllables
        )

    def contain_in_word(self, grapheme_symbol: str) -> bool:
        return any(
            grapheme_symbol == grapheme.symbol or grapheme_symbol == grapheme.uppercase
            for grapheme in self.graphemes
        )

    def contain_in_word_list(self, grapheme_symbols: list[str]) -> bool:
        return any(self.contain_in_word(symbol) for symbol in grapheme_symbols)

    def is_tone_contain_in_word(self, tone: str) -> bool:
        return tone in "".join(grapheme.symbol for grapheme in self.graphemes)

    def is_open_syllable(self, index: int) -> bool:
        syllable = self.get_syllable(index)
        return bool(syllable and syllable.is_open_syllable())

    def is_closed_syllable(self, index: int) -> bool:
        syllable = self.get_syllable(index)
        return bool(syllable and syllable.is_closed_syllable())

    def is_syllable_onset(self, grapheme: str) -> bool:
        return any(syllable.is_onset(grapheme) for syllable in self.syllables)

    def is_syllable_coda(self, grapheme: str) -> bool:
        return any(syllable.is_coda(grapheme) for syllable in self.syllables)

    def is_syllable_init(self, grapheme: str) -> bool:
        return any(
            syllable.is_syllable_initial(grapheme) for syllable in self.syllables
        )

    def is_syllable_medial(self, grapheme: str) -> bool:
        return any(syllable.is_syllable_medial(grapheme) for syllable in self.syllables)

    def is_syllable_final(self, grapheme: str) -> bool:
        return any(syllable.is_syllable_final(grapheme) for syllable in self.syllables)

    def is_buildable_word(self, graphemes_taught: list[str]) -> bool:
        return all(
            syllable.is_buildable(graphemes_taught) for syllable in self.syllables
        )

    def is_sight_word(self) -> bool:
        return bool(
            self.settings.sight_words
            and self.settings.sight_words.is_sight_word(self.display_word)
        )

    def is_same(self, word: Word) -> bool:
        """Return whether two words have identical grapheme sequences."""

        if self.grapheme_count() != word.grapheme_count():
            return False
        return all(
            self.get_grapheme(index).is_same(word.get_grapheme(index))
            for index in range(self.grapheme_count())
        )

    def is_minimal_pair(self, word: Word, ignore_tone: bool) -> int:
        """Return the only differing grapheme index, or ``-1``.

        This is PrimerPro's grapheme-level minimal-pair predicate: two words
        must have the same grapheme count and differ at exactly one position.
        """

        if self.grapheme_count() != word.grapheme_count():
            return -1
        differences = []
        for index in range(self.grapheme_count()):
            left = self._pair_grapheme(index, ignore_tone)
            right = word._pair_grapheme(index, ignore_tone)
            if not left.is_same(right):
                differences.append(index)
        return differences[0] if len(differences) == 1 else -1

    def is_minimal_pair_for_graphemes(
        self,
        word: Word,
        ignore_tone: bool,
        grapheme1: Grapheme,
        grapheme2: Grapheme,
    ) -> bool:
        """Return whether the single difference is the requested grapheme contrast."""

        if self.grapheme_count() != word.grapheme_count():
            return False
        found = False
        for index in range(self.grapheme_count()):
            left = self._pair_grapheme(index, ignore_tone)
            right = word._pair_grapheme(index, ignore_tone)
            if not left.is_same(right):
                if found:
                    return False
                if left.is_same(grapheme1) and right.is_same(grapheme2):
                    found = True
                else:
                    return False
        return found

    def is_minimal_pair_harmony(
        self,
        word: Word,
        ignore_tone: bool,
        grapheme1: Grapheme,
        grapheme2: Grapheme,
    ) -> bool:
        """Return whether all differences are the same requested substitution.

        PrimerPro uses this when allowing vowel harmony: words may differ at
        more than one position if every difference is the same vowel contrast.
        """

        if self.grapheme_count() != word.grapheme_count():
            return False
        differences = []
        for index in range(self.grapheme_count()):
            left = self._pair_grapheme(index, ignore_tone)
            right = word._pair_grapheme(index, ignore_tone)
            if not left.is_same(right):
                differences.append((left, right))
        return bool(differences) and all(
            left.is_same(grapheme1) and right.is_same(grapheme2)
            for left, right in differences
        )

    def highlight_missing_graphemes(self, graphemes_taught: list[str]) -> str:
        """Highlight graphemes absent from the exact taught list.

        This mirrors PrimerPro's residue highlighting, which does not apply the
        positional underscore rules used by the buildability predicate.
        """

        symbols = []
        for grapheme in self.graphemes:
            if grapheme.symbol in graphemes_taught:
                symbols.append(grapheme.symbol)
            else:
                symbols.append(f"{HIGHLIGHT_ON}{grapheme.symbol}{HIGHLIGHT_OFF}")
        return "".join(symbols)

    def get_key(self) -> str:
        """Return PrimerPro's zero-padded ordinal sort key for the word."""

        return "".join(
            str(ord(character)).zfill(6)
            for grapheme in self.graphemes
            for character in grapheme.symbol
            if grapheme.symbol.strip()
        )

    def _pair_grapheme(self, index: int, ignore_tone: bool) -> Grapheme:
        if ignore_tone:
            return self.get_grapheme_without_tone(index)
        return self.get_grapheme(index)

    def _build_word(self) -> None:
        cleaned = _clean_word(self.orig_word, self.settings)
        syllable_text = ""
        explicit_syllables = False
        character_index = 0
        while character_index < len(cleaned):
            character = cleaned[character_index]
            if character == self.syllable_break_character:
                self.syllables.append(Syllable(syllable_text, self.settings))
                syllable_text = ""
                explicit_syllables = True
                character_index += 1
                continue
            grapheme, end_index = _next_grapheme(
                cleaned, character_index, self.settings
            )
            self.graphemes.append(grapheme)
            self.display_word += grapheme.symbol
            syllable_text += grapheme.symbol
            character_index = end_index + 1

        if explicit_syllables:
            self.syllables.append(Syllable(syllable_text, self.settings))

        self.cv_pattern = self._build_cv_pattern()
        if not self.syllables and self.orig_word != "":
            self._build_word_syll_breaks()

    def _build_cv_pattern(self) -> str:
        return "".join(
            _cv_marker(grapheme, self.settings) for grapheme in self.graphemes
        )

    def _build_word_syll_breaks(self) -> None:
        if not self.graphemes:
            return
        syllables: list[list[Grapheme]] = [[self.graphemes[0]]]
        for index in range(1, len(self.graphemes)):
            previous = _without_tone(self.graphemes[index - 1], self.settings)
            current = _without_tone(self.graphemes[index], self.settings)
            next_grapheme = (
                _without_tone(self.graphemes[index + 1], self.settings)
                if index + 1 < len(self.graphemes)
                else None
            )
            if (
                previous
                and current
                and next_grapheme
                and previous.is_vowel
                and current.is_consonant
                and next_grapheme.is_vowel
            ):
                syllables.append([self.graphemes[index]])
            else:
                syllables[-1].append(self.graphemes[index])
        self.syllables = [
            Syllable("".join(grapheme.symbol for grapheme in syllable), self.settings)
            for syllable in syllables
        ]


class Paragraph:
    def __init__(self, text: str, settings: Settings) -> None:
        self.settings = settings
        self.original_paragraph = text
        self.sentences: list[Sentence] = []
        self._build_sentences(text)

    def add_sentence(self, sentence: Sentence) -> None:
        self.sentences.append(sentence)

    def get_sentence(self, index: int) -> Sentence | None:
        if 0 <= index < len(self.sentences):
            return self.sentences[index]
        return None

    def sentence_count(self) -> int:
        return len(self.sentences)

    def word_count(self) -> int:
        return sum(sentence.word_count() for sentence in self.sentences)

    def syllable_count(self) -> int:
        return sum(sentence.syllable_count() for sentence in self.sentences)

    def max_number_of_words_in_sentences(self) -> int:
        return max((sentence.word_count() for sentence in self.sentences), default=0)

    def max_number_of_syllables_in_words(self) -> int:
        return max(
            (
                word.syllable_count()
                for sentence in self.sentences
                for word in sentence.words
            ),
            default=0,
        )

    def avg_number_of_word_in_sentences(self) -> int:
        """Return the integer average words per sentence."""

        if not self.sentences:
            return 0
        return self.word_count() // len(self.sentences)

    def as_string(self) -> str:
        return " ".join(sentence.as_string() for sentence in self.sentences).strip()

    def _build_sentences(self, text: str) -> None:
        endings = self.settings.option_settings.ending_punct
        start = 0
        while start < len(text):
            end_positions = [text.find(punct, start) for punct in endings]
            end_positions = [position for position in end_positions if position >= 0]
            if end_positions:
                end = min(end_positions)
                punctuation = text[end]
            else:
                end = len(text)
                punctuation = " "
            sentence_text = text[start:end].strip()
            if sentence_text:
                sentence = Sentence(sentence_text + punctuation, self.settings)
                if sentence.word_count() > 0:
                    self.add_sentence(sentence)
            start = end + 1


class Sentence:
    no_punctuation = " "

    def __init__(self, text: str, settings: Settings) -> None:
        self.settings = settings
        self.original_sentence = text
        self.words: list[Word] = []
        self.ending_punctuation = text[-1]
        if self.ending_punctuation in self.settings.option_settings.ending_punct:
            text = text[:-1]
        else:
            self.ending_punctuation = self.no_punctuation
        self._build_words(text)

    def add_word(self, word: Word) -> None:
        self.words.append(word)

    def get_word(self, index: int) -> Word | None:
        if 0 <= index < len(self.words):
            return self.words[index]
        return None

    def word_count(self) -> int:
        return len(self.words)

    def syllable_count(self) -> int:
        return sum(word.syllable_count() for word in self.words)

    def as_string(self) -> str:
        text = " ".join(word.display_word for word in self.words)
        return (text + self.ending_punctuation).strip()

    def _build_words(self, text: str) -> None:
        token = ""
        separators = set(self.settings.option_settings.general_punct)
        for character in text:
            if character in separators:
                self._add_token(token)
                token = ""
            else:
                token += character
        self._add_token(token)

    def _add_token(self, token: str) -> None:
        token = token.strip()
        if token:
            word = Word(token, self.settings)
            if word.display_word:
                self.add_word(word)


class TextData:
    """Story or corpus text parsed into paragraphs, sentences, and words."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.file_name = ""
        self.paragraphs: list[Paragraph] | None = None

    def add_paragraph(self, paragraph: Paragraph) -> None:
        if self.paragraphs is None:
            self.paragraphs = []
        self.paragraphs.append(paragraph)

    def get_paragraph(self, index: int) -> Paragraph | None:
        if self.paragraphs is not None and 0 <= index < len(self.paragraphs):
            return self.paragraphs[index]
        return None

    def paragraph_count(self) -> int:
        return len(self.paragraphs or [])

    def sentence_count(self) -> int:
        return sum(paragraph.sentence_count() for paragraph in self.paragraphs or [])

    def word_count(self) -> int:
        return sum(paragraph.word_count() for paragraph in self.paragraphs or [])

    def syllable_count(self) -> int:
        return sum(paragraph.syllable_count() for paragraph in self.paragraphs or [])

    def max_number_of_words_in_sentences(self) -> int:
        return max(
            (
                paragraph.max_number_of_words_in_sentences()
                for paragraph in self.paragraphs or []
            ),
            default=0,
        )

    def max_number_of_syllables_in_words(self) -> int:
        return max(
            (
                paragraph.max_number_of_syllables_in_words()
                for paragraph in self.paragraphs or []
            ),
            default=0,
        )

    def avg_number_of_words_in_sentences(self) -> int:
        """Return PrimerPro's integer average words per sentence."""

        if self.paragraph_count() == 0:
            return 0
        return (
            sum(
                paragraph.avg_number_of_word_in_sentences()
                for paragraph in self.paragraphs or []
            )
            // self.paragraph_count()
        )

    def avg_number_of_syllables_in_words(self) -> int:
        """Return PrimerPro's integer average syllables per word."""

        return self.syllable_count() // self.word_count() if self.word_count() else 0

    def build_word_list(self) -> WordList:
        word_list = WordList(self.settings)
        for word in self.iter_words():
            word_list.add_word(word)
        return word_list

    def build_word_list_with_no_duplicates(self) -> WordList:
        """Build a sorted word list using the first occurrence of each display word."""

        word_list = WordList(self.settings)
        seen = set()
        for word in self.iter_words():
            if word.display_word not in seen:
                word_list.add_word(word)
                seen.add(word.display_word)
        return word_list

    def update_grapheme_counts(
        self,
        inventory: GraphemeInventory,
        ignore_sight_words: bool,
        ignore_tone: bool,
    ) -> GraphemeInventory:
        """Update inventory text counts from every parsed word.

        Sight words can be skipped. When ``ignore_tone`` is true, tone graphemes
        count toward their tone-bearing unit instead of the tone itself.
        """

        inventory.reset_text_counts()
        for word in self.iter_words():
            if ignore_sight_words and word.is_sight_word():
                continue
            for grapheme in word.graphemes:
                _increment_grapheme_count(inventory, grapheme, ignore_tone)
        return inventory

    def iter_words(self) -> list[Word]:
        return [
            word
            for paragraph in self.paragraphs or []
            for sentence in paragraph.sentences
            for word in sentence.words
        ]


class WordList:
    """Sorted collection of PrimerPro words keyed by grapheme ordinal strings."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings
        self._words: list[tuple[str, Word]] = []

    def add_word(self, word: Word) -> None:
        """Add a word and keep the list sorted by PrimerPro key."""

        key = word.key
        while any(existing_key == key for existing_key, _word in self._words):
            suffix = 1
            candidate = f"{key}{suffix:06d}"
            while any(existing_key == candidate for existing_key, _word in self._words):
                suffix += 1
                candidate = f"{key}{suffix:06d}"
            key = candidate
        word.key = key
        self._words.append((key, word))
        self._words.sort(key=lambda item: item[0])

    def del_word(self, index: int) -> None:
        del self._words[index]

    def get_word(self, index: int) -> Word:
        return self._words[index][1]

    def get_word_key(self, index: int) -> str:
        return self._words[index][0]

    def word_count(self) -> int:
        return len(self._words)

    def is_word_in_list(self, word: str | Word) -> bool:
        """Return whether a display word or grapheme-equivalent word is present."""

        if isinstance(word, Word):
            return any(existing.is_same(word) for _key, existing in self._words)
        return any(existing.display_word == word for _key, existing in self._words)

    def get_display_line_for_word(
        self,
        index: int,
        highlights: list[str] | str | None = None,
    ) -> str:
        """Return PrimerPro's tab-delimited display line for a word."""

        word = self.get_word(index)
        if highlights:
            display = word.get_word_with_highlight_grapheme(highlights)
        else:
            display = word.display_word
        return f"{display}\t\t"

    def iter_words(self) -> list[Word]:
        return [word for _key, word in self._words]


class GraphemeTaughtOrder:
    """Ordered list of graphemes considered taught in a primer lesson."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._graphemes: list[str] = []

    def add_grapheme(self, grapheme: str) -> None:
        self._graphemes.append(grapheme)

    def del_grapheme(self, index: int) -> None:
        del self._graphemes[index]

    def get_grapheme(self, index: int) -> str:
        return self._graphemes[index]

    def count(self) -> int:
        return len(self._graphemes)

    def is_taught_grapheme(self, grapheme: str) -> bool:
        return grapheme in self._graphemes

    def get_missing_graphemes(self) -> str:
        """Return taught entries whose base symbol is absent from the inventory."""

        missing = []
        inventory = self.settings.grapheme_inventory
        for grapheme in self._graphemes:
            bare = grapheme.strip("_")
            if inventory is None or not inventory.is_in_inventory(bare):
                missing.append(grapheme)
        return " ".join(missing)


class SightWords:
    """Exact-match sight-word list used as an optional residue exception."""

    def __init__(self, _settings: Settings) -> None:
        self._words: list[str] = []

    def add_word(self, word: str) -> None:
        self._words.append(word)

    def del_word(self, index: int) -> None:
        del self._words[index]

    def get_word(self, index: int) -> str:
        return self._words[index]

    def count(self) -> int:
        return len(self._words)

    def is_sight_word(self, word: str) -> bool:
        return word in self._words


def _clean_word(text: str, settings: Settings) -> str:
    ignored = set(settings.option_settings.import_ignore_chars)
    general = set(settings.option_settings.general_punct)
    general.discard(Word.syllable_break_character)
    general.discard(" ")
    ending = set(settings.option_settings.ending_punct)
    ending.discard(Word.syllable_break_character)
    cleaned = []
    for character in text:
        if character in "[]":
            continue
        if character in ignored or character in general or character in ending:
            continue
        if character == '"':
            continue
        cleaned.append(character)
    return "".join(cleaned)


def _decompose(text: str, settings: Settings) -> list[Grapheme]:
    graphemes = []
    index = 0
    while index < len(text):
        grapheme, end_index = _next_grapheme(text, index, settings)
        graphemes.append(grapheme)
        index = end_index + 1
    return graphemes


def _next_grapheme(text: str, index: int, settings: Settings) -> tuple[Grapheme, int]:
    inventory = settings.grapheme_inventory
    max_size = settings.option_settings.max_size_grapheme
    lookahead = min(max_size, len(text) - index)
    for size in range(lookahead, 0, -1):
        candidate = text[index : index + size]
        if inventory is not None:
            grapheme = inventory.get_grapheme(candidate)
            if grapheme is not None:
                return grapheme, index + size - 1
    return Grapheme(text[index]), index


def _without_tone(grapheme: Grapheme | None, settings: Settings) -> Grapheme | None:
    if grapheme is None or not grapheme.is_tone:
        return grapheme
    inventory = settings.grapheme_inventory
    if inventory is None:
        return grapheme
    tone_index = inventory.find_tone_index(grapheme.symbol)
    if tone_index < 0:
        return grapheme
    tone = inventory.get_tone(tone_index)
    return tone.tone_bearing_unit or grapheme


def _cv_marker(grapheme: Grapheme | None, settings: Settings) -> str:
    if grapheme is None:
        return "?"
    grapheme = _without_tone(grapheme, settings)
    if grapheme is None:
        return "?"
    if grapheme.is_consonant:
        return settings.option_settings.cv_cns
    if grapheme.is_vowel:
        return settings.option_settings.cv_vwl
    if grapheme.is_syllograph:
        return settings.option_settings.cv_syllograph
    if grapheme.is_tone:
        return settings.option_settings.cv_tone
    return "?"


def _matches_taught_position(
    symbol: str,
    graphemes_taught: list[str],
    index: int,
    grapheme_count: int,
) -> bool:
    for taught in graphemes_taught:
        if taught == symbol:
            return True
        if index == 0 and taught.endswith("_") and taught[:-1] == symbol:
            return True
        if (
            index == grapheme_count - 1
            and taught.startswith("_")
            and taught[1:] == symbol
        ):
            return True
    return False


def _increment_grapheme_count(
    inventory: GraphemeInventory,
    grapheme: Grapheme,
    ignore_tone: bool,
) -> None:
    consonant_index = inventory.find_consonant_index(grapheme.symbol)
    if consonant_index >= 0:
        inventory.get_consonant(consonant_index).incr_count_in_text_data()
    vowel_index = inventory.find_vowel_index(grapheme.symbol)
    if vowel_index >= 0:
        inventory.get_vowel(vowel_index).incr_count_in_text_data()
    tone_index = inventory.find_tone_index(grapheme.symbol)
    if tone_index >= 0:
        tone = inventory.get_tone(tone_index)
        if ignore_tone:
            tbu = tone.tone_bearing_unit
            if tbu is not None:
                _increment_grapheme_count(inventory, tbu, False)
        else:
            tone.incr_count_in_text_data()
    syllograph_index = inventory.find_syllograph_index(grapheme.symbol)
    if syllograph_index >= 0:
        inventory.get_syllograph(syllograph_index).incr_count_in_text_data()

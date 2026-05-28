# PrimerPro Python Port

This package ports the core PrimerPro decodability objects and searches to
Python. PrimerPro is a primer-development tool: it helps a literacy worker
answer practical questions such as which words are buildable from the graphemes
already taught, which story words still contain untaught residue, and which
grapheme contrasts form minimal pairs.

The port preserves the original application's behavior rather than redefining a
new phonics model.

## Core Concepts

PrimerPro is driven by a project-specific grapheme inventory. A grapheme is the
written unit PrimerPro should treat as atomic. It may be a single letter, a
multigraph such as `sh`, `ng`, or `igh`, a tone-marked written form, or a
syllograph. Words are segmented by the longest matching inventory entry, up to
`OptionSettings.max_size_grapheme`.

Decodability is checked against a taught-grapheme list. A word is buildable when
every syllable is buildable from that taught list. PrimerPro also supports
position-marked taught graphemes: `t_` can satisfy syllable-initial `t`, and
`_t` can satisfy syllable-final `t`.

Sight words are exact-match exceptions used by searches such as residue search.
They do not make a word buildable; they let a search ignore a known exception.

Tone graphemes can point to a tone-bearing unit. Tone-aware methods can either
compare the written tone grapheme directly or substitute the tone-bearing unit,
depending on the operation.

## Package Layout

- `primerpro.model`: settings, graphemes, inventories, words, syllables,
  paragraphs, text data, word lists, taught graphemes, and sight words.
- `primerpro.search`: buildable-word, residue, minimal-pair, tone-chart, and
  tone-pair searches.
- `primerpro.__init__`: public exports for the model and search classes.

## Basic Usage

Create settings with a grapheme inventory, parse text, then check words against
a taught-grapheme list.

```python
from primerpro import (
    Consonant,
    GraphemeInventory,
    Paragraph,
    Settings,
    TextData,
    Vowel,
)

settings = Settings()
inventory = GraphemeInventory(settings)

for symbol in ["m", "n", "p", "s", "sh", "t"]:
    inventory.add_consonant(Consonant(symbol))

for symbol in ["a", "i"]:
    inventory.add_vowel(Vowel(symbol))

settings.grapheme_inventory = inventory

text_data = TextData(settings)
text_data.add_paragraph(Paragraph("sat ship thin.", settings))

taught = ["s", "a", "t", "sh", "i", "p"]

for word in text_data.iter_words():
    print(word.display_word, word.is_buildable_word(taught))
```

Expected result:

```text
sat True
ship True
thin False
```

## Searches

Use `BuildableWordSearchTD` to list buildable words from text data:

```python
from primerpro import BuildableWordSearchTD

search = BuildableWordSearchTD(1, settings)
search.graphemes = taught
search.execute_buildable_word_search(text_data)

print(search.search_count)
print(search.search_results)
```

Use `ResidueSearch` to list not-yet-decodable words and highlight missing
graphemes:

```python
from primerpro import ResidueSearch

search = ResidueSearch(1, settings)
search.graphemes = taught
search.execute_residue_search(text_data)

print(search.search_results)
```

## Demo Script

The phonics benchmark includes a runnable short tutorial with an English
inventory, taught-grapheme stage, residue report, and sight-word exception:

```bash
uv run --package primerpro python \
  benchmarks/phonics_benchmark/scripts/decodable_text_demo.py
```

## Development

Run the PrimerPro tests from the repository root:

```bash
uv run --package primerpro pytest packages/primerpro/tests
```

Run lint checks for the package:

```bash
uv run ruff check packages/primerpro/src/primerpro
```

## Notes On Scope

This is not a statistical readability model. It is a faithful port of
PrimerPro's rule-based behavior:

- Text is cleaned and segmented into configured graphemes.
- Buildability is binary at the word level.
- Residue reports untaught graphemes.
- Frequency counts operate over decomposed graphemes.
- Minimal-pair helpers compare aligned grapheme sequences.
- Vowel-harmony minimal pairs allow repeated instances of the same vowel
  contrast.

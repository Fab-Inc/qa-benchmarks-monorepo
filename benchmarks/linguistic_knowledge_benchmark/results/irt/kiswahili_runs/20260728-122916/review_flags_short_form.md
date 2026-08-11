# Short-answer questions to review (≥4 models flagged)

Questions where **at least 4 of the 10 tested models** produced a short-answer that the
strict LLM judge marked wrong for a **form-slip** or **borderline** reason (valid
synonym, accepted variant, or a genuine form-vs-knowledge judgment call) — not a real
knowledge error.

- **IDs are the FINAL dataset ids** (`selected_dataset_cleaned.json`), so a reviewer can
  find each item directly.
- Source: `short_form_miss_audit_swahili_v10-full-dataset.csv` (per-model
  categorisation) mapped onto the final selection.
- Solution tags: **[add]** = add accepted answer(s) to `expected_answers`; **[reword]**
  = clarify/revise the item; **[keep]** = the strict rejection is justified, leave
  as-is.

______________________________________________________________________

## Q135 — "What is the Swahili term for the act of negating a verb form?" · 7 models

**Expected:** `kukanusha`

| model             | type       | output                    |
| ----------------- | ---------- | ------------------------- |
| gemini-3.1-pro    | form_slip  | Ukanushaji                |
| gemma-4-31b       | form_slip  | Ukanushaji                |
| claude-sonnet-4-6 | form_slip  | Ukanushi                  |
| gemini-flash-lite | form_slip  | Ukanushi                  |
| gpt-4.1           | form_slip  | Ukanushi                  |
| gpt-5.4-nano      | borderline | Negation (English)        |
| mistral-small     | borderline | Negation marker (English) |

**Proposed — [add] + \[reword\]:** most models gave the noun *ukanushaji/ukanushi*
("negation, the process"), a valid answer to "the act of negating". Add `ukanushaji`,
`ukanushi` to `expected_answers`. **Also review against Q142** — the two items (act vs.
negated form) are near-duplicates; sharpen both stems so the intended distinction is
unambiguous.

## Q150 — "…term for reduplication (the doubling of a word)?" · 6 models

**Expected:** `urudufu`

| model             | type       | output           |
| ----------------- | ---------- | ---------------- |
| gemini-3.1-pro    | form_slip  | Uradidi          |
| gpt-4.1           | form_slip  | Urudufishaji     |
| claude-sonnet-4-6 | borderline | Urudiaji         |
| gemini-flash-lite | borderline | Urudiaji         |
| gemma-4-31b       | borderline | Urudiaji         |
| mistral-small     | borderline | Double (English) |

**Proposed — \[add\]:** `uradidi` is a standard alternative term for reduplication → add
it to `expected_answers` (optionally `urudufishaji`). Do **not** add *urudiaji*
("repetition", broader).

## Q46 — "…one-word contracted form of the phrase bibi yake ('his/her grandmother')." · 5 models

**Expected:** `bibiye`

| model             | type       | output   |
| ----------------- | ---------- | -------- |
| claude-haiku      | form_slip  | bibake   |
| gemini-flash-lite | form_slip  | Bibiaye  |
| mistral-small     | form_slip  | Bibi'ke  |
| gpt-4.1           | borderline | bibiyake |
| gemma-4-31b       | borderline | bibiyake |

**Proposed — [reword] / \[add\]:** the -ake contraction has several valid surface forms
(*bibiye ~ bibiaye ~ bibake*). Either add `bibiaye` (and dialectal `bibake`) or reword
to name the exact target form. Note *bibiyake* is plain concatenation, not a contraction
→ stays wrong.

## Q133 — "…the act of solving a riddle (kitendawili) is called what?" · 5 models

**Expected:** `kutegua`

| model             | type       | output                 |
| ----------------- | ---------- | ---------------------- |
| gpt-4.1           | form_slip  | Tegua kitendawili      |
| claude-haiku      | borderline | Kutatua kitendawili    |
| claude-sonnet-4-6 | borderline | Kutatua kitendawili    |
| gemma-3-12b       | borderline | Kutatua                |
| gpt-5.4-nano      | borderline | Utatuzi wa kitendawili |

**Proposed — \[add\]:** *kutegua* is the idiomatic term; accept surface variants that
contain the root — `tegua`, `kutegua kitendawili`. Reviewer decision on whether the
general *kutatua* ("solve") should also count; recommend keeping it strict (kutegua is
the taught idiom).

## Q142 — "…term for the negated form of a verb (the negation)?" · 5 models

**Expected:** `ukanusho`

| model             | type      | output     |
| ----------------- | --------- | ---------- |
| claude-sonnet-4-6 | form_slip | Ukanushi   |
| gemini-flash-lite | form_slip | Kanusho    |
| gemma-4-31b       | form_slip | Ukanushaji |
| gpt-5.4-nano      | form_slip | Ukanushaji |
| gpt-4.1           | form_slip | Kukanusha  |

**Proposed — [add] + \[reword\]:** add same-root variants `kanusho`, `ukanushi`. gpt-4.1
answered *kukanusha* — i.e. the expected answer to **Q135** — confirming the two
negation items overlap. Disambiguate Q135/Q142 (act vs. result) or merge them.

## Q148 — "What is the Swahili term for 'direct speech'?" · 5 models

**Expected:** `usemi halisi`

| model             | type       | output                    |
| ----------------- | ---------- | ------------------------- |
| claude-sonnet-4-6 | form_slip  | Usemi wa moja kwa moja    |
| gpt-4.1           | form_slip  | Usemi wa moja kwa moja    |
| claude-haiku      | borderline | Hotuba ya moja kwa moja   |
| gemma-4-31b       | borderline | Usemi wa moja (truncated) |
| gpt-5.4-nano      | borderline | Maneno ya moja kwa moja   |

**Proposed — \[add\]:** `usemi wa moja kwa moja` is a genuinely used synonym for direct
speech → add to `expected_answers`.

## Q137 — "What is the Swahili grammatical term for possessives?" · 4 models

**Expected:** `vimilikishi`

| model             | type       | output                                       |
| ----------------- | ---------- | -------------------------------------------- |
| claude-sonnet-4-6 | form_slip  | Vivumishi vimilikishi                        |
| gemma-3-12b       | borderline | Vi-miliki                                    |
| claude-haiku      | borderline | Possessive adjectives or genitives (English) |
| mistral-small     | borderline | Possessive Concord (English)                 |

**Proposed — \[add\]:** accept `vivumishi vimilikishi` ("possessive adjectives",
contains the term). English answers stay wrong.

## Q5 — "…alphabet unit listed immediately after n and before ny." · 4 models

**Expected:** `ng'`

| model          | type       | output             |
| -------------- | ---------- | ------------------ |
| claude-haiku   | borderline | ng (no apostrophe) |
| gemma-4-31b    | borderline | ng (no apostrophe) |
| gemma-3-12b    | borderline | Ŋ (IPA symbol)     |
| ministral-3-8b | borderline | N'                 |

**Proposed — \[keep\]:** this is a D1 orthography item and the apostrophe on `ng'` is
the exact distinction being tested, so rejecting `ng` is correct. Optionally state in
the stem that the apostrophe is required. (Be aware models systematically drop it.)

## Q69 — "Write the verb meaning 'I have eaten', from the stem -la." · 4 models

**Expected:** `nimekula`

| model        | type       | output                     |
| ------------ | ---------- | -------------------------- |
| gpt-4.1      | form_slip  | Nimesha kula (nimeshakula) |
| gemma-3-12b  | borderline | Nime-la                    |
| gemma-4-31b  | borderline | Nimela                     |
| gpt-5.4-nano | borderline | nime-la                    |

**Proposed — [keep] (optional [add]):** `nimekula` is correct; *nimela* drops the
obligatory *ku-* that monosyllabic stems require — a real teaching point, so keep it
wrong. Optionally add `nimeshakula` ("have already eaten") as an accepted variant.

## Q71 — "…'you (singular) will not weed' (negative future), from -palilia." · 4 models

**Expected:** `hutapalilia`

| model         | type       | output                        |
| ------------- | ---------- | ----------------------------- |
| gpt-4.1       | form_slip  | huta palilia (internal space) |
| gemma-4-31b   | borderline | hautapalilia                  |
| gpt-5.4-nano  | borderline | hautapalilia                  |
| mistral-small | borderline | Huutapalilia                  |

**Proposed — \[keep\]:** *hu-* is the standard 2sg negative prefix; *hau-* is
non-standard, so rejecting it is defensible. The only true form-slip is the internal
space (*huta palilia*); no content change needed.

______________________________________________________________________

## Flagged but NOT in the final dataset — no action

Five questions with ≥4 flags were dropped by the IRT selection and are **not** in the
final dataset, so they need no review here (original ids): **68** (`chake`→"kiatu
chake"), **179** (`rudi`→"njoo/rejea/kuja/ja"), **105** (`utasikia`→animate agreement),
**284** (`kikomo`→"nukta"), **292** (alternative term). If any are re-added later, they
carry the same synonym/variant issues.

## Pattern for the reviewer

Most high-recurrence flags are **D9 grammar-terminology short-answers** where a valid
Swahili synonym or same-root variant exists → fix by **broadening `expected_answers`**
(Q150, Q148, Q137, Q133, and the negation pair Q135/Q142, which should also be
disambiguated). A few borderline items (**Q5, Q69, Q71**) are cases where the strict
rejection is **linguistically correct** — leave them.

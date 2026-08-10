## The model behind this

The pipeline runs over a **document set** — one or many sources — and is the same every
time; only the inputs change. Each stage is a **distinct technical step** with its own
craft and its own precise instructions:

- **1a Digest** — *how to read a pile of long sources well*: open each one, and record
  per document what it contains across the domain × grade grid, counting testable facts
  with locators. Each fact carries a **verbatim example** quoted from the source while
  the document is open. One `by_document` block per source — no cross-source rollups
  here.
- **1b Merge** — *how to consolidate the per-document maps into one usable feature map*:
  merge duplicate facts (keeping every source + its tier), order each domain's features
  by grade, and **carry each example through from the digest** — no sources are
  re-attached here, so 1b is a pure JSON→JSON step. No counts, no filtering — Stage 3
  allocates. Language-specific structure rides in as a `subdomain` tag; the fixed D1–D9
  spine stays put.
- **2 Generate (sharded)** — *how to write good candidate items*: run once per
  **subset** of the sources, given the full feature map but generating only for features
  whose sources are attached. **Oversample** (allow more than one candidate per feature)
  and make **no global decisions** — no counts, no cross-domain balance. This shards the
  context so large document sets fit.
- **3a Pool & QC** — *how to combine and quality-check candidates*: merge all shard
  outputs, score each item against the anti-gaming rules, and flag near-duplicates. No
  sources attached; annotate, don't drop.
- **3b Select (IRT)** — *how to pick the final dataset with Item Response Theory*: a
  **script** (`estimate_irt.py`, not a prompt) that scores 3a's QC-passed, de-duplicated
  pool on a panel of models, fits IRT (difficulty × discrimination) to the per-item
  results, and selects the final set to avoid saturation, spread the models out, and
  keep a balanced domain×grade coverage — emitting the selected dataset (renumbered)
  plus per-item / per-model tables and diagnostic plots.

Because the digest and plan are explicit and **attributed per source**, you can run the
pipeline over different document sets and **compare the outputs**:

Output quality follows input quality; this staged, attributed design makes that
relationship **visible, reproducible, and controllable**. Runs are independent, so they
can go in **series or in parallel**.

**How to run**

- **1a:** fresh chat, **attach the document set** for this run, paste the 1a block, save
  the JSON. (Repeat with a different set for a comparison run.)
- **1b:** if 1a was run in several passes, first **manually concatenate the per-run
  digest JSONs into one file** — that single combined digest is the input to 1b. Then
  fresh chat; **attach the combined 1a digest as a JSON file** (no source PDFs), paste
  the 1b block, save the `feature_map`. A JSON-to-JSON merge that carries examples
  through.
- **2:** for each **shard** (a subset of the cited sources that fits context), new chat;
  attach that shard's source PDFs **and the `feature_map` JSON file**, paste the stage 2
  block, save the candidate array. Repeat over shards until every source has been
  covered.
- **3a:** new chat, no sources; **attach the Stage 2 candidate arrays** — this can be
  several JSON files (one per shard run) or a single JSON file (if Stage 2 ran only
  once, or you concatenated the shard outputs into one) — paste the 3a block, save the
  annotated pool (QC verdicts + dup groups).
- **3b:** score the QC-passed pool on a panel of models, then run `estimate_irt.py` on
  those results to fit IRT and select the final dataset (selected ids + renumbered
  dataset + diagnostics).

______________________________________________________________________

## STAGE 1a — Resource digest (the survey)

> Fresh chat. **Attach the document set for this run** (reference grammar, curriculum,
> and any teacher guides / scope-and-sequence / graded readers). Paste the block.

```
You are a descriptive linguist and early-grade literacy specialist working on Swahili
(Standard / Kenya). You are helping build a knowledge benchmark that tests whether
a Swahili-capable language model holds the linguistic content knowledge a Primary
literacy teacher needs. The benchmark is a RETRIEVAL test: each item probes one
discrete, knowable linguistic fact — not a teaching decision and not open-ended
reasoning — and a fluent Swahili speaker should be able to answer every item
regardless of teaching experience.

Your job in THIS step is NOT to write questions. It is to SURVEY the attached document
set — going through each source one by one — and summarise what each one contains that
could be useful for building that benchmark, organised over a fixed grid of 9
linguistic domains x Primary grades G1-G5 (plus out-of-band G0 and beyond_primary).
This map (the "digest") is the auditable evidence base a later step uses to decide how
many questions to write where.

THE GRID (the context your survey is organised over)
DOMAINS (9; language-agnostic spine, instantiated for Swahili):
  D1 Phonics & Orthography       — sound-letter correspondence, spelling rules, syllable structure
  D2 Phonology & Suprasegmentals — phoneme contrasts, length, stress/tone, prosody
  D3 Nominal Morphology & Agreement — number, gender/class, determiner/adjective/possessive agreement
  D4 Verbal Morphology           — agreement, tense/aspect/mood, derivation & valency, non-finite forms, negation
  D5 Syntax                      — word order, argument structure, relativisation, focus, ellipsis, quantifiers
  D6 Semantics & Lexicon         — word meaning, polysemy, semantic roles, scope
  D7 Pragmatics & Discourse      — anaphora, reference tracking, information structure, narrative
  D8 Sociolinguistic & Register  — honorifics, dialect, register, idiom/proverb
  D9 Metalinguistic Reasoning    — rule induction from paradigms; explicit grammatical analysis

DOMAIN FILING (which domain a fact belongs to — file by WHAT A SOLVER MUST KNOW to
answer it, NOT by what the fact is superficially about):
  - A Swahili grammatical / metalinguistic TERM (naming a rule, class, or structure) -> D9,
    regardless of the term's subject.
  - What a word MEANS, or the word FOR something -> D6, regardless of the word's subject.
  - Variation BETWEEN SPEAKERS (register, honorifics, proverbs, dialect, variety) -> D8.
  Boundary cases where the spine overlaps:
  - Proverbs & fixed sayings -> D8. Figurative use of ordinary words -> D6.
  - Discourse structures (how a text is built) -> D7. The Swahili LABELS for those
    structures -> D6. What a text is DOING -> D7; what a word/idiom MEANS -> D6 (D8 if a
    fixed saying).
  - Register, honorifics, variety -> D8. Culturally-themed vocabulary -> D6.
  File a fact where the KNOWLEDGE lives; do not let its topic pull it into the wrong domain.

GRADES (anchor each fact to where it is FIRST introduced, even if the curriculum
spirals back to it later in a harder form):
  G0 = pre-primary / below Grade 1,
  G1 = Grade 1   G2 = Grade 2   G3 = Grade 3   G4 = Grade 4   G5 = Grade 5
  beyond_primary = above Grade 5
  G1-G5 are the core Primary band; G0 and beyond_primary catch content that falls
  outside it. Record a fact at its true grade — do NOT cram out-of-band content into
  G1/G5 or drop it.

DISCIPLINE (read first)
- Go through the sources ONE BY ONE: open and read EVERY attached source.
- Work ONLY from the attached sources. Your own prior knowledge of Swahili may be
  incomplete or subtly wrong and is NOT evidence.
- Every coverage claim carries a RESOLVABLE LOCATOR: source id + page/section + heading.
- "absent" / "not covered" is a REQUIRED, first-class value. Where a source is silent
  on a cell, say so. NEVER invent coverage to fill the grid.
- Count TESTABLE FACTS, not prose. A testable fact is one discrete, locatable fact you
  could write a single retrieval question from — answerable by a fluent speaker.
- FILE EACH FACT by WHAT A SOLVER MUST KNOW to answer it, not by its topic (see DOMAIN
  FILING above). A term -> D9; a meaning/word -> D6; speaker-variation -> D8 — even when
  the fact's subject matter belongs elsewhere.
- QUOTE A VERBATIM EXAMPLE for each testable fact, copied EXACTLY from this source (the
  target-language form/phrase + a short English gloss). COPY the form; never paraphrase
  it or reconstruct it from memory.
- ATTRIBUTE everything to a specific source. The point of the digest is to show what
  EACH source contributes, so a later run with a different set can be compared to it.
- When you place a fact at a grade, record HOW you know its grade using the
  grade_placement_evidence enum, honestly: language_empirical |
  related_language_extrapolated | curriculum_anchored | typologist_judgement |
  source_attested. If a grade is your own judgement rather than attested in a source,
  say "typologist_judgement" — do NOT dress it up as "curriculum_anchored".

CLASSIFY EACH SOURCE BY TIER (the FIRST thing you do with a document, before its deep review)
For each document, before surveying its content, glance at its first few pages (title
page, front matter, publisher, intended audience), identify WHAT KIND of document it is,
record it as "tier", then go on to survey it:
  curriculum       — official syllabus, curriculum framework, scope-and-sequence (e.g.
                     the national Kiswahili syllabus).
  teacher_material — teacher's guides, lesson manuals, pacing charts.
  student_material — pupil's books, graded readers, workbooks, exercise books.
  other            — anything else (e.g. a reference/descriptive grammar — the "facts"
                     source for verifying forms; dictionaries; academic papers).
Give a one-line "tier_basis" citing what told you (e.g. "title page: 'Standard II Pupil's Book'").

Produce the resource digest as ONE strict JSON object:

"resource_digest":
   {
     "by_document": [                          // one entry per attached source, in the order you read them
       {
         "source_name":"<the file name of this source, used consistently in ALL locators>",
         "tier":"curriculum | teacher_material | student_material | other",
         "tier_basis":"<=12-word why, citing what you saw (title page, front matter, publisher, audience)>",
         "by_domain":[
           {
             "domain":"D1".."D9",
             "by_grade":[
               {
                 "grade":"G0 | G1..G5 | beyond_primary",
                 "coverage":"rich|some|thin|absent",
                 "n_testable_facts_est":<int>,
                 "example_facts":[                         // one object per concrete testable fact
                   {
                     "fact":"<=12-word phrasing of a concrete testable fact found here",
                     "example":"<verbatim target-language form/phrase quoted from THIS source> — <short English gloss>"
                   }
                 ],
                 "locators":["<source_name>:<page/section>:<heading>", ...],
                 "grade_basis":"<grade_placement_evidence enum value> - <=8-word why>",
                 "notes":"<caveats, or 'not covered'>"
               }
             ],
             "domain_summary":"<2-3 sentences: what THIS document offers in this domain - where it is rich vs thin, and at which grades its testable content concentrates>"
           }
         ],
         "document_summary":"<2-3 sentences: what this document as a whole brings - its strongest domains/grades and its main gaps>"
       }
     ]
   }

RULES
- SURVEY ONLY. Do NOT write questions and do NOT assign item counts yet.
- Every `example_facts[].example` is a VERBATIM quote from this source (form + gloss),
  copied exactly — not paraphrased or recalled.
- Testable-fact counts are estimates, but each must be backed by specific locators.
- grade_basis uses the grade_placement_evidence enum honestly (see DISCIPLINE).
- Cover the WHOLE grid: report thin and absent cells as explicitly as rich ones.
- Ground every claim in the attached documents and cite locations (source id +
  page/section + heading).
- Output ONLY the JSON object. No preamble, no markdown fences.

Produce the JSON object now.
```

______________________________________________________________________

## STAGE 1b — Merge the digest into a feature map

> Fresh chat. A **JSON-to-JSON merge** — the only input is the 1a digest. If 1a was run
> in several passes, **manually concatenate the per-run digest JSONs into one file
> first** and use that combined digest. Because the digest can be large, **attach it as
> a JSON file** rather than pasting it inline (the INPUTS slot in the block just points
> to that attached file). Paste the block.

```
You are a descriptive linguist and benchmark designer working on Swahili (Standard /
Kenya). You are helping build a knowledge benchmark that tests whether a
Swahili-capable language model holds the linguistic content knowledge a Primary
literacy teacher needs. The benchmark is a RETRIEVAL test: each item probes one
discrete, knowable linguistic fact — not a teaching decision and not open-ended
reasoning — and a fluent Swahili speaker should be able to answer every item regardless
of teaching experience. Items sit on a fixed grid of 9 linguistic domains x Primary
grades G1-G5 (plus out-of-band G0 and beyond_primary).

This is the MERGE step. A prior pass surveyed the sources and produced a
"resource_digest": an auditable, PER-DOCUMENT map of what each attached source contains
across that domain x grade grid, with per-grade testable facts, locators, and each
document's tier. Your job is to CONSOLIDATE those per-document maps into ONE feature
map — grouped by domain, ordered by grade — that the item-writing step builds
questions from. You do NOT decide how many questions to write or which cells matter
most; a later step does that. You only merge duplicate facts,
organise, and keep every source visible.

THE GRID (fixed; the plan is organised over it)
DOMAINS (9; language-agnostic spine — instantiate it for Swahili via the per-feature
subdomain tags you add below, NOT by changing the spine):
  D1 Phonics/Orthography · D2 Phonology · D3 Nominal morphology · D4 Verbal morphology ·
  D5 Syntax · D6 Semantics/Lexicon · D7 Pragmatics/Discourse · D8 Sociolinguistic/
  Register · D9 Metalinguistic reasoning.
GRADES (anchor each fact to where it is FIRST introduced; grades may spiral):
  G0 = pre-primary / below Grade 1
  G1 = Grade 1   G2 = Grade 2   G3 = Grade 3   G4 = Grade 4   G5 = Grade 5
  beyond_primary = above Grade 5 (lower-secondary or scholarly)
grade_placement_evidence enum (how a grade claim is grounded — record honestly):
  language_empirical | related_language_extrapolated | curriculum_anchored |
  typologist_judgement | source_attested

Produce ONE strict JSON object — a "feature_map":

"feature_map":
   {
     "by_domain":[
       {
         "domain":"D1".."D9",
         "features":[              // ALL features for this domain, ORDERED by grade: G0 → G1..G5 → beyond_primary
           {
             "subdomain":"<language-specific area WITHIN the domain, drawn from the sources, e.g. 'noun-class concord (cl. 7/8)' or 'applicative extension' — free text, NOT a fixed list>",
             "grade":"G0 | G1..G5 | beyond_primary",
             "feature":"<short name>",
             "description":"<the rule/fact in English, precise enough to write a question from>",
             "example":"<verbatim Swahili example + English gloss>",
             "grade_placement_evidence":"<enum value>",
             "sources":[           // 1+ entries; MORE THAN ONE = the same fact attested by several documents
               {"locator":"<source_name>:<page/section>:<heading> — must match a locator in the digest","tier":"curriculum | teacher_material | student_material | other"}
             ]
           }
         ]
       }
     ]
   }

HOW TO MERGE (this is the whole job)
- INCLUDE EVERY testable fact the digest recorded. No counts, no importance ranking, no
  filtering — that is a later step's job. A cell the digest marked "absent" simply yields no
  feature; never invent one to fill the grid.
- MERGE DUPLICATES — ONE FEATURE PER FACT, NOT PER LABEL. If two or more documents attest
  the SAME underlying fact, emit ONE feature and list each document under "sources" (with
  its tier). This holds EVEN WHEN the sources name the fact differently or you would phrase
  it differently: merge on the fact itself, not on a matching label. Do NOT drop the extra
  source — several sources backing a fact is a strength signal the item-writing step uses,
  so it must stay visible.
- CARRY THE TIER. Take each source's tier from the digest's by_document[].tier and put it
  on its "sources" entry. Every locator must match one already in the digest.
- GRADE ON CONFLICT. If the merged documents place the same fact at different grades,
  record the grade from the HIGHEST-TIER source (curriculum > teacher_material >
  student_material > other) and note the disagreement in the description.
- ORDER each domain's features by grade, easiest first (G0 → G1..G5 → beyond_primary).
- REFILE MISPLACED FACTS. File each feature by WHAT A SOLVER MUST KNOW to answer it, not by
  its topic: a Swahili grammatical/metalinguistic TERM -> D9; a word's MEANING or the word
  FOR something -> D6; variation BETWEEN SPEAKERS (register, honorifics, proverbs, variety)
  -> D8. Overlaps: proverbs/fixed sayings -> D8 but figurative use of ordinary words -> D6;
  discourse STRUCTURES -> D7 but their Swahili LABELS -> D6. If the digest placed a fact in
  a domain that contradicts this, MOVE it to the correct domain and note the move in the
  description. Grade is unchanged by a refile.
- SUBDOMAIN TAG. Tag each feature with an emergent subdomain so within-domain breadth is
  visible; the fixed D1–D9 spine does not change.
- CARRY THE EXAMPLE. Copy each feature's `example` verbatim from the digest fact it comes
  from (`example_facts[].example`). Do NOT re-quote from memory and do NOT invent one; the
  digest already captured it from the source.

RULES
- MERGE ONLY — do NOT write questions, assign counts, or rank importance.
- Every feature traces to at least one digest locator, and every locator carries its tier.
- Honour the grade_placement_evidence enum honestly.
- Output ONLY the JSON object. No preamble, no markdown fences.

INPUTS
- RESOURCE DIGEST — one block per document. It is the JSON file attached to this chat:
  <<the resource_digest JSON file attached here>>

This is a JSON-to-JSON merge: build the feature_map from the attached digest alone.

Produce the JSON object now (feature_map).
```

______________________________________________________________________

# STAGE 2 — Candidate generation (sharded)

> Run this **once per shard** — a subset of the sources small enough to fit context. New
> chat; **attach that shard's source PDFs only** (a subset of what the feature_map
> cites) and **attach the FULL `feature_map` as a JSON file** (the INPUTS slot in the
> block just points to that attached file). You generate candidates ONLY for features
> whose sources are in this shard; the rest of the map is context. Repeat over shards
> until every source has been covered, then concatenate the outputs for Stage 3a.

```
You are a linguist item-writer building a knowledge benchmark for Swahili (Standard /
Kenya). The benchmark tests whether a Swahili-capable language model holds the
linguistic content knowledge a Primary literacy teacher needs: how the language WORKS —
its phonics/orthography, phonology, morphology, syntax, lexicon and metalinguistic
structure — as opposed to its ability to USE the language. You write questions IN ENGLISH
about Swahili.

THIS IS A SHARDED RUN. You are given the FULL feature_map but only a SUBSET of its cited
sources (this shard). Generate candidate items ONLY for features that cite at least one
of the ATTACHED sources; skip every other feature silently (it is handled in another
shard). You are a CANDIDATE GENERATOR, not the final assembler: oversample and do not
make any whole-dataset decisions (no target counts, no cross-domain balancing, no answer
letter balancing, no final numbering) — a later step selects and balances. Output a JSON
array of candidate items for this shard.

CONTEXT (what kind of item to write)
This is a RETRIEVAL test: each item probes one discrete, knowable linguistic fact. A
literate first-language speaker of the target language should be able to answer every item
from knowledge of the language alone — without pedagogical knowledge, world knowledge, or
open-ended reasoning. Crucially, the item must require knowledge of THIS language: if it
can be answered by general reasoning, by copying or segmenting a string shown in the stem,
or by ranking the options without reading the stem, it is not a knowledge item.

THE GRID (assigned upstream; carry both onto each item)
DOMAINS (language-agnostic spine, instantiated for Swahili):
  D1 Phonics & Orthography  D2 Phonology & Suprasegmentals  D3 Nominal Morphology &
  Agreement  D4 Verbal Morphology  D5 Syntax  D6 Semantics & Lexicon  D7 Pragmatics &
  Discourse  D8 Sociolinguistic & Register  D9 Metalinguistic Reasoning
GRADES — where the content is FIRST introduced (even if revisited later in harder form):
  G1-G5 = Primary Grades 1-5; G0 = pre-primary; beyond_primary = above Grade 5.
  Grade marks the content level, not the item's difficulty (see DIFFICULTY).

FRAMEWORK TAG (literacy-construct layer; tag EVERY item)
Each item already carries a linguistic domain (D1-D9). This tag records, separately, which
strand of the Simple View of Reading (Gough & Tunmer, 1986) the tested knowledge serves,
and which GEEAP (2025) core instructional skill an error in it would corrupt. The layers
are independent: the domain says what KIND of fact it is; the framework tag says what the
fact is FOR in learning to read.

THE FUNCTIONAL TEST (apply this, not the domain label)
Ask: in a child's reading, does this knowledge serve turning WRITTEN FORMS into words
(decoding), or turning words into MEANING (language_comprehension)?
- "decoding": letter-sound mapping, spelling conventions, syllable structure, sound
  contrasts as written, segmenting written word forms into readable parts (including
  morpheme-level segmentation of long agglutinative words).
- "language_comprehension": what forms MEAN or how they AGREE — concord, tense/aspect
  semantics, word meaning, sentence interpretation, register, discourse. Oral-language
  content (greetings, exchange patterns, proverbs the curriculum teaches) belongs HERE,
  under geeap_component oral_language — early-grade curricula carry it inside the literacy
  programme.
- "outside_flr_construct": the knowledge has NO bearing on supporting a child's literacy
  (e.g. language-family classification, dialectology, etymology of linguistic labels). If a
  fact only supports an outside_flr_construct item, DO NOT WRITE it — skip the feature. This
  tag should therefore be rare-to-absent in your output; use it only to justify skipping,
  never to ship an item.

PER-DOMAIN CROSSWALK (defaults + overrides for svr_strand and geeap_component)
The two axes are INDEPENDENT: svr_strand records what the knowledge is FOR in learning to
read; geeap_component records the lesson TYPE an error would corrupt. Start from the
domain's default row; apply an override ONLY when its stated condition is met. The
FUNCTIONAL TEST above is the tie-breaker, never the domain label alone.

  D1 Phonics & Orthography      default svr=decoding, geeap=systematic_phonics
     override: spelling/punctuation conventions (apostrophe in ng', number-words,
     Arabic-origin spellings) -> geeap=writing. Segmentation/blending stay systematic_phonics.
  D2 Phonology & Suprasegmentals default svr=decoding, geeap=phonological_awareness
     override: stress/prosody items -> geeap=reading_fluency (svr STAYS decoding).
  D3 Nominal Morphology         default svr=language_comprehension, geeap=oral_language
     override: vowel-merger morphophonology (written form) -> svr=decoding,
     geeap=systematic_phonics. An irregular plural is NOT a merger.
  D4 Verbal Morphology          default svr=language_comprehension, geeap=oral_language
     override: verb-morpheme SEGMENTATION -> svr=decoding, geeap=systematic_phonics.
     Building/assembling a form is NOT segmentation (it stays the default).
  D5 Syntax                     default svr=language_comprehension, geeap=oral_language
     override: interpreting PASSAGE-level structure -> geeap=reading_comprehension.
     Sentence-level grammar does NOT qualify.
  D6 Semantics & Lexicon        default svr=language_comprehension, geeap=oral_language
     override: polysemy/figurative meaning that requires RESOLVING the sense in context ->
     geeap=reading_comprehension. Simply reporting that a word is polysemous stays oral_language.
  D7 Pragmatics & Discourse     default svr=language_comprehension, geeap=reading_comprehension
     override: direct vs reported speech with a PUNCTUATION focus -> geeap=writing.
  D8 Sociolinguistic & Register default svr=language_comprehension, geeap=oral_language
     override: proverbs met IN A READING PASSAGE the item asks the reader to interpret ->
     geeap=reading_comprehension. Dialectology / language classification ->
     svr=outside_flr_construct, geeap=none.
  D9 Metalinguistic Reasoning   svr and geeap INHERIT the rule the item is about:
     a spelling/letter term -> decoding/systematic_phonics; a punctuation term ->
     decoding/writing; a fluency/prosody term -> decoding/reading_fluency; a
     grammar/agreement/word-class term -> language_comprehension/oral_language.

APPLYING AN OVERRIDE
- An override fires ONLY when its stated condition is met. If not met, use the default.
  Do NOT apply an override just because a keyword matches.
- geeap "none" is valid ONLY together with svr=outside_flr_construct.

COVERAGE (this shard)
A feature is ELIGIBLE only if at least one source in its `sources` field is a file
ATTACHED to this prompt (match `sources[].locator`'s source_name against the attached
filenames). Generate candidates for every eligible feature and for NO other — if none of
a feature's sources are attached, skip it silently (another shard handles it).
For each eligible feature write 1-K candidates (default K=2): one when the feature
supports a single good item, up to K when it spans genuinely distinct sub-cases (e.g. the
same concord in two different noun classes) — each candidate uses a different example and
frame, not a reworded twin. Choose answer_type per candidate to suit the fact (production
facts suit short_form; contrasts and judgements suit MCQ). Do NOT assign per-domain
counts or balance grades, answer types, or answer letters across the set.

DIFFICULTY REFLECTS TEACHER KNOWLEDGE, NOT STUDENT LEVEL. The grade sets the content; the
item must be pitched at the teacher responsible for that grade and assess command of that
content — applying a rule to a real form, an irregular formation, or a distinction commonly
mistaken — not student-level recall or anything recoverable from the stem.

FEATURE FIDELITY. Copy `linguistic_feature` VERBATIM and test exactly the fact it names; if
an item drifts to a different fact, rewrite it back or drop it. Check the feature is filed
by WHAT MUST BE KNOWN (a term -> D9, a meaning/word -> D6, speaker-variation -> D8); if its
domain looks wrong, skip it.

OUTPUT — a JSON array of candidate item objects (no global numbering yet):
  {
    "shard_id": "<short label for this shard, e.g. the attached-doc set name>",
    "candidate_id": "<unique within this shard, e.g. c001, c002 …>",
    "domain": "<carried from the feature>",
    "subdomain": "<carried from the feature>",
    "grade": "<carried; G0 | G1..G5 | beyond_primary>",
    "answer_type": "multiple_choice" | "short_form",
    "question": "<English question; embed target-language material verbatim>",
    "options": {"A":"...","B":"...","C":"...","D":"..."},   // MCQ only; omit for short_form
    "expected_answers": ["B"],            // MCQ: one letter. short_form: list of accepted strings
    "linguistic_feature": "<carried from the feature>",
    "svr_strand": "decoding | language_comprehension | outside_flr_construct",
    "geeap_component": "systematic_phonics | phonological_awareness | oral_language | reading_fluency | reading_comprehension | writing | none",
    "sources": [ {"locator":"...","tier":"..."} ],             // carried; the one(s) you verified against
    "grade_placement_evidence": "<carried from the feature>"
  }

ANTI-GAMING RULES (every item must pass ALL of these)

1. NO ANSWER IN THE STEM (the most important rule). The word, morpheme, or letters the item
   asks for must NOT be visible in any form shown in the stem — including inside /phoneme
   slashes/, in a gloss, or as separable parts the solver only has to join. If the answer
   can be read off, copied, or segmented out of what the stem prints, the item is broken.
   - Wrong: "In atakaa, which marker shows the future?" (-ta- is sitting in atakaa).
   - Right: give the parts and require the whole — "Mark the verb -soma for
     third-person-singular future; write the full form." (answer atasoma, not shown).

2. DON'T STATE OR DEMONSTRATE THE TESTED RULE. Do not name the class or rule and then ask
   for the very form it dictates, and do not show a worked example of the same
   transformation the item asks for. Covers: stating "M-WA" then asking for the wa- form;
   the alliterative-concord trap (matching the adjective prefix to a noun prefix already on
   screen); and rule-induction items that hand a complete regular paradigm.

3. USE THE CASE WHERE SURFACE-MATCHING FAILS. For morphology and concord, prefer an
   irregular, an exception, or a NON-alliterative class (where the agreement prefix differs
   from the visible noun prefix), so a reader who only "makes it rhyme" or extrapolates the
   pattern gets it wrong. Regular, transparent, alliterative examples do not test knowledge.

4. LENGTH PARITY (hard constraint). All four options within ~15 characters of one another;
   the correct option must NOT be the longest or the most complete. No "kitchen-sink"
   option that bundles two or more ideas unless all four do. Match hedging and jargon across
   all four (no "typically/often" or stacked technical terms only on the key).

5. UNIFORM OR ABSENT GLOSSES. A parenthetical gloss or label must appear on ALL four options
   or on none. An option's gloss must never name the property being tested (do not tag only
   the correct option "(animate)" when the stem is about animacy).

6. PARALLEL OPTIONS. All four options the same grammatical form and register; if they are
   competing forms, keep them minimal so the linguistic contrast is the only signal. No
   option a substring of another. No absolutist words ("always/never/only") in distractors
   only.

7. DISTRACTORS MUST FUNCTION. Each distractor encodes a distinct, nameable error a real
   learner or speaker would make (the wrong concord, the wrong TAM marker, an
   over-regularised plural). No distractor should be eliminable just by reading the stem,
   and none should be implausibly absurd. Exactly one correct option; no true/false, no
   "all/none of the above".

8. SHORTCUT CHECK (apply before finalising each MCQ). Cover the stem and look only at the
   options: if the answer is still identifiable — longest, only hedged/glossed/complete one,
   or the only one echoing the topic — rewrite it.

9. CONSTRUCT CHECK (sharpened). The item must turn on a fact about THIS language. Reject and
   rewrite anything answerable by: concatenating syllables, reading letters off a shown
   word, general literacy, or general grammar/linguistics not specific to the target
   language.

10. STEM STANDS ALONE. The stem must be self-contained: never reference a document,
   curriculum, grade, or lesson (no "as in the syllabus", "at this grade", "in the reading").

11. NO SCAFFOLDING CLUE. A descriptive hint in the stem must not make the answer findable
   without the tested knowledge. Wrong: asking for a word while stating it "has a doubled
   vowel in each of its two parts" — that describes the answer's shape and lets a solver
   reverse-engineer it. Describe the target by its function, not by features that reveal it.

POSITIVE TEMPLATES (use these forms, especially in the form domains D1-D4)
- PRODUCTION: give the morphemes/parts, require the assembled form (encoding, not
  recognition). Short_form suits this.
- EXCEPTION: target an irregular or a form where the regular pattern over-applies.
- JUDGEMENT: ask which form is correct or which is wrong (grammaticality, spelling), so the
  answer cannot be lifted off a string.
- CONTRAST: a minimal pair where the surface does not disambiguate ("which begins with a
  single nasal rather than nasal-plus-g, ng'ombe or ngao?").

CARRY-OVER RULES
- ENGLISH MEDIUM; only quoted target-language forms appear in Swahili. Plain language, no
  rare metalinguistic jargon unless it is the thing being tested.
- TEXT-ONLY, IMPERSONAL: phrase every stem impersonally (no "you"); never require the model
  to speak, listen, or hear. Test only what is recoverable from the written form.
- TRANSLATION ANSWER-NEUTRAL: a short English gloss may aid clarity but must not reveal the
  tested answer; for D6 meaning items, omit the gloss of the queried word.
- GROUNDING: every item supported by one of its feature's sources; carry that (locator,
  tier) onto the item and verify forms against the ATTACHED source, not memory. On
  disagreement, ground in the highest tier (curriculum > teacher_material >
  student_material > other).
- LETTER BALANCE: spread the correct MCQ answer roughly evenly across A-D over the set.
- SHORT_FORM: answer 1-3 words; list all accepted spelling/spacing/hyphen variants in
  expected_answers (not synonyms or paraphrases).
- Output ONLY the JSON array. No preamble, no markdown fences.

INPUTS
- SOURCES: this shard's subset of the cited PDFs. Only features citing one of THESE are
  eligible.
- FEATURE_MAP — the full feature_map (all 9 domains; features grade-ordered). It is the
  JSON file attached to this chat. Carry subdomain / grade / sources /
  grade_placement_evidence onto each candidate, and assign svr_strand and geeap_component
  using the FRAMEWORK TAG block above:
  <<the feature_map JSON file attached here>>

Produce the JSON array of candidate items for THIS shard now.
```

______________________________________________________________________

# STAGE 3a — Pool & quality-check the candidates

> New chat, no sources. Attach every Stage-2 shard output (or one concatenated JSON file
> of all candidates). This step ANNOTATES only — it never rewrites, adds, or drops a
> candidate.

```
You are a benchmark editor reviewing candidate items for a Swahili (Standard / Kenya)
linguistic-knowledge benchmark. Several sharded generation runs each produced candidate
items from different source subsets. POOL them, QUALITY-CHECK each one, and mark
DUPLICATES — without rewriting, adding, or deleting any candidate. Every input candidate
appears exactly once in your output, annotated with three new fields.

You work from the candidate TEXT alone — no sources are attached. Judge only what is
checkable from the item itself (the shape / anti-gaming rules below), NOT whether a
Swahili form is faithful to a source (that was verified at generation).

QUALITY CHECK — score each candidate against these rules:
1. NO ANSWER IN THE STEM — the asked-for form/morpheme/letters are not visible in the stem
   in ANY form, including inside /phoneme slashes/, a gloss, or as separable parts to join.
2. RULE NOT STATED OR DEMONSTRATED — the item does not name the rule/class then ask for the
   form it dictates, and shows no worked twin of the transformation it asks for.
3. LENGTH PARITY (MCQ) — the four options are within ~15 characters; the key is not the
   longest or most complete; hedging/jargon matched across all four.
4. UNIFORM OR ABSENT GLOSSES (MCQ) — a gloss/label is on all four options or none, and never
   names the tested property.
5. PARALLEL OPTIONS (MCQ) — all four the same form/register; none a substring of another; no
   absolutist words ("always/never/only") in distractors only.
6. DISTRACTORS FUNCTION (MCQ) — each distractor is a distinct, nameable error; none absurd;
   exactly one correct; no true/false, no "all/none of the above".
7. SHORTCUT CHECK (MCQ) — with the stem covered, the key is not identifiable by length,
   gloss, completeness, or topic echo.
8. CONSTRUCT CHECK — the item turns on a fact about Swahili, not general reasoning,
   string-copying, or general literacy/linguistics.
9. WELL-FORMED — required fields present for its answer_type (MCQ: 4 options + exactly one
   letter in expected_answers; short_form: no options, 1-3 word expected_answers).
10. ENGLISH MEDIUM — the question is written in English; only quoted target-language forms
    appear in Swahili, and no rare metalinguistic jargon is used unless it is the very
    thing being tested.
11. TEXT-ONLY & IMPERSONAL — the stem is phrased impersonally (no "you") and never asks the
    reader to speak, listen, or hear; everything tested is recoverable from the written form.
12. TRANSLATION ANSWER-NEUTRAL — any English gloss aids clarity without revealing the tested
    answer; for a meaning item, the queried word is not glossed.
13. STEM STANDS ALONE — the stem references no document, curriculum, grade, or lesson.
14. NO SCAFFOLDING CLUE — no descriptive hint in the stem makes the answer findable without
    the tested knowledge (e.g. stating the target word's letter/shape pattern).
15. IN-CONSTRUCT — svr_strand is not "outside_flr_construct" (such items should not exist;
    fail any that do).
Rules 3-7 apply to multiple_choice items only; short_form items are exempt from them.
Rules 1-2 and 8-15 apply to every item.

DEDUPLICATE — ONE GROUP PER FACT, NOT PER FEATURE LABEL:
- Give every candidate a "dup_group_id". Candidates that probe the SAME underlying fact
  share one id — judge this on the fact the item actually tests (its stem + answer), NOT on
  a matching `linguistic_feature` label. Two candidates with DIFFERENT feature labels still
  share an id if they test the same fact; two with the SAME label are separate groups if
  they test genuinely different facts. A candidate with no twin is its own group of one.
- Within each group set "quality_rank" = 1 for the best candidate (most rules passed,
  best-formed, hardest to game), 2 for the next, and so on. Rank on quality only; do NOT
  drop the losers — the assembler keeps them as backups.

Produce ONE strict JSON array — every input candidate, unchanged except for three ADDED
fields:
  {
    ...all fields of the input candidate, VERBATIM (shard_id, candidate_id, domain,
       subdomain, grade, answer_type, question, options, expected_answers,
       linguistic_feature, svr_strand, geeap_component, sources, grade_placement_evidence)...,
    "qc_verdict": {
      "pass": true|false,                 // false if ANY applicable rule fails
      "failed_rules": [<numbers of the rules that failed>],
      "note": "<=15-word reason if failed, else ''>"
    },
    "dup_group_id": "<id shared by same-fact candidates, e.g. g001>",
    "quality_rank": <int within the dup_group; 1 = best>
  }

RULES
- ANNOTATE ONLY — never edit a candidate's text, add candidates, or remove any. Count in ==
  count out.
- Output ONLY the JSON array. No preamble, no markdown fences.

INPUTS
- CANDIDATES — the candidate items from the generation runs. They are the JSON file(s)
  attached to this chat (one file per generation run, or a single combined file; several
  files may be attached):
  <<the candidate JSON file(s) attached here>>

Produce the annotated JSON array now.
```

______________________________________________________________________

# STAGE 3b — Select the final dataset with IRT (script, not a prompt)

Stage 3b is a deterministic script — `estimate_irt.py` — not a chat prompt. First take
3a's annotated pool, keep `qc_verdict.pass` items and collapse each `dup_group_id` to
its best (`quality_rank` 1), and run that de-duplicated candidate pool against a panel
of models with the eval harness. Then feed the per-item correctness matrix to
`estimate_irt.py`: it fits Item Response Theory (Rasch/1PL difficulty and a corrected
point-biserial discrimination) per item and selects a trimmed final dataset with three
goals — (1) no saturation (drop items every model gets right; keep the top model under a
soft accuracy ceiling), (2) differentiation (favour high-discrimination items so models
spread out), and (3) coverage (a per-domain×grade quota, largest-remainder allocation,
proportional to the discriminating pool). It also ranks items by leave-k-out selection
stability and prints validation diagnostics (ranking recovery, discrimination Gini).
Outputs the selected question ids, the selected dataset (renumbered 1..n), per-item and
per-model CSVs, and diagnostic plots. See that script's header for options and how to
run it.

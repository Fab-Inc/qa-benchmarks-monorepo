# How we're building the Linguistic Knowledge benchmark

*Pilot language: Swahili (Standard / Kenya). Target: ~100–150 items.* *Diagram:
`pipeline_workflow_v10.svg`. Prompts: `process_prompts_chat_ready_v10.md`. Stage 3b
script: `stage_3b_assemble_v10.py`.* *Stages 1a–3a run by hand in a **chat interface**
(ChatGPT, Claude, Gemini) — attach files and paste prompts in the browser; Stage 3b is a
small local script. This is **not** an API pipeline: no code calls the model, which is
why each chat stage is a self-contained, copy-pasteable prompt.*

## What we're building

A benchmark that measures how much an LLM **knows about the formal structure of a
language** — its spelling/sound rules, phonology, morphology, syntax, lexicon, and
ability to reason about its own grammar — as opposed to how well it can *use* the
language. It is a **retrieval test**: each item probes one discrete, knowable fact that
a fluent native speaker could answer regardless of teaching experience.

Items live on a fixed, language-agnostic grid:

- **9 domains (D1–D9):** Phonics/Orthography · Phonology · Nominal morphology · Verbal
  morphology · Syntax · Semantics/Lexicon · Pragmatics/Discourse ·
  Sociolinguistic/Register · Metalinguistic reasoning. The spine never changes;
  language-specific structure rides in per feature as a free-text **`subdomain`** tag.
- **Grades (G0, G1–G5, beyond_primary):** where the content is *first introduced*. Grade
  marks the **content level**, not the question's difficulty.

## Inputs: a document set (1 … N sources)

Any number of sources, each classified into a **tier** that encodes its authority:

| tier               | examples                                                     |
| ------------------ | ------------------------------------------------------------ |
| `curriculum`       | official syllabus / scope-and-sequence                       |
| `teacher_material` | teacher's guides, lesson manuals, pacing charts              |
| `student_material` | pupil's books, graded readers, workbooks                     |
| `other`            | reference grammar (the "facts" source), dictionaries, papers |

Tier is the tie-breaker throughout: when sources disagree on a fact or its grade, the
**highest tier wins**.

> Numbering note: "Stage 1 / Stage 2 …" here name steps of this **chat authoring
> pipeline**. They are unrelated to the eval-harness "Stage 1 (run-bench) / Stage 2
> (judge-bench)" in the README — different numbering, different job.

## Key design ideas

- **Same process every run; only the documents change.** Each chat stage is a **single,
  editable, language-agnostic prompt**. To port to a new language you swap the sources
  and the language name — the grid, the JSON shapes, and the rules stay put.
- **Survey → merge → generate → assemble.** The first two stages only *find and
  organise* what the sources contain; no questions are written and no counts are set
  until later. This keeps the build auditable: every item traces back through the
  feature map to a source locator.
- **Per-source attribution → comparable, parallel runs.** The digest is computed **one
  document at a time** (so sources can be processed in parallel or added later), and the
  merge keeps every contributing source visible. Diffing the digests / feature maps of
  two runs shows exactly what each source buys you on the D×G grid.
- **Sharded so context never binds.** Both the merge (1b) and the writer (2) are freed
  from holding every PDF at once: 1b works from the digest JSON alone, and 2 runs once
  per **subset** of the sources. This lets the pipeline scale to large document sets.
- **Quota-driven allocation, done deterministically.** There is no upfront per-cell
  quota inside the LLM stages; instead a **per-domain×grade quota** in the Stage-3b
  script decides the final shape — reproducible, diffable, and re-runnable when the
  quota changes. That quota is either **derived proportionally from the 1b feature map**
  (each cell gets a share of the target item count matching its feature density, with
  chosen grades excludable) or a **hard-coded table**. The chat stages oversample; the
  script selects. **WIP:** an **IRT-based allocation** is being explored for Stage 3b
  (selecting/weighting cells by item difficulty and discrimination, building on the
  Stage-5 `estimate_irt.py` analysis, rather than by feature density alone). This is not
  finalized — the current feature_map/hard-coded quota approach may be **dropped in
  favour of it** once validated.
- **Grounded in sources at capture.** The verbatim example for each fact is quoted
  **once** at the digest (1a), while that single document is open; the writer (2)
  re-verifies forms against its shard's sources. The merge (1b) no longer re-attaches
  PDFs — it carries the captured examples through.
- **Knowledge, not use; teacher-level difficulty.** Items must turn on a fact about the
  language (a construct check rewrites anything answerable by general reasoning).
  Difficulty is pitched at the **teacher** responsible for that grade's content —
  applying a rule, handling an irregular form, or a distinction learners commonly miss —
  not student recall.
- **File by what must be known; tag by a fixed crosswalk; dedupe by fact.** A fact is
  filed in the domain a solver must *know* to answer it (a term → D9, a meaning → D6,
  speaker-variation → D8), each item's SVR/GEEAP tags follow a per-domain
  default+override crosswalk, and duplicates are collapsed on the underlying fact, not
  the feature label — refinements from pedagogy-expert review.
- **Humans are the gate; the LLM stages produce auditable JSON.** Automated/manual
  checks only triage attention; people accept, fix, or reject, and their fixes re-run
  the prompts.

## The stages

Each chat stage is one prompt; Stage 3b is a script. Inputs/outputs in brackets.

**1a · Digest — survey each source** *(in: PDF → out: JSON)* Goal: per document, map
what it offers across the D×G grid, quoting a **verbatim example** for each testable
fact while the source is open. Why we need it: sources are long, messy, and uneven —
this turns each one into a compact, locator-backed inventory of what is actually
testable, so later stages reason over evidence instead of re-reading PDFs. Key
guidelines: classify the **tier first**; work **only** from the attached source; every
claim carries a **resolvable locator** (file:page:heading); **`absent` is a first-class
value**; count **testable facts** (not prose) and quote each one's example verbatim;
record **how** each grade was placed (`grade_placement_evidence`, honestly). Output: one
`by_document` block per source.

**1b · Merge — consolidate into one feature map** *(in: JSON → out: JSON)* Goal: fold
the per-document digests into a single `feature_map`, grouped by domain and ordered by
grade. Why we need it: the writer needs one clean, de-duplicated view of the language —
not N overlapping per-source maps. A **pure JSON→JSON step** — no sources are
re-attached; each feature's `example` is **carried through verbatim from the digest**.
Key guidelines: **include every fact** (no counts, no filtering); **merge duplicates**
into one feature that lists **all** its sources + tiers; on a **grade conflict**, take
the highest-tier source's grade; tag each feature with an emergent **`subdomain`**.

**2 · Generate — write candidate items (sharded)** *(in: JSON + PDF subset → out: JSON)*
Goal: produce **candidate** items, running **once per shard** — a subset of the sources
small enough to fit context. Why we need it: sharding keeps large document sets inside
the context window, and generating candidates (rather than the final set) lets a later
step do the global balancing an LLM does unreliably. Key guidelines: given the **full
feature_map** but only a **subset of the PDFs**, generate only for features whose
sources are attached; **oversample** (1–K candidates per feature) and make **no global
decisions** — no counts, no cross-domain balance; **ground** every candidate in an
attached source; carry the grid tags and framework tags through; obey the **anti-gaming
rules**. Candidates carry `shard_id` + `candidate_id`, not a global number.

**3a · Pool & QC — annotate the candidates** *(in: JSON → out: JSON)* Goal: pool every
shard's candidates, **quality-check** each against the anti-gaming/shape rules, and mark
**duplicates** — annotating only, never rewriting or dropping. No sources attached. Adds
`qc_verdict`, `dup_group_id`, and a within-group `quality_rank` to each candidate.

**3b · Assemble — build the final dataset (script)** *(in: JSON → out: JSON)* Goal: a
deterministic script (`stage_3b_assemble_v10.py`) that turns the annotated pool into the
final benchmark. Keeps `qc_verdict.pass` items, collapses each `dup_group_id` to its
best, fills a **per-domain×grade quota** — either derived proportionally from the 1b
feature map (`--feature-map`, scaled to `--total-items`, with `--exclude-grades`
dropped) or a hard-coded table — nudges the MCQ/short_form mix and A–D letter balance,
and assigns sequential `question_id`. Emits the **dataset** (v8 item schema), a ranked
**backups** pool (for the gate to pull replacements), and a **gap report** of cells left
under quota.

> **WIP:** a Stage-5 IRT-informed allocation (see `estimate_irt.py`) is being explored
> as a replacement for this feature_map/hard-coded quota — the allocation approach
> described above is the *current* state, not necessarily the final one.

**4 · Human Review gate — the real gate** *(in: JSON → out: JSON)* Goal: a linguist / L1
speaker judges each item — correct? natural? right grade? Accept, fix, or reject.
Rejects **loop back** to re-run an earlier stage, or pull a replacement from the
Stage-3b backups.

**5 · Evaluate — run & score** *(in: JSON → out: CSV)* Goal: run models against the
accepted items with the existing harness; score per item and per domain.

## User journey (how to run a build)

Stages 1a–3a happen in a **chat window** (ChatGPT / Claude / Gemini): each step is a new
or continued conversation where you attach files, paste a prompt, and save the model's
reply. Stage 3b is a local script; the eval harness is Stage 5.

1. **1a:** fresh chat → attach the document set → paste the 1a prompt → save the
   per-document digest JSON (with verbatim examples). Sources are independent — run them
   together, in parallel, or add more later.
1. **1b:** fresh chat → **attach the 1a digest JSON file** (no PDFs) → paste the 1b
   prompt → save the `feature_map`.
1. **2 (per shard):** for each subset of the sources that fits context, new chat →
   attach that shard's PDFs + the `feature_map` JSON file → paste the 2 prompt → save
   the candidate array. Repeat until every source has been covered.
1. **3a:** new chat, no PDFs → attach the concatenated shard arrays → paste the 3a
   prompt → save the annotated pool.
1. **3b:** run `python stage_3b_assemble_v10.py --pool <3a pool>` → get the dataset,
   backups, and gap report.
1. **4:** review every item; rejected items send you back to an earlier stage or to the
   backups.
1. **5:** run the eval harness on the accepted set.
1. **Compare runs** by diffing the per-document digests and the feature maps across
   different document sets.

## Trade-offs (what to watch)

The v9 sharded design removes the old one-pass context ceiling, but shifts the pressure
points:

- **3a is the new context choke point.** The pool-and-QC step must hold the *whole*
  candidate pool at once to dedupe and quality-check globally, and oversampling plus
  cross-shard overlap inflate it. It is text (not PDFs), so the ceiling is far higher —
  but with many sources it can still bind. Levers: lower the per-feature K, reduce shard
  overlap, or shard 3a by domain.
- **Grounding is not re-verified after generation.** 1b dropped the verbatim re-check
  and 3a has no sources, so form-accuracy rests on the 1a capture and the Stage-2 write.
  An error there propagates — the human gate is the sole backstop.
- **Sharding is operator judgement, with redundant work.** You choose the Stage-2 doc
  subsets; a split that separates a feature's corroborating sources changes which shard
  verifies it, and features whose sources span shards are generated more than once
  (extra tokens, a bigger pool for 3a).
- **The hard-coded quota can't exceed pool supply.** If a cell's quota asks for more
  items than the QC-passed pool holds, Stage 3b under-fills and reports the gap —
  coverage still follows the sources, exactly as the digest's `absent`/`thin` cells
  predict.
- **Longer pipeline, more handoffs.** 1a → 1b → 2×N shards → 3a → 3b means more JSON
  boundaries plus a script dependency, so schema drift between steps has more places to
  creep in; the shared item schema is the contract to hold.
- **Grade anchoring is soft when curricula are silent.** Facts not attested in a
  curriculum fall back to `typologist_judgement`; treat those grades as estimates.
- **Manual cost.** Per-document digesting, per-shard generation, and human review are
  deliberate (for grounding and auditability) but add tokens and effort per run.

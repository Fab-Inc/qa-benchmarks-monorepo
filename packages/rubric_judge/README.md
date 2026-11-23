# Rubric Judge

LLM-as-a-judge over YAML rubrics.

## Quickstart

```yaml
# rubric.yaml
Language:
  name: Language
  weight: 1.0
  criteria:
    - id: lang_1
      question: Is every sentence grammatical?
      criterion_type: positive
```

```python
from rubric_judge import evaluate_text, load_rubric, score_judgment

judgment = evaluate_text(
    my_text,
    rubric=load_rubric("rubric.yaml"),
    tags=set(),                    # gating tags — see "Gating"
    judge_model="gpt-4.1-nano",
    llm=my_llm,                    # your model callable — see "The llm callable"
)
scores = score_judgment(judgment)  # {"final_score": ..., "section_scores": ...}
```

## Writing a rubric

A rubric maps section names to section definitions:

```yaml
Language:
  name: Language
  weight: 1.0                    # share of the final score; 0 excludes it
  call_per: section              # "section": one call for all criteria;
                                 # "question": one call per criterion
  section_preface: null          # extra guidance shown before the criteria
  requires_justification: false  # judge must justify each verdict
  criteria:
    - id: lang_1
      question: Is every sentence grammatical?
      criterion_type: positive
      allows_na: false           # may the judge answer "N/A"?
      evidence: true             # judge must quote the deciding text
      weight: 2.0                # points it's worth (default 1)
    - id: lang_2
      question: List every unnatural phrase, with severity.
      criterion_type: evidence_list
      evidence_items:
        quote: str
        severity: [low, medium, high]
      scoring_rules:
        - severity_in: [medium, high]
          less_than: 1                          # 1 pt: no medium/high items
        - less_than: 3                          # 1 pt: at most 2 items total
    - id: lang_fail_1
      question: The text contains multiple ungrammatical sentences.
      criterion_type: fail_condition
```

The three criterion types: `positive` earns its points on Yes; `fail_condition` zeroes
the whole section on Yes; `evidence_list` has no verdict — the judge returns a list of
items instead.

Criterion fields:

| Field             | Meaning                                                                                                                                                         |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `criterion_type`  | `positive`, `fail_condition`, or `evidence_list`                                                                                                                |
| `allows_na`       | judge may answer `N/A`; excluded from the denominator                                                                                                           |
| `gating`          | judge this criterion only for texts with matching tags — see [Gating](#gating)                                                                                  |
| `evidence`        | judge returns `{evidence, response}`: a quote plus the verdict                                                                                                  |
| `evidence_items`  | turn `evidence` (or an `evidence_list` answer) into a structured list: property name → `"str"` or a list of allowed values                                      |
| `evidence_fields` | extra named scalar fields an `evidence_list` answer returns alongside its items                                                                                 |
| `scoring_rules`   | how an `evidence_list` earns points — see below                                                                                                                 |
| `weight`          | points the criterion is worth (default 1). Scales verdicts and each scoring-rule point; meaningless on `fail_condition`. Scoring-only: never forces a re-judge. |

### `scoring_rules`

Each rule is one point, earned when the count of matching items satisfies the rule's
bound:

```yaml
scoring_rules:
  - less_than: 1                          # no items at all
  - severity_in: [medium, high]
    less_than: 1                          # no medium/high items
  - where: {field: severity, not: high}
    less_than: 3                          # at most 2 non-high items
  - less_than: 3
    greater_than: 1                       # exactly 2 items (exclusive range)
```

Matchers: none (counts all items), `severity_in: [...]` (items with those severities),
or `where: {field: F, in|not: V|list}` (a field condition). Both bounds are strict;
together they make an exclusive range. An empty `scoring_rules: []` means the list is
evidence-only and earns no points. Legacy forms (`max_items`, bare `severity_in`,
`max_count_where`) are still understood when rescoring judgments stored before
`less_than` existed.

`evidence_list` criteria must declare `evidence_items` or `evidence_fields`.

### `call_per: question`

Makes one call per criterion, run concurrently, instead of batching the section into a
single call. More calls, but each verdict is independent. A failed call only N/As its
own criterion instead of the whole section.

### The standard prompt

Unless a section says otherwise, each call's prompt is assembled like this:

```text
{context}                          # only when context= is passed

You are evaluating {subject}.      # subject=, default "a text"

## Scoring Guide                   # only when section_preface is set

{preface}

---

TEXT TO EVALUATE:

{text}

Answer ALL of the following questions for the section "<name>".

1. [criterion_id] question         # one line per criterion
2. ...

{instructions}                     # generated to match the call's schema
```

- `context` is optional background you pass to `evaluate_text` — the original question,
  who wrote the text, whatever the judge needs beyond the text itself.
- `subject` names the kind of thing being judged.
- The criteria render as a numbered `[id] question` list.
- The closing instructions are generated to match the call's response schema — their
  text changes with `allows_na`, `evidence`, `requires_justification` and
  `evidence_list` criteria.

### `prompt_template` — a custom prompt

Setting `prompt_template` on a section discards the standard prompt and uses your
template instead; the rendered text is the whole prompt. The slots are the same pieces,
arranged however you like:

| Slot             | Filled with                                        |
| ---------------- | -------------------------------------------------- |
| `{text}`         | the text being judged                              |
| `{context}`      | `context=` — empty if unset                        |
| `{subject}`      | `subject=`                                         |
| `{preface}`      | `section_preface` — empty if unset                 |
| `{criteria}`     | the numbered `[id] question` list                  |
| `{instructions}` | the generated response instructions                |
| `{???}`          | make a custom slot by passing it in `placeholders` |

```yaml
Correctness:
  name: Correctness
  prompt_template: |
    Question: {question}
    Model's answer: {text}
    Does it match an expected answer?
    {instructions}
  criteria:
    - id: correct
      question: The answer matches an expected answer.
      criterion_type: positive
```

```python
evaluate_text(
    "The capital of France is Paris.",            # -> {text}
    rubric=rubric,
    tags=set(),
    judge_model="gpt-4.1-nano",
    llm=my_llm,
    placeholders={"question": "What is the capital of France?"},   # -> {question}
)
```

Omit `{criteria}` or `{instructions}` and the model won't see them — the schema still
parses whatever comes back, so it's on you to include what the call needs.

### Placeholders

`{name}` in rubric text — questions, `section_preface`, `prompt_template` — is filled
from `evaluate_text(placeholders=...)`.

- An unfilled slot raises `ValueError`.
- `{{name}}` renders as a literal `{name}`.
- `placeholders=None` skips rendering entirely; braces pass through verbatim.

Filled values sit inside the rendered prompts, so they feed the call fingerprints that
drive resume — different fills re-judge.

## Gating

A criterion's `gating` is checked against the text's **tags** — a `set[str]` you compute
and pass to `evaluate_text(tags=...)`. Four forms:

| `gating`        | Applies when                                            |
| --------------- | ------------------------------------------------------- |
| absent/`null`   | always                                                  |
| `"tag"`         | `"tag"` is in the text's tags                           |
| `"!tag"`        | `"tag"` is *not* in the text's tags                     |
| `"X+"` / `"X-"` | the text's ordered tag is at-or-above / at-or-below `X` |

**Binary gates** — "does the text have property X?":

```yaml
- id: dialogue_1
  question: Does every line of dialogue have a clear speaker?
  gating: has-dialogue
```

```python
tags = {"has-dialogue"} if has_dialogue(text) else set()
evaluate_text(text, rubric=rubric, tags=tags, ...)
```

**Category gates** — tags are plain strings, so an `axis:value` convention gives you
categories:

```python
tags = {f"genre:{item.genre}", f"lang:{item.language}"}
# e.g. {"genre:poem", "lang:sw"}
```

```yaml
- id: rhyme_1
  question: Do the rhymes land?
  gating: genre:poem        # judged only for poems
- id: prose_1
  question: Do the paragraphs each move the story on?
  gating: "!genre:poem"     # judged for everything else
```

**Ordered gates** — pass `ordered_tags`, the ascending list of tag names forming a
scale. The text's *highest* ordered tag is its level; `+` means "at or above", `-` means
"at or below". A text with no ordered tag matches neither form.

```python
evaluate_text(
    text,
    rubric=rubric,
    tags={"S3"},
    ordered_tags=("S1", "S2", "S3", "S4"),
    ...
)
```

```yaml
- id: plot_arc
  question: Does the story build to a resolution?
  gating: S2+   # S2 and up — too much to ask of S1
- id: simple_plot
  question: Is the sequence of events easy to follow?
  gating: S2-   # S1 or S2 — simplicity is expected only early on
```

Any strings can be ordered tags — `("beginner", "intermediate", "advanced")` works as
well as `("S1", ..., "S4")`. One `ordered_tags` list describes one axis; unordered tags
live alongside it in the same tag set.

A gated-out criterion is excluded from the prompt, stored as
`{"response": "N/A", "gated_out": true}` in its rubric position, and excluded from the
scoring denominator (counted separately as `gated_out_count`).

## Calling the judge

`evaluate_text` accepts these options:

```python
judgment = evaluate_text(
    my_text,
    rubric=rubric,
    tags={"S1"},                        # caller-computed gating tags
    judge_model="gpt-4.1-nano",
    llm=my_llm,
    ordered_tags=("S1", "S2", "S3"),    # enables "S2+"/"S2-" gating
    subject="a story for a child",      # -> "You are evaluating a story for a child."
    context="The author is six.",       # background prepended to every section prompt
    placeholders={"language": "English"},
    system_prompt="You are an expert evaluator of {language} texts.",
)
```

`system_prompt` is the judge persona sent with every call. It renders placeholders like
rubric text and is part of each call's fingerprint, so changing it re-judges.

### The `llm` callable

The package does not include an LLM client; the `llm` argument provides the model call.
It is invoked once per judge call with this signature:

```python
def my_llm(prompt, *, model, max_tokens, response_schema):
    ...                                # call your model here
    return {"response_text": reply}
```

What it receives:

- `prompt` — the fully rendered prompt
- `model` — the `judge_model` name passed to `evaluate_text`
- `max_tokens`, `response_schema` — the completion cap and the pydantic model the reply
  must parse into. Pass the schema to your client's structured-output API, or just
  return JSON — parsing tolerates surrounding prose.
- `system_prompt=...` — passed only when one is set, so single-message backends keep
  working (fold it into the prompt if yours can't send one)

What to return: a dict with `"response_text"` (required). Optional keys — `tokens_used`,
`tokens_used_completion`, `tokens_used_reasoning`, `latency_seconds`, `model_version` —
are recorded per call under `token_usage` on the judgment.

A call that raises is retried twice, then fails: its criteria come back `N/A` and the
section lands in `failed_sections`.

With `fdllm` the adapter looks like:

```python
from fdllm import LLMMessage, get_caller

def my_llm(prompt, *, model, max_tokens, response_schema, system_prompt=None):
    messages = [LLMMessage(Role="user", Message=prompt)]
    if system_prompt:
        messages.insert(0, LLMMessage(Role="system", Message=system_prompt))
    out = get_caller(model).call(
        messages, max_tokens=max_tokens, response_schema=response_schema
    )
    return {
        "response_text": out.Message,
        "tokens_used": out.TokensUsed,
        "tokens_used_completion": out.TokensUsedCompletion,
        "tokens_used_reasoning": out.TokensUsedReasoning,
    }
```

## The three phases

`evaluate_text` is a thin wrapper over three steps you can call separately:

```python
from rubric_judge import JudgeItem, plan_judgment, execute_plan, assemble_judgment

item = JudgeItem(text=my_text, tags={"S1"}, placeholders={...}, context=...)

# 1. item + rubric -> the calls the judge model will see
plan = plan_judgment(item, rubric=rubric, ordered_tags=..., subject=...)
plan.calls           # JudgeCall list: prompt, schema, criterion_ids, section
plan.call_fingerprints()  # per-prompt hashes — drives resume

# 2. run the calls (or export them — see below)
answers = execute_plan(plan, llm=my_llm, judge_model="gpt-4.1-nano")

# 3. answers + rubric -> the same judgment evaluate_text writes
judgment = assemble_judgment(plan, answers)
```

### Resume

Store `plan.call_fingerprints()` on each judgment row as `call_shas` — one hash per
prompt, covering exactly what the model is asked. On a rerun, `rubric_judge.resume`
compares hashes call by call, so only changed prompts run again:

- `partition_calls(row, call_shas, plan)` → `(reusable, stale)` call keys
- `filter_calls(plan, stale)` — the plan restricted to the calls that need re-judging
- `merge_judgment(cached_row, fresh, plan, reusable)` — cached criterion rows plus
  freshly judged ones

```python
plan = plan_judgment(item, rubric=rubric)
fps = plan.call_fingerprints()
reusable, stale = partition_calls(cached_row, fps, plan)

sub = filter_calls(plan, stale)
fresh = assemble_judgment(sub, execute_plan(sub, llm=my_llm, judge_model=m))

judgment = merge_judgment(cached_row, fresh, plan, reusable)
judgment["call_shas"] = fps
```

Anything that changes a call's prompt — question text, placeholders, context, subject,
system prompt, gating — reruns just that call. In a `call_per: question` section each
criterion is its own call, so editing one question re-judges only that question.
Scoring-only edits (`scoring_rules`, `weight`) never invalidate; they're re-applied at
rescore.

## Ensemble voting

Judge with several models and majority-vote the verdicts:

```python
from rubric_judge import JudgeItem, evaluate_ensemble

judgment = evaluate_ensemble(
    JudgeItem(text=my_text, tags={"S1"}),
    rubric=rubric,
    judges={                            # judge model name -> its llm callable
        "claude-sonnet-5": llm_a,
        "gpt-5.6-terra": llm_b,
        "gemini-3.6-flash": llm_c,
    },
    subject="a story for a child",      # same options as evaluate_item
)
```

How rows merge:

- The merged `response` is the majority over judges that answered Yes/No. N/A rows —
  errors, gated-out — don't vote. Ties record `"tie": true` and `response: "N/A"`.
- Rows get `votes` (judge → verdict) and `agreement` (share agreeing with the verdict).
- Non-verdict fields come from the `primary` judge — the first judge by default — when
  it voted with the majority, else from the first judge that did.
- `evidence_list` rows don't vote; they pass through from the primary judge.
- `token_usage` records gain a `judge` key.

The separable form is `execute_ensemble` + `assemble_ensemble`, or `merge_ensemble` over
already-assembled judgments.

## Scoring

A section's score is `100 × (weighted yes / applicable)`; `N/A` and gated-out criteria
drop out of the denominator, a triggered `fail_condition` forces 0, and the final score
is the weighted mean of non-excluded sections. All scoring is offline — it reads the
stored judgment, never calls an LLM.

- `score_section(results)` — one section's score from its stored criterion rows.
- `score_judgment(judgment, ...)` — section scores plus the weighted final. Options:
  - `weights={section: w}` — override the stored section weights
  - `criterion_weights={id: w}` — override each criterion's stored weight
  - `evidence_rules={id: [...]}` — replace a criterion's stored `scoring_rules` (`[]`
    removes its points)
  - `extra_section_scorer` — callable returning `(name, score, {"weight": w, ...})` to
    inject a deterministic section (e.g. code-computed metrics)
- `deduction_score(judgment, ...)` — alternate family: start at 100 and deduct per
  evidence item (by severity), per "No" answer, and per triggered fail condition.
- `rescore_file(path, weights_path=None, **score_kwargs)` — rescore every row of a JSONL
  judgments file; `weights_path` is a `{Section: weight}` YAML map, other kwargs pass
  through to `score_judgment`.
